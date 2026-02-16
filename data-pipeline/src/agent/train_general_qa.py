"""
Main training agent. Using torch.compile and bfloat16 by default. Optionally (Q)LoRA.

"""

import logging
import os
from collections import deque
import multiprocessing as mp
from functools import partial
from dataclasses import asdict
from vllm import LLM, EngineArgs, SamplingParams
import bitsandbytes as bnb
import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor
import torch.distributed as dist
import wandb
from src.agent.dataset import TorchRLDSInterleavedDataset
from src.agent.model_averaging import ModelAveraging
from src.model.vla.pizero import PiZero
from src.model.vla.processing import VLAProcessor
from src.utils.decorator import main_rank_only
from src.utils.metric import get_action_accuracy
from src.utils.monitor import (
    MainRankFilter,
    Timer,
    log_allocated_gpu_memory,
    log_execution_time,
)
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions
import json

log = logging.getLogger(__name__)
def run_qwen_vl_batch(llm, processor, images, texts, images_dir, global_rank=0, total_samples_processed=0) -> list[dict]:
    prompts = []
    image_batches = []
    image_filenames = []
    results_list = []

    for idx, (image, text) in enumerate(zip(images, texts)):
        # 构建更具体的指令，让模型理解这是机器人任务场景
        instruction = f"""You are an AI assistant analyzing robot arm camera images and task instructions. 

Given the robot arm camera image and the task instruction: "{text}"

Please generate a natural question-answer pair about this image and task. The question should be open-ended and could ask about:
- Objects visible in the image
- The robot arm's current state or position
- How to accomplish the given task
- What obstacles or challenges might exist
- Safety considerations for the task
- Or any other relevant aspect of the image and task

Please respond in the following format:
Question: [Your question here]
Answer: [Your detailed answer here]

Make sure the question is natural and the answer is informative and helpful for understanding the robot arm's environment and task."""
        # 保存图片到指定路径
        global_idx = total_samples_processed + idx
        image_filename = f"rank_{global_rank:02d}_{global_idx:04d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        image = Image.fromarray(image[0].cpu().numpy())
        image.save(image_path)
        placeholders = [{"type": "image", "image": image_path}]
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specialized in analyzing robot arm camera images and task instructions."},
            {"role": "user", "content": [*placeholders, {"type": "text", "text": instruction}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompts.append(prompt)
        image_batches.append(image)

        width, height = image.size[0], image.size[1]
        sft_data = {
            "id": global_idx,
            "image": f"rank_{global_rank:02d}/{image_filename}",
            "width": width,
            "height": height,
        }
        
        results_list.append(sft_data)


    outputs = llm.generate(
        [{"prompt": p, "multi_modal_data": {"image": imgs}} for p, imgs in zip(prompts, image_batches)],
        sampling_params=SamplingParams(temperature=0.1, max_tokens=512),
    )

    assert len(outputs) == len(prompts)
    
    for idx_1 in range(len(outputs)):
        response_text = outputs[idx_1].outputs[0].text.strip()
        
        # 解析模型输出，提取问题和答案
        question, answer = _parse_qa_response(response_text)
        sft_data = results_list[idx_1]
        sft_data.update({
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        })
        results_list[idx_1] = sft_data
    return results_list

def _parse_qa_response(response_text):
    """解析模型输出，提取问题和答案部分"""
    # 尝试从输出中提取Question和Answer
    lines = response_text.split('\n')
    question = ""
    answer = ""
    
    in_question = False
    in_answer = False
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith('question:'):
            in_question = True
            in_answer = False
            question = line[9:].strip()  # 去掉"Question:"前缀
        elif line.lower().startswith('answer:'):
            in_question = False
            in_answer = True
            answer = line[7:].strip()   # 去掉"Answer:"前缀
        elif in_question and line:
            question += " " + line
        elif in_answer and line:
            answer += " " + line
    
    # 如果没有找到标准格式，尝试其他解析方式
    if not question or not answer:
        # 尝试用换行符分割
        parts = response_text.split('\n\n')
        if len(parts) >= 2:
            question = parts[0].strip()
            answer = parts[1].strip()
        else:
            # 如果都失败了，使用原始文本作为答案，生成默认问题
            question = "What can you observe in this robot arm camera image and what does the task instruction tell us?"
            answer = response_text
    
    # 清理和格式化
    question = question.strip()
    answer = answer.strip()
    
    # 确保问题和答案不为空
    if not question:
        question = "What can you observe in this robot arm camera image and what does the task instruction tell us?"
    if not answer:
        answer = "I can see the robot arm camera image, but I need more context to provide a detailed answer."
    
    return question, answer

class TrainAgent:
    def __init__(self, cfg):
        # device setup
        self.cfg = cfg
        self.gpu_id = cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.multi_gpu = cfg.multi_gpu
        world_size = 1  # single gpu
        if self.multi_gpu:
            global_rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            world_size = int(os.environ["WORLD_SIZE"])
            group_rank = int(os.environ["GROUP_RANK"])
            log.info(
                f"GPU local ID: {self.gpu_id}. Global rank: {global_rank}. Local rank: {local_rank}. Local world size: {local_world_size}. World size: {world_size}. Group rank: {group_rank}"
            )
            for i in range(torch.cuda.device_count()):
                log.info(
                    f"Local rank: {local_rank}, GPU UUID: {torch.cuda.get_device_properties(i).uuid}"
                )
        self.main_rank = not self.multi_gpu or global_rank == 0
        log.addFilter(MainRankFilter(main_rank=self.main_rank))

        # logging
        self.use_wandb = cfg.get("wandb", False) and self.main_rank
        if self.use_wandb:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                id=self.wandb_id if hasattr(self, "wandb_id") else None,
                resume="allow",  # not using resume_from
            )
        self.debug = cfg.get("debug", False)
        self.save_model_freq = int(cfg.save_model_freq)
        self.save_model_start = int(cfg.get("save_model_start", 0))
        self.log_freq = cfg.log_freq
        self.log_dir = cfg.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # training params
        self.n_updates = int(cfg.n_updates)
        self.max_grad_norm = cfg.max_grad_norm
        self.use_amp = cfg.get("use_amp", True)
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32
        self.use_torch_compile = cfg.get("use_torch_compile", True)
        
        # 数据转换参数
        self.max_samples = cfg.get("max_samples", None)  # 如果设置，限制处理的样本数量
        self.samples_per_file = cfg.get("samples_per_file", 10000)  # 每个jsonl文件保存的样本数量
        self.use_multiprocessing = cfg.get("use_multiprocessing", True)  # 是否使用多进程
        self.max_processes = cfg.get("max_processes", None)  # 最大进程数，None表示使用batch_size
        self.force_single_process = cfg.get("force_single_process", False)  # 强制使用单进程
        self.images_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        if world_size == 1:
            engine_args = EngineArgs(
                model=cfg.resume_checkpoint_path,
                max_model_len=4096 * 4,
                max_num_seqs=1024,
                limit_mm_per_prompt={"image": 1},
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                enforce_eager=True,
                # device="cuda",
                # gpu_memory_utilization=0.6,
            )
        else:
            engine_args = EngineArgs(
                model=cfg.resume_checkpoint_path,
                max_model_len=4096 * 4,
                max_num_seqs=1024,
                limit_mm_per_prompt={"image": 1},
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                enforce_eager=True,
                device=f"cuda:{self.gpu_id % 8}",
                # gpu_memory_utilization=0.6,
            )
        
        self.llm = LLM(**asdict(engine_args))
        self.llm_processor = AutoProcessor.from_pretrained(cfg.resume_checkpoint_path)
        dist.barrier()
        # determine batch size and gradient accumulation steps
        self.grad_accumulation_steps = max(
            cfg.global_batch_size // cfg.per_device_batch_size // world_size, 1
        )
        actual_global_batch_size = (
            cfg.per_device_batch_size * self.grad_accumulation_steps * world_size
        )

        # dataloader --- spawn one for each rank, num_workers=0
        self.train_dataloader = DataLoader(
            TorchRLDSInterleavedDataset(cfg.data.train, train=True).dataset,
            batch_size=cfg.per_device_batch_size,
            pin_memory=True,
        )
        self.run_eval = cfg.data.get("val", False)
        log.info(f"Total number of gradient updates: {self.n_updates}")
        log.info(f"Global batch size: {actual_global_batch_size}")
        log.info(f"Per device batch size: {cfg.per_device_batch_size}")
        log.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")

        ########### Input processing ###########

        # flow matching timestep sampling
        self.flow_sampling = cfg.get("flow_sampling", "beta")
        assert self.flow_sampling in [
            "uniform",
            "beta",
        ], f"Invalid flow matching timestep sampling mode: {self.flow_sampling}"
        if self.flow_sampling == "beta":
            flow_alpha = cfg.get("flow_alpha", 1.5)
            flow_beta = cfg.get("flow_beta", 1)
            self.flow_t_max = 1 - cfg.get("flow_sig_min", 0.001)
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        # processor --- assume paligemma for now
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

    def run(self):
        timer = Timer()
        cnt_batch = 0 if not hasattr(self, "cnt_batch") else self.cnt_batch
        cnt_update = (
            0 if not hasattr(self, "cnt_update") else self.cnt_update
        )  # resume training if loaded checkpoint
        loss_deque = deque(maxlen=self.grad_accumulation_steps)
        new_eval_from_last_log = False
        
        # 数据转换相关变量
        total_samples_processed = 0
        current_file_index = 0
        current_file_samples = 0
        
        # 获取全局rank用于文件名
        global_rank = 0
        if self.multi_gpu:
            global_rank = int(os.environ["RANK"])
        
        image_dir = os.path.join(self.images_dir, f"rank_{global_rank:02d}")
        os.makedirs(image_dir, exist_ok=True)
        if current_file_samples >= self.samples_per_file:
            current_file_index += 1
            current_file_samples = 0
            log.info(f"Created new jsonl file: {global_rank}_{current_file_index}.jsonl")

        for batch in self.train_dataloader:
            """
            batch: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
            observation: 'image_primary' (torch.Size([bsz, 1, H, W, 3], uint8), 'image_wrist', 'timestep' (torch.Size([bsz, 1])), 'pad_mask_dict', 'timestep_pad_mask', 'task_completed' (torch.Size([bsz, window, 4]), 'proprio' (torch.Size([bsz, window, proprio_dim])
            task: 'language_instruction', 'pad_mask_dict', 'image_primary', 'image_wrist', 'timestep' (torch.Size([bsz]))
            action (torch.Size([bsz, window, horizon, action_dim], float32)
            action_pad_mask (torch.Size([bsz, window, horizon, action_dim]))
            """
            # 数据格式转换和保存，而不是训练
            # 保存图片到指定路径
            images = batch["observation"]["image_primary"]
            texts = [
                text.decode("utf-8") for text in batch["task"]["language_instruction"]
            ]
            sample_data_list = run_qwen_vl_batch(self.llm, self.llm_processor, images, texts, image_dir, global_rank, total_samples_processed)
            
            # 使用多进程并行处理样本
            batch_size = len(sample_data_list)
            # 更新计数
            total_samples_processed += batch_size
            current_file_samples += batch_size

            # 检查是否需要创建新文件
            if current_file_samples >= self.samples_per_file:
                current_file_index += 1
                current_file_samples = batch_size  # 重置为当前batch的大小
                log.info(f"Created new jsonl file: {global_rank}_{current_file_index}.jsonl")

            with open(os.path.join(self.log_dir, f"rank_{global_rank}_{current_file_index}.jsonl"), "a", encoding="utf-8") as f:
                for sample_data in sample_data_list:
                    f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
            dist.barrier()
            # 检查是否达到总样本数量限制
            if self.max_samples and total_samples_processed >= self.max_samples:
                log.info(f"Reached target sample count ({self.max_samples}), stopping data processing")
                log.info(f"Total files created: {current_file_index + 1}")
                log.info(f"Total samples processed: {total_samples_processed}")
                return
            # 计数
            cnt_batch += 1
            # 记录进度
            if cnt_batch % self.log_freq == 0:
                log.info(f"Processed batch {cnt_batch}, total samples: {total_samples_processed}, current file: {global_rank}_{current_file_index}.jsonl ({current_file_samples}/{self.samples_per_file})")
        
        log.info(f"--------------------------------")
        log.info(f"Finished processing all samples")
        log.info(f"Total samples processed: {total_samples_processed}")
        log.info(f"Total files created: {current_file_index + 1}")
        log.info(f"Total batches processed: {cnt_batch}")
        log.info(f"--------------------------------")