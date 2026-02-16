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
import re

log = logging.getLogger(__name__)
def run_qwen_vl_batch(llm, processor, images, texts, images_dir, global_rank=0, total_samples_processed=0) -> list[dict]:
    prompts = []
    image_batches = []
    image_filenames = []
    results_list = []

    for idx, (image, text) in enumerate(zip(images, texts)):
        # 构建更具体的指令，让模型理解这是机器人任务场景
        instruction = f"""You are an AI assistant specializing in visual grounding analysis of robot arm camera images and task instructions.

Given the robot arm camera image and the task instruction: "{text}"

Please generate a natural question-answer pair specifically focused on visual grounding capabilities. The question should target object localization and could ask about:
- Object pointing: "Where is the [specific object] located in the image?"
- Multiple object pointing: "Point to all the [objects] visible in the scene."
- Object detection: "Can you locate and mark the boundaries of the [object]?"
- Multiple object detection: "Find and mark all instances of [objects] in the image."
- Spatial localization: "Where exactly can we find the [object] that the robot needs to interact with?"
- Region identification: "Which area of the image contains the [target object]?"
- Precise positioning: "What are the exact coordinates of the [object] in the image?"

You should either return a set of 2D points or a set of 2D bounding box(es) as the awswer for the specific visual grounding question.

Please respond in the following format:
Question: [Your grounding question here]
Answer: [Your detailed localization answer with specific coordinate information]

The point format could be like the following:
coordinate_patterns = [
        r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',  # (x, y)
        r'position\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # position x, y
        r'at\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # at x, y
        r'coordinates?\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # coordinate(s) x, y
    ]

The box format could be like the following:
box_patterns = [
        r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # [x1, y1, x2, y2]
        r'box\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # box x1, y1, x2, y2
        r'bounds?\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # bound(s) x1, y1, x2, y2
    ]

Make sure the question focuses on object localization, positioning, or boundary detection, and the answer provides precise location information about objects in the robot's visual field."""
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
        
        # 获取对应的图像尺寸信息
        sft_data = results_list[idx_1]
        image_width = sft_data["width"]
        image_height = sft_data["height"]
        
        # 解析grounding模型输出，提取问题和答案，并处理坐标信息
        question, answer = _parse_grounding_response(response_text, image_width, image_height)
        
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

def _parse_coordinates_from_text(text, image_width, image_height):
    """从自然语言描述中提取坐标信息并转换为0-1000的整数格式"""
    points = []
    boxes = []
    
    # 匹配各种坐标格式
    coordinate_patterns = [
        r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',  # (x, y)
        r'position\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # position x, y
        r'at\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # at x, y
        r'coordinates?\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # coordinate(s) x, y
    ]
    
    # 匹配边界框格式
    box_patterns = [
        r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # [x1, y1, x2, y2]
        r'box\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # box x1, y1, x2, y2
        r'bounds?\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',  # bound(s) x1, y1, x2, y2
    ]
    
    # 提取点坐标
    for pattern in coordinate_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            x, y = float(match[0]), float(match[1])
            # 假设原始坐标是相对坐标 (0-1) 或绝对像素坐标
            if x <= 1.0 and y <= 1.0:  # 相对坐标
                norm_x = int(x * 1000)
                norm_y = int(y * 1000)
            else:  # 绝对像素坐标
                norm_x = int((x / image_width) * 1000)
                norm_y = int((y / image_height) * 1000)
            
            # 确保坐标在0-1000范围内
            norm_x = max(0, min(1000, norm_x))
            norm_y = max(0, min(1000, norm_y))
            points.append([norm_x, norm_y])
    
    # 提取边界框
    for pattern in box_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            x1, y1, x2, y2 = float(match[0]), float(match[1]), float(match[2]), float(match[3])
            
            # 转换为0-1000的整数坐标
            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:  # 相对坐标
                norm_x1, norm_y1 = int(x1 * 1000), int(y1 * 1000)
                norm_x2, norm_y2 = int(x2 * 1000), int(y2 * 1000)
            else:  # 绝对像素坐标
                norm_x1, norm_y1 = int((x1 / image_width) * 1000), int((y1 / image_height) * 1000)
                norm_x2, norm_y2 = int((x2 / image_width) * 1000), int((y2 / image_height) * 1000)
            
            # 确保坐标在0-1000范围内
            norm_x1 = max(0, min(1000, norm_x1))
            norm_y1 = max(0, min(1000, norm_y1))
            norm_x2 = max(0, min(1000, norm_x2))
            norm_y2 = max(0, min(1000, norm_y2))
            
            boxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
    
    return points, boxes

def _parse_grounding_response(response_text, image_width, image_height):
    """解析grounding模型输出，提取问题和答案，并处理坐标信息"""
    # 先提取基本的问题和答案
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
            question = "Where are the objects located in this robot arm camera image?"
            answer = response_text
    
    # 清理和格式化
    question = question.strip()
    answer = answer.strip()
    
    # 确保问题和答案不为空
    if not question:
        question = "Where are the objects located in this robot arm camera image?"
    if not answer:
        answer = "I can see the robot arm camera image, but I need more context to provide location information."
    
    # 从答案中提取坐标信息
    points, boxes = _parse_coordinates_from_text(answer, image_width, image_height)
    
    # 如果找到了坐标信息，将其格式化并添加到答案中
    if points or boxes:
        # formatted_answer = answer
        if points:
            formatted_answer = f" <point>{points}</point>"
            question += " Your answer should be formatted as \"<point>[[x1, y1], [x2, y2], ...]</point>\". The point coordinates are normalized to integers between 0 and 1000. Return the answer in the point format directly."
        if boxes:
            formatted_answer = f" <box>{boxes}</box>"
            question += " Your answer should be formatted as \"<box>[[x1, y1, x2, y2], [x3, y3, x4, y4], ...]</box>\". The bounding box coordinates are normalized to integers between 0 and 1000. Return the answer in the bounding box format directly."
        answer = formatted_answer
    
    return question, answer

def _parse_qa_response(response_text):
    """解析模型输出，提取问题和答案部分（保持向后兼容）"""
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
                # gpu_memory_utilization=0.95,
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
                # gpu_memory_utilization=0.95,
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