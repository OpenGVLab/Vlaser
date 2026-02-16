# train_backup_spatial_intelligence.py
"""
Data processing script for generating spatial intelligence QA pairs.
Saves images and JSONL files in a structured format based on the user's specified schema.
This version is updated for efficient, shuffled, multi-GPU processing.
"""

import logging
import os
from dataclasses import asdict
from typing import List, Dict, Any

from vllm import LLM, EngineArgs, SamplingParams
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor
import torch.distributed as dist
import json
import hydra

# ==================== 导入您的数据集类 ====================
from QA_dataset import AlohaAgileXFolderDataset

log = logging.getLogger(__name__)


def _parse_qa_response(response_text: str):
    """解析模型输出，提取问题和答案部分"""
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
            question = line[9:].strip()
        elif line.lower().startswith('answer:'):
            in_question = False
            in_answer = True
            answer = line[7:].strip()
        elif in_question and line:
            question += " " + line
        elif in_answer and line:
            answer += " " + line
    if not question or not answer:
        parts = response_text.split('\n\n')
        if len(parts) >= 2:
            question = parts[0].strip()
            answer = parts[1].strip()
        else:
            question = "What can you observe in this robot arm camera image and what does the task instruction tell us?"
            answer = response_text
    question = question.strip() or "What can you observe in this robot arm camera image and what does the task instruction tell us?"
    answer = answer.strip() or "I can see the robot arm camera image, but I need more context to provide a detailed answer."
    return question, answer


class DataProcessingAgent:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # 分布式/多卡设置
        self.multi_gpu = bool(cfg.get("multi_gpu", False))
        world_size = 1
        global_rank = 0
        if self.multi_gpu:
            global_rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            log.info(f"Multi-GPU mode. Global rank: {global_rank}, World size: {world_size}")
            torch.cuda.set_device(local_rank)

        self.main_rank = (not self.multi_gpu) or (global_rank == 0)

        self.log_freq = int(cfg.get("log_freq", 100))
        self.log_dir = str(cfg.log_dir)

        # vLLM 初始化
        engine_args = EngineArgs(
            model=cfg.resume_checkpoint_path,
            tensor_parallel_size=int(cfg.vllm_tp),
            pipeline_parallel_size=1,
            max_model_len=int(cfg.get("max_model_len", 4096)),
            max_num_seqs=int(cfg.get("max_num_seqs", 16)),
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            trust_remote_code=True,
        )
        self.llm = LLM(**asdict(engine_args))
        self.llm_processor = AutoProcessor.from_pretrained(
            cfg.resume_checkpoint_path,
            trust_remote_code=True
        )

        # DataLoader 初始化
        log.info("Initializing dataset from AlohaAgileXFolderDataset...")
        dataset = AlohaAgileXFolderDataset(cfg.data.train, train=True)
        
        sampler = None
        shuffle_in_loader = not self.multi_gpu

        if self.multi_gpu:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=True,
                seed=cfg.get("seed", 42)
            )
            log.info(f"Using DistributedSampler for rank {global_rank} with shuffling.")

        self.train_dataloader = DataLoader(
            dataset,
            batch_size=int(cfg.per_device_batch_size),
            pin_memory=True,
            num_workers=int(cfg.get("num_workers", 4)),
            sampler=sampler,
            shuffle=shuffle_in_loader
        )
        
        log.info(f"DataLoader initialized. Shuffle enabled via {'Sampler' if self.multi_gpu else 'DataLoader'}.")
        log.info(f"Per device batch size: {cfg.per_device_batch_size}")
        
        if self.multi_gpu:
            dist.barrier()

    def run(self):
        total_samples_processed_rank = 0

        output_root = self.cfg.log_dir
        images_output_root = os.path.join(output_root, "images")
        jsonl_output_root = os.path.join(output_root, "jsonl")
        input_images_root = self.cfg.data.train.images_root

        os.makedirs(images_output_root, exist_ok=True)
        os.makedirs(jsonl_output_root, exist_ok=True)

        if self.main_rank:
            log.info(f"Output root: {output_root}")
            log.info(f"Input images root (for relative path calc): {input_images_root}")
        
        if self.multi_gpu and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(0)

        for batch_idx, batch in enumerate(self.train_dataloader):
            processed_results = self._process_batch(
                batch,
                input_images_root,
                images_output_root,
                output_root  # ★ 将主输出目录传进去，用于计算相对路径
            )

            grouped_for_jsonl: Dict[str, List[Dict[str, Any]]] = {}
            for result in processed_results:
                action = result["action_name"]
                domain = result["domain_name"]
                jsonl_path = os.path.join(jsonl_output_root, "aloha-agilex", action, f"{domain}.jsonl")
                if jsonl_path not in grouped_for_jsonl:
                    grouped_for_jsonl[jsonl_path] = []
                grouped_for_jsonl[jsonl_path].append(result["sft_data"])

            for jsonl_path, sft_data_list in grouped_for_jsonl.items():
                os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    for sft_data in sft_data_list:
                        f.write(json.dumps(sft_data, ensure_ascii=False) + "\n")

            batch_size = len(processed_results)
            total_samples_processed_rank += batch_size

            if (batch_idx + 1) % self.log_freq == 0:
                log.info(f"[Rank {dist.get_rank() if self.multi_gpu else 0}] Processed batch {batch_idx + 1}, total samples on this rank: {total_samples_processed_rank}")

        log.info(f"-------------------------------- [Rank {dist.get_rank() if self.multi_gpu else 0}]")
        log.info(f"Finished processing all assigned samples. Total on this rank: {total_samples_processed_rank}")
        log.info(f"--------------------------------")

    # ==================== _process_batch 方法已更新，以生成您指定的JSON格式 ====================
    def _process_batch(self, batch, input_root: str, output_images_root: str, output_root: str):
        # 从batch中提取所有需要的信息
        images_tensors = batch["observation"]["image_primary"]
        prompt_instructions = [t.decode("utf-8") for t in batch["task"]["language_instruction"]]
        original_filepaths = batch["meta"]["filepath"]
        task_names = batch["meta"]["task"]
        domain_names = batch["meta"]["domain"]
        episode_names = batch["meta"]["episode"]
        # 获取原始任务指令 (可能为空字符串)
        original_task_instructions = [t.decode("utf-8") for t in batch["task"]["task_instruction"]]

        prompts = []
        image_pil_list = []
        structured_results = []

        for i in range(len(images_tensors)):
            # 这是喂给大模型的prompt，保持不变
            instruction_for_vlm = (
                'You are an AI assistant specializing in spatial intelligence analysis of robot arm camera images and task instructions.\n\n'
                f'Given the robot arm camera image and the task instruction: "{prompt_instructions[i]}"\n\n'
                'Please generate a concise question–answer pair focused on spatial reasoning about the robot’s environment '
                '(e.g., object locations, relative positions, clearances, reachable space, obstacles, grasp pose feasibility). '
                'Format with "Question:" and "Answer:".'
            )

            original_path = original_filepaths[i]
            try:
                relative_path = os.path.relpath(original_path, input_root)
            except ValueError:
                parts = original_path.split(os.sep)
                try:
                    idx = parts.index("aloha-agilex")
                    relative_path = os.path.join(*parts[idx:])
                except ValueError:
                    relative_path = os.path.basename(original_path)

            output_image_path = os.path.join(output_images_root, relative_path)
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

            arr = images_tensors[i][0].cpu().numpy()
            pil_image = Image.fromarray(arr, mode="RGB")
            if not os.path.exists(output_image_path):
                pil_image.save(output_image_path)

            placeholders = [{"type": "image", "image": output_image_path}]
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant specialized in analyzing robot arm camera images and task instructions."},
                {"role": "user", "content": [*placeholders, {"type": "text", "text": instruction_for_vlm}]},
            ]
            prompt = self.llm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            prompts.append(prompt)
            image_pil_list.append(pil_image)

            # ★★★ 构建与您示例完全一致的JSON结构 ★★★
            image_path_for_json = os.path.relpath(output_image_path, start=output_root).replace(os.sep, '/')

            sft_data = {
                "image": image_path_for_json,
                "width": pil_image.width,
                "height": pil_image.height,
                "task_name": task_names[i],
                "domain": domain_names[i],
                "episode": episode_names[i],
                "source": original_filepaths[i],
                "task_instruction": original_task_instructions[i],
            }

            structured_results.append({
                # 这部分信息仅用于在run()方法中对文件进行分组，不会写入最终的JSONL
                "action_name": task_names[i],
                "domain_name": domain_names[i],
                "sft_data": sft_data,
            })

        outputs = self.llm.generate(
            [{"prompt": p, "multi_modal_data": {"image": img}} for p, img in zip(prompts, image_pil_list)],
            sampling_params=SamplingParams(temperature=0.1, max_tokens=512),
        )

        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text.strip()
            question, answer = _parse_qa_response(response_text)
            # 将生成的问答对添加到sft_data字典中
            structured_results[i]["sft_data"]["conversations"] = [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer}
            ]

        return structured_results


def setup_distributed(cfg: DictConfig):
    if cfg.get("multi_gpu", False):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)
            log.info(f"Distributed training enabled. Rank {rank}/{world_size}")
        else:
            log.error("RANK, WORLD_SIZE, and LOCAL_RANK env vars must be set for multi-GPU.")
            exit(1)


@hydra.main(config_path=".", config_name="SpatialQA", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_distributed(cfg)
    agent = DataProcessingAgent(cfg)
    agent.run()


if __name__ == "__main__":
    # 依赖 (logging, hydra)
    import logging
    import hydra
    from omegaconf import DictConfig

    main()