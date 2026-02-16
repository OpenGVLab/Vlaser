# -*- coding: utf-8 -*-
"""
Data processing script for generating visual grounding QA pairs.
This script is updated to run efficiently on multiple GPUs with shuffled data output.

Per-line JSONL schema (exactly as requested):
{
  "image": "images/aloha-agilex/<DATASET_ROOT>/<task_name>/<domain>/frames/<episode>/<filename>.jpg",
  "width": <int>,
  "height": <int>,
  "task_name": "<task_name>",
  "domain": "<domain>",
  "episode": "<episode>",
  "source": "<abs/original/image/path>",
  "task_instruction": "<episode-level instruction or empty string>",
  "conversations": [
    {"from": "human", "value": "<image>\\n<Question text>"},
    {"from": "gpt",   "value": " <box>... </box>  or  <point>... </point>"}
  ]
}
"""

import os
import re
import json
import logging
from dataclasses import asdict

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
# ==================== 新增 IMPORT ====================
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor
from vllm import LLM, EngineArgs, SamplingParams
import hydra

# ==================== 数据集类 ====================
from QA_dataset import AlohaAgileXFolderDataset

log = logging.getLogger(__name__)


# ==================== 坐标解析 (保持不变) ====================
def _parse_coordinates_from_text(text, image_width, image_height):
    """从自然语言描述中提取坐标信息并转换为 0-1000 的整数格式"""
    points, boxes = [], []

    coordinate_patterns = [
        r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
        r'position\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',
        r'at\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',
        r'coordinates?\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',
    ]
    box_patterns = [
        r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',
        r'box\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',
        r'bounds?\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)',
    ]

    # points
    for p in coordinate_patterns:
        for m in re.findall(p, text, re.IGNORECASE):
            x, y = float(m[0]), float(m[1])
            nx = int(x * 1000) if x <= 1.0 else int((x / image_width) * 1000)
            ny = int(y * 1000) if y <= 1.0 else int((y / image_height) * 1000)
            nx = max(0, min(1000, nx))
            ny = max(0, min(1000, ny))
            points.append([nx, ny])

    # boxes
    for p in box_patterns:
        for m in re.findall(p, text, re.IGNORECASE):
            x1, y1, x2, y2 = map(float, m)
            if max(x1, y1, x2, y2) <= 1.0:
                coords = [int(v * 1000) for v in (x1, y1, x2, y2)]
            else:
                coords = [
                    int((x1 / image_width) * 1000),
                    int((y1 / image_height) * 1000),
                    int((x2 / image_width) * 1000),
                    int((y2 / image_height) * 1000),
                ]
            coords = [max(0, min(1000, c)) for c in coords]
            boxes.append(coords)

    return points, boxes


def _parse_grounding_response(response_text, image_width, image_height):
    """解析 grounding 模型输出，提取 Question/Answer，并结构化坐标"""
    lines = response_text.split('\n')
    q, a = "", ""
    in_q, in_a = False, False

    for line in lines:
        s = line.strip()
        if s.lower().startswith('question:'):
            in_q, in_a = True, False
            q = s[9:].strip()
        elif s.lower().startswith('answer:'):
            in_q, in_a = False, True
            a = s[7:].strip()
        elif in_q and s:
            q += " " + s
        elif in_a and s:
            a += " " + s

    if not q or not a:
        parts = response_text.split('\n\n')
        if len(parts) >= 2:
            q, a = parts[0].strip(), parts[1].strip()
        else:
            q = "Where are the objects located in this robot arm camera image?"
            a = response_text

    q = q.strip() or "Where are the objects located in this robot arm camera image?"
    a = a.strip() or "I can see the robot arm camera image, but I need more context to provide location information."

    points, boxes = _parse_coordinates_from_text(a, image_width, image_height)

    # 优先 box，其次 point
    if boxes:
        a = f' <box>{boxes}</box>'
        q += ' Your answer should be formatted as "<box>[[x1, y1, x2, y2], ...]</box>". The bounding box coordinates are normalized to integers between 0 and 1000.'
    elif points:
        a = f' <point>{points}</point>'
        q += ' Your answer should be formatted as "<point>[[x1, y1], ...]</point>". The point coordinates are normalized to integers between 0 and 1000.'

    a = a.replace("'", '"')  # JSON 兼容
    return q, a


# ==================== 主处理类 ====================
class DataProcessingAgent:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # 分布式环境设置
        self.multi_gpu = bool(cfg.get("multi_gpu", False))
        world_size = 1
        global_rank = 0
        if self.multi_gpu:
            global_rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            log.info(f"Multi-GPU mode. Global rank: {global_rank}, World size: {world_size}")
            torch.cuda.set_device(local_rank)

        # vLLM 引擎
        engine_args = EngineArgs(
            model=cfg.resume_checkpoint_path,
            max_model_len=cfg.get("max_model_len", 4096 * 4),
            max_num_seqs=cfg.get("max_num_seqs", 1024),
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=cfg.get("vllm_tp", 1),
            pipeline_parallel_size=1,
            enforce_eager=True,
        )
        self.llm = LLM(**asdict(engine_args))
        self.llm_processor = AutoProcessor.from_pretrained(cfg.resume_checkpoint_path)

        # ==================== DataLoader 初始化 (★ 已修改) ====================
        log.info("Initializing dataset from AlohaAgileXFolderDataset...")
        dataset = AlohaAgileXFolderDataset(cfg.data.train, train=True)
        
        sampler = None
        shuffle_in_loader = not self.multi_gpu

        if self.multi_gpu:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=True,  # ★ 在Sampler中启用打乱
                seed=cfg.get("seed", 42)
            )
            log.info(f"Using DistributedSampler for rank {global_rank} with shuffling.")

        self.loader = DataLoader(
            dataset,
            batch_size=cfg.per_device_batch_size,
            pin_memory=True,
            num_workers=cfg.get("num_workers", 4),
            sampler=sampler,
            shuffle=shuffle_in_loader
        )
        log.info(f"DataLoader initialized. Shuffle enabled via {'Sampler' if self.multi_gpu else 'DataLoader'}.")
        # ==================== 修改结束 ====================

        # 输出根
        self.output_root = cfg.log_dir
        self.jsonl_root_out = os.path.join(self.output_root, "jsonl", "aloha-agilex")
        os.makedirs(self.jsonl_root_out, exist_ok=True)

        self.input_images_root = cfg.data.train.images_root
        self.dataset_root_name = os.path.basename(os.path.normpath(self.input_images_root)) or "aloha-agilex"

        self.log_freq = cfg.get("log_freq", 100)

        if self.multi_gpu:
            dist.barrier()

    def _make_prompt(self, text: str, image_path: str):
        """构造多模态对话 prompt"""
        instruction = (
            'You are an AI assistant specializing in visual grounding analysis of robot arm camera images and task instructions.\n\n'
            f'Given the robot arm camera image and the task instruction: "{text}"\n\n'
            'Please generate a natural question-answer pair specifically focused on visual grounding capabilities. '
            'You may ask about object pointing/multiple pointing/detection/bounding boxes/region identification/precise positioning with coordinates.\n'
            'Return coordinates as points or bounding boxes when applicable.'
        )
        placeholders = [{"type": "image", "image": image_path}]
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specialized in analyzing robot arm camera images and task instructions."},
            {"role": "user", "content": [*placeholders, {"type": "text", "text": instruction}]},
        ]
        return self.llm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def run(self):
        total_samples_on_rank = 0
        if self.multi_gpu and hasattr(self.loader.sampler, 'set_epoch'):
            self.loader.sampler.set_epoch(0)

        for bidx, batch in enumerate(self.loader):
            results = self._process_batch(batch)

            # 分桶写入
            grouped = {}
            for item in results:
                task = item["task_name"]
                domain = item["domain"]
                jsonl_path = os.path.join(self.jsonl_root_out, task, f"{domain}.jsonl")
                grouped.setdefault(jsonl_path, []).append(item)

            for path, lines in grouped.items():
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    for obj in lines:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            total_samples_on_rank += len(results)
            if (bidx + 1) % self.log_freq == 0:
                rank = dist.get_rank() if self.multi_gpu else 0
                log.info(f"[Rank {rank}] Processed batch {bidx+1}, total samples on this rank: {total_samples_on_rank}")

        rank = dist.get_rank() if self.multi_gpu else 0
        log.info(f"--------------------------------\n[Rank {rank}] Finished processing all assigned samples. Total: {total_samples_on_rank}\n--------------------------------")

    def _process_batch(self, batch):
        imgs = batch["observation"]["image_primary"]
        lang_instrs = [t.decode("utf-8") for t in batch["task"]["language_instruction"]]
        # ★ 从 batch['meta'] 中获取更可靠的任务名和其他元数据 ★
        task_names = batch["meta"]["task"]
        ep_instrs = [t.decode("utf-8") for t in batch["task"]["task_instruction"]]

        src_paths = batch["meta"]["filepath"]
        domains = batch["meta"]["domain"]
        episodes = batch["meta"]["episode"]

        prompts, pil_list, meta_list = [], [], []

        for i in range(len(imgs)):
            src_abs = src_paths[i]
            task = task_names[i]
            domain = domains[i]
            episode = episodes[i]

            # 输出图片相对路径
            rel_image_path_part = os.path.relpath(src_abs, self.input_images_root)
            rel_image = os.path.join("images", "aloha-agilex", rel_image_path_part)
            
            out_image_path = os.path.join(self.output_root, rel_image)
            os.makedirs(os.path.dirname(out_image_path), exist_ok=True)

            pil = Image.fromarray(imgs[i][0].cpu().numpy())
            if not os.path.exists(out_image_path):
                pil.save(out_image_path)

            prompt = self._make_prompt(lang_instrs[i], out_image_path)

            prompts.append(prompt)
            pil_list.append(pil)
            meta_list.append({
                "image": rel_image.replace(os.sep, "/"), # 确保存储为UNIX风格路径
                "width": pil.width,
                "height": pil.height,
                "task_name": task,
                "domain": domain,
                "episode": episode,
                "source": src_abs,
                "task_instruction": ep_instrs[i],
            })

        # vLLM 批量推理
        outputs = self.llm.generate(
            [{"prompt": p, "multi_modal_data": {"image": img}} for p, img in zip(prompts, pil_list)],
            sampling_params=SamplingParams(temperature=0.1, max_tokens=512),
        )

        results = []
        for i, out in enumerate(outputs):
            resp = out.outputs[0].text.strip()
            w, h = meta_list[i]["width"], meta_list[i]["height"]
            q, a = _parse_grounding_response(resp, w, h)

            # ★ 直接使用 meta_list 中的字典作为基础，保证格式一致 ★
            obj = meta_list[i]
            obj["conversations"] = [
                {"from": "human", "value": f"<image>\n{q}"},
                {"from": "gpt", "value": a}
            ]
            results.append(obj)

        return results


# ==================== 分布式初始化 & 入口 ====================
def setup_distributed(cfg: DictConfig):
    if cfg.get("multi_gpu", False):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)
            log.info(f"Distributed enabled. Rank {rank}/{world_size}")
        else:
            log.error("RANK, WORLD_SIZE, and LOCAL_RANK env vars must be set for multi-GPU.")
            exit(1)


@hydra.main(config_path=".", config_name="GroundingQA", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Rank %(rank)s] - %(message)s')
    rank_filter = logging.Filter()
    rank_filter.filter = lambda record: setattr(record, 'rank', dist.get_rank() if dist.is_initialized() else 0) or True
    logging.getLogger().addFilter(rank_filter)

    log.info("=" * 50 + f"\nConfiguration loaded:\n{OmegaConf.to_yaml(cfg)}\n" + "=" * 50)

    setup_distributed(cfg)

    try:
        agent = DataProcessingAgent(cfg)
        agent.run()
        log.info("Script finished successfully.")
    except Exception as e:
        log.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()