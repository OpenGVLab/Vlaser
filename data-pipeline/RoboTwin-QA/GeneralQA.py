"""
Main data-generation agent (single process).
Use vLLM tensor-parallel (TP) to shard a 32B VLM across multiple GPUs.
DO NOT launch with torchrun.

Launch example:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train_backup_general_prompt.py --config daqi_config.yaml
"""

import os
import json
import logging
from dataclasses import asdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, EngineArgs, SamplingParams
import wandb

# ---- 你的数据集适配器（已支持 task_instruction 注入与 meta 信息）----
# from src.agent.dataset import TorchRLDSInterleavedDataset
from QA_dataset import AlohaAgileXFolderDataset as TorchRLDSInterleavedDataset

# 其余依赖（占位/日志工具）
from src.model.vla.processing import VLAProcessor
from src.utils.monitor import MainRankFilter, Timer

log = logging.getLogger(__name__)


# ------------------ 工具函数 ------------------
def _parse_qa_response(response_text: str):
    lines = response_text.split("\n")
    question, answer = "", ""
    in_question = False
    in_answer = False
    for line in lines:
        line = line.strip()
        if line.lower().startswith("question:"):
            in_question, in_answer = True, False
            question = line[9:].strip()
        elif line.lower().startswith("answer:"):
            in_question, in_answer = False, True
            answer = line[7:].strip()
        elif in_question and line:
            question += " " + line
        elif in_answer and line:
            answer += " " + line
    if not question or not answer:
        parts = response_text.split("\n\n")
        if len(parts) >= 2:
            question = parts[0].strip()
            answer = parts[1].strip()
        else:
            question = "What can you observe in this robot arm camera image and what does the task instruction tell us?"
            answer = response_text
    question = question.strip() or "What can you observe in this robot arm camera image and what does the task instruction tell us?"
    answer = answer.strip() or "I can see the robot arm camera image, but I need more context to provide a detailed answer."
    return question, answer


def _split_meta_batch(meta_batch: Dict[str, List]):
    """DataLoader 默认把每个字段 collate 成 list，这里还原为 per-sample dict 列表。"""
    if not isinstance(meta_batch, dict):
        return meta_batch
    keys = list(meta_batch.keys())
    if not keys:
        return []
    n = len(meta_batch[keys[0]])
    out = []
    for i in range(n):
        out.append({k: meta_batch[k][i] for k in keys})
    return out


def _make_instruction_text(task_name: str, task_instruction: str) -> str:
    return (
        "You are an AI assistant analyzing robot arm camera images and task instructions.\n\n"
        f'Task name: "{task_name}"\n'
        f'Task instruction: "{task_instruction}"\n\n'
        "Please generate a natural question-answer pair about this image and task. The question should be open-ended and could ask about:\n"
        "- Objects visible in the image\n"
        "- The robot arm's current state or position\n"
        "- How to accomplish the given task\n"
        "- What obstacles or challenges might exist\n"
        "- Safety considerations for the task\n"
        "- Or any other relevant aspect of the image and task\n\n"
        "Respond in the format:\n"
        "Question: ...\n"
        "Answer: ...\n"
        "Ensure the question is natural and the answer is informative."
    )


### MODIFICATION START ###
# 1. 修改了函数签名，增加了 task_instructions 参数
def run_vlm_batch(
    llm,
    processor,
    images,                 # [B, 1, H, W, 3] uint8
    metas: List[dict],      # 每条样本的 meta 字典
    task_instructions: List[str], # 新增参数
    images_dir: str,        # log_dir/images
    dataset_root_parent: str,
    log_dir: str,
):
### MODIFICATION END ###
    """对一个 batch 的图像/文本，调用多模态大模型生成 QA，并把图片副本 + SFT 行返回。"""
    prompts = []
    image_batches = []
    results_list = []

    ### MODIFICATION START ###
    # 2. 修改了循环，同时遍历 metas, images, 和新的 task_instructions
    for meta, image, task_instruction in zip(metas, images, task_instructions):
    ### MODIFICATION END ###
        # 取关键信息
        src_path = meta.get("filepath")
        task_name = meta.get("task")
        domain = meta.get("domain")
        episode = meta.get("episode")
        
        ### MODIFICATION START ###
        # 3. 删除了原来错误获取 task_instruction 的行
        # task_instruction = meta.get("task_instruction") or "" # <- This line is now deleted
        ### MODIFICATION END ###

        # === 目标图片保存路径：log_dir/images/aloha-agilex/.../frame_xxx.jpg
        rel_relpath = os.path.relpath(src_path, start=dataset_root_parent)  # 以 <.../dataset> 的父目录为基准
        dst_full = os.path.join(images_dir, rel_relpath)
        os.makedirs(os.path.dirname(dst_full), exist_ok=True)

        # 保存“处理后的”图片（DataLoader 已 resize 到 cfg.data.train.resize_to）
        pil_img = Image.fromarray(image[0].cpu().numpy())
        if not os.path.exists(dst_full):
            pil_img.save(dst_full)

        # 生成 prompt
        instruction = _make_instruction_text(task_name, task_instruction)
        placeholders = [{"type": "image", "image": dst_full}]
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant specialized in analyzing robot arm camera images and task instructions.",
            },
            {"role": "user", "content": [*placeholders, {"type": "text", "text": instruction}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompts.append(prompt)
        image_batches.append(pil_img)

        width, height = pil_img.size
        image_rel_for_json = os.path.relpath(dst_full, start=log_dir).replace(os.sep, "/")  # e.g. images/aloha-agilex/...

        results_list.append(
            {
                "image": image_rel_for_json,
                "width": width,
                "height": height,
                "task_name": task_name,
                "domain": domain,
                "episode": episode,
                "source": src_path,
                "task_instruction": task_instruction,
            }
        )

    # 推理（把每个样本的 prompt + 对应图片喂给 vLLM）
    outputs = llm.generate(
        [{"prompt": p, "multi_modal_data": {"image": img}} for p, img in zip(prompts, image_batches)],
        sampling_params=SamplingParams(temperature=0.1, max_tokens=512),
    )
    assert len(outputs) == len(prompts)

    # 解析 QA，拼到结果里
    for i, out in enumerate(outputs):
        response_text = out.outputs[0].text.strip()
        q, a = _parse_qa_response(response_text)
        results_list[i].update({"conversations": [{"from": "human", "value": f"<image>\n{q}"}, {"from": "gpt", "value": a}]})

    return results_list


# ------------------ 主类 ------------------
class TrainAgent:
    def __init__(self, cfg):
        self.cfg = cfg

        # ===== 强制走单进程 =====
        world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size_env > 1:
            raise RuntimeError(
                "This script uses single-process vLLM tensor-parallel. "
                "Do NOT launch with torchrun. Use: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train_backup_general_prompt.py --config daqi_config.yaml"
            )

        # ===== 日志 / 目录 =====
        logging.getLogger().setLevel(logging.INFO)
        log.addFilter(MainRankFilter(main_rank=True))

        self.use_wandb = bool(cfg.get("wandb", False))
        if self.use_wandb:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
            )

        self.log_freq = int(cfg.get("log_freq", 50))
        self.log_dir = cfg.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # 图片根：在 log_dir 下保留原数据集层级
        self.images_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        # JSONL 根目录（按 (task, domain) 分桶单文件）
        self.jsonl_root = os.path.join(self.log_dir, "jsonl")
        os.makedirs(self.jsonl_root, exist_ok=True)

        # ===== 生成规模 =====
        self.max_samples = cfg.get("max_samples", None)          # None 表示不限制
        self.per_device_batch_size = int(cfg.get("per_device_batch_size", 1))

        # ===== vLLM 引擎：用张量并行把 32B 切到多卡 =====
        tp = int(cfg.get("vllm_tp", max(1, torch.cuda.device_count())))
        engine_args = EngineArgs(
            model=cfg.resume_checkpoint_path,
            max_model_len=4096 * 4,
            max_num_seqs=128,  # 并发数，适当保守；如显存紧张可再降
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            enforce_eager=True,
        )
        self.llm = LLM(**asdict(engine_args))
        self.llm_processor = AutoProcessor.from_pretrained(cfg.resume_checkpoint_path)

        # ===== DataLoader =====
        ds_wrap = TorchRLDSInterleavedDataset(cfg.data.train, train=True)
        base_dataset = getattr(ds_wrap, "dataset", ds_wrap)  # 兼容是否有 .dataset 属性
        self.train_dataloader = DataLoader(
            base_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

        # ===== 处理器（与 vLLM 推理无关，但保留）=====
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path, padding_side="right")
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )

        # ===== 保存路径辅助 =====
        # 例如 images_root=/home/.../dataset/aloha-agilex
        # 我们以其上一级目录作为 relpath 起点，确保 images/aloha-agilex/... 的结构
        self.dataset_images_root = cfg.data.train.images_root
        self.dataset_root_parent = os.path.dirname(self.dataset_images_root)
        self.dataset_name = os.path.basename(self.dataset_images_root)  # 'aloha-agilex'

        # 每个 (task, domain) 一把句柄
        self._bucket_files: Dict[tuple, "io.TextIOWrapper"] = {}

    def _bucket_jsonl_path(self, task_name: str, domain: str) -> str:
        # log_dir/jsonl/aloha-agilex/<task_name>/<domain>.jsonl
        p = os.path.join(self.jsonl_root, self.dataset_name, task_name, f"{domain}.jsonl")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def _write_bucket_lines(self, samples: List[dict]):
        """把样本按 (task_name, domain) 分桶，分别写入对应的单个 JSONL 文件。"""
        for s in samples:
            task_name = s.get("task_name", "unknown_task")
            domain = s.get("domain", "unknown_domain")
            key = (task_name, domain)
            if key not in self._bucket_files:
                path = self._bucket_jsonl_path(task_name, domain)
                self._bucket_files[key] = open(path, "a", encoding="utf-8")
            fh = self._bucket_files[key]
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    def close(self):
        for fh in self._bucket_files.values():
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
        self._bucket_files.clear()

    def run(self):
        timer = Timer()
        cnt_batch = 0
        total_samples_processed = 0

        # rank 目录（单进程固定 rank_00，仅用于图片二级目录）
        image_dir = os.path.join(self.images_dir, self.dataset_name)
        os.makedirs(image_dir, exist_ok=True)

        try:
            for batch in self.train_dataloader:
                # 取图像 & meta（meta 是 dict-of-lists，需拆回 per-sample 列表）
                images = batch["observation"]["image_primary"]  # [B, 1, H, W, 3] uint8
                metas = _split_meta_batch(batch["meta"])        # List[dict]

                ### MODIFICATION START ###
                # 4. 从 batch["task"] 中正确地提取 task_instruction 列表
                task_instructions = [t.decode("utf-8") for t in batch["task"]["task_instruction"]]
                ### MODIFICATION END ###

                # 调用多模态大模型
                ### MODIFICATION START ###
                # 5. 将提取出的指令列表传递给 run_vlm_batch 函数
                sample_data_list = run_vlm_batch(
                    self.llm,
                    self.llm_processor,
                    images,
                    metas,
                    task_instructions, # 新增的参数
                    image_dir,
                    self.dataset_root_parent,
                    self.log_dir,
                )
                ### MODIFICATION END ###

                bsz = len(sample_data_list)
                total_samples_processed += bsz

                # 写入各自 (task, domain) 的 JSONL
                self._write_bucket_lines(sample_data_list)

                # 达到目标条数则退出
                if self.max_samples and total_samples_processed >= self.max_samples:
                    log.info(f"[rank 00] Reached target sample count ({self.max_samples}), stopping")
                    log.info(f"[rank 00] Total samples processed: {total_samples_processed}")
                    break

                cnt_batch += 1
                if cnt_batch % self.log_freq == 0:
                    log.info(f"[rank 00] Processed batch {cnt_batch}, total samples: {total_samples_processed}")

        finally:
            self.close()
            log.info("-------- Summary --------")
            log.info(f"[rank 00] Finished. Total samples processed: {total_samples_processed}")
            log.info("-------------------------")


# ------------------ 入口 ------------------
@hydra.main(config_path=".", config_name="GeneralQA", version_base=None)
def main(cfg: OmegaConf):
    """Hydra-managed entry point."""
    log.info("Initializing data generation agent...")
    agent = TrainAgent(cfg)
    log.info("Starting data generation run...")
    agent.run()
    log.info("Data generation finished.")


if __name__ == "__main__":
    # 依赖 (logging, hydra)
    import logging
    import hydra
    from omegaconf import OmegaConf

    main()