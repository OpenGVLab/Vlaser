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
import time
from tqdm import tqdm

log = logging.getLogger(__name__)

# ===== Ceph 相关（仅在需要时才 import） =====
def build_ceph_client(conf_path="petreloss.conf"):
    from petrel_client.client import Client
    return Client(conf_path)

def pil_loader(img_str):
    import io
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')

def load_image_by_path_ceph(client, fn):
    img_value_str = client.get(fn)
    img = pil_loader(img_value_str)
    return img

# ===== 通用图片加载（本地/ceph 自动切换） =====
def load_image(img_path: str, smart_resize=None, from_ceph=False, ceph_client=None) -> Image.Image:
    if from_ceph:
        assert ceph_client is not None, "Ceph client must be provided for remote loading!"
        img = load_image_by_path_ceph(ceph_client, img_path)
    else:
        img = Image.open(img_path).convert("RGB")
    if smart_resize:
        w, h = img.size
        rh, rw = smart_resize(h, w, max_pixels=1024 * 28 * 28)
        img = img.resize((rw, rh))
    return img

# ===== Prompt 模板 =====
# ===== QA数据评分模板 =====
prompt_template_general_qa_score = """You are an expert in evaluating question-answer pairs for robot arm camera images and task instructions. Please evaluate the quality of the generated QA pair based on **Relevance, Informativeness, and Clarity**.

You will receive the following inputs:  
--- The robot arm camera image, the task instruction, the generated question, and the generated answer.

Carefully evaluate whether the question and answer are relevant to the image and task, informative, and clearly expressed.

## Scoring Criteria (1–3):

**1 – Poor**  
The QA pair is irrelevant, uninformative, or unclear.  
- The question or answer doesn't relate to the image or task instruction.
- The answer is vague, generic, or provides no useful information.
- The question is poorly formulated or doesn't make sense.
**Examples:**  
- Question: "What is this?" Answer: "It's an image." (too generic)
- Question: "How to cook?" Answer: "Use a pan." (irrelevant to robot task)
- Question: "What color?" Answer: "Color." (unclear and uninformative)

**2 – Fair**  
The QA pair is somewhat relevant but lacks depth or clarity.  
- The question relates to the image/task but is too simple or generic.
- The answer provides basic information but lacks detail or specificity.
- Some aspects are unclear or could be better explained.
**Examples:**  
- Question: "What objects are in the image?" Answer: "There are some objects." (too vague)
- Question: "What is the robot doing?" Answer: "The robot is moving." (lacks detail)
- Question: "How to complete the task?" Answer: "Follow the instructions." (not specific enough)

**3 – Good**  
The QA pair is highly relevant, informative, and clearly expressed.  
- The question is specific, well-formulated, and directly relates to the image and task.
- The answer provides detailed, accurate, and useful information.
- Both question and answer are clear and help understand the robot's environment and task.
**Examples:**  
- Question: "What objects are visible in the robot arm's workspace and which one should be manipulated based on the task instruction?" Answer: "I can see a red cup, a blue bowl, and a green plate on the table. According to the task instruction to 'pick up the cup', the robot should focus on the red cup located in the center of the workspace."
- Question: "What obstacles might prevent the robot from completing the task?" Answer: "The workspace appears clear, but the target object is positioned near the edge of the table, which may require careful positioning to avoid knocking it over during manipulation."

### Output Requirement:
You must return only a single integer score from 1 to 3. Do not include any explanation, labels, or extra content.

Question: {question}
Answer: {answer}
"""

prompt_template_spatial_qa_score = """You are an expert in evaluating spatial intelligence question-answer pairs for robot arm camera images and task instructions. Please evaluate the quality based on **Spatial Reasoning Accuracy, Detail Level, and Relevance**.

You will receive the following inputs:  
--- The robot arm camera image, the task instruction, the generated spatial question, and the generated spatial answer.

Carefully evaluate whether the question targets spatial reasoning, and whether the answer provides accurate and detailed spatial information.

## Scoring Criteria (1–3):

**1 – Poor**  
The spatial QA pair lacks spatial reasoning focus or provides incorrect/vague spatial information.  
- The question doesn't target spatial aspects (counting, relationships, distances, orientation, etc.).
- The answer provides incorrect spatial information or is too vague.
- The spatial reasoning is flawed or irrelevant to the task.
**Examples:**  
- Question: "What is in the image?" Answer: "Objects." (not spatial)
- Question: "How many objects?" Answer: "Some." (vague, no specific count)
- Question: "Where is object A?" Answer: "It's there." (no spatial detail)
- Question: "What is the distance?" Answer: "Close." (not quantitative)

**2 – Fair**  
The spatial QA pair addresses spatial reasoning but lacks precision or detail.  
- The question targets spatial aspects but could be more specific.
- The answer provides basic spatial information but lacks quantitative details or precision.
- Some spatial relationships are mentioned but not fully explained.
**Examples:**  
- Question: "How many cups are there?" Answer: "There are a few cups." (not specific count)
- Question: "Where is the red object relative to the blue one?" Answer: "The red one is near the blue one." (lacks specific direction/distance)
- Question: "What is the spatial arrangement?" Answer: "Objects are arranged on the table." (too general)

**3 – Good**  
The spatial QA pair demonstrates strong spatial reasoning with precise, detailed information.  
- The question clearly targets specific spatial aspects (counting, relationships, distances, orientation, geometry, etc.).
- The answer provides accurate, quantitative, and detailed spatial information.
- The spatial reasoning is relevant to the robot task and helps understand the workspace layout.
**Examples:**  
- Question: "How many objects are visible in the scene and what are their types?" Answer: "I can count 5 objects: 2 red cups positioned on the left side of the table, 1 blue bowl in the center, and 2 green plates arranged on the right side."
- Question: "What is the relative position of the target object compared to the robot arm's current position?" Answer: "The target object (red cup) is located approximately 30cm to the right and 15cm forward from the robot arm's current end-effector position, requiring a diagonal reach motion."
- Question: "Which objects are within the robot's reachable workspace?" Answer: "Based on the robot arm's reach, the blue bowl (center, 25cm away) and the left red cup (20cm away) are within reach. The rightmost green plate (45cm away) is outside the immediate reachable zone."

### Output Requirement:
You must return only a single integer score from 1 to 3. Do not include any explanation, labels, or extra content.

Question: {question}
Answer: {answer}
"""

prompt_template_grounding_qa_score = """You are an expert in evaluating visual grounding question-answer pairs for robot arm camera images and task instructions. Please evaluate the quality based on **Grounding Accuracy, Coordinate Validity, and Localization Precision**.

You will receive the following inputs:  
--- The robot arm camera image, the task instruction, the generated grounding question, and the generated grounding answer (which should contain coordinates or bounding boxes).

Carefully evaluate whether the question targets object localization, and whether the answer provides valid and accurate coordinate information.

## Scoring Criteria (1–3):

**1 – Poor**  
The grounding QA pair lacks localization focus or provides invalid/incorrect coordinate information.  
- The question doesn't target object localization, pointing, or detection.
- The answer lacks coordinate information or contains invalid coordinates (out of bounds, wrong format).
- The coordinates don't match the described object location.
**Examples:**  
- Question: "What is in the image?" Answer: "A cup." (no coordinates)
- Question: "Where is the object?" Answer: "It's on the table." (no coordinates)
- Question: "Point to the cup." Answer: "The cup is at (1500, 2000)." (coordinates out of bounds for normalized 0-1000 range)
- Question: "Find the object." Answer: "Box: [100, 200, 50, 300]." (invalid box format, x2 < x1)

**2 – Fair**  
The grounding QA pair addresses localization but coordinates are imprecise or partially valid.  
- The question targets localization but could be more specific.
- The answer contains coordinates but they may be approximate, slightly off, or lack precision.
- Some coordinate information is present but formatting could be improved.
**Examples:**  
- Question: "Where is the object?" Answer: "The object is at position (500, 300)." (single point, but should specify if multiple objects exist)
- Question: "Point to all cups." Answer: "Cups are at (200, 150) and (600, 400)." (coordinates present but may not be precise)
- Question: "Mark the boundaries." Answer: "Box: [100, 100, 400, 300]." (valid box but may not accurately bound the object)

**3 – Good**  
The grounding QA pair demonstrates precise localization with valid and accurate coordinate information.  
- The question clearly targets object localization, pointing, detection, or precise positioning.
- The answer provides valid, accurate coordinates (points or bounding boxes) in the correct format.
- Coordinates are normalized to 0-1000 range and accurately represent object locations.
- Multiple objects are properly localized if applicable.
**Examples:**  
- Question: "Where is the red cup located in the image? Provide coordinates." Answer: "The red cup is located at point (450, 320). <point>[[450, 320]]</point>"
- Question: "Point to all instances of cups visible in the scene." Answer: "I can locate 2 cups: the red cup at (450, 320) and the blue cup at (680, 250). <point>[[450, 320], [680, 250]]</point>"
- Question: "Can you locate and mark the boundaries of the target object?" Answer: "The target object (red cup) is bounded by box [380, 280, 520, 360]. <box>[[380, 280, 520, 360]]</box>"
- Question: "Find and mark all instances of plates in the image." Answer: "I found 2 plates: plate 1 at box [100, 200, 250, 350], plate 2 at box [600, 180, 750, 330]. <box>[[100, 200, 250, 350], [600, 180, 750, 330]]</box>"

### Output Requirement:
You must return only a single integer score from 1 to 3. Do not include any explanation, labels, or extra content.

Question: {question}
Answer: {answer}
"""

# ===== Prompt模板字典 =====
PROMPT_TEMPLATES = {
    "general_qa_score": prompt_template_general_qa_score,
    "spatial_qa_score": prompt_template_spatial_qa_score,
    "grounding_qa_score": prompt_template_grounding_qa_score,
}

# ====== 评测批量主流程 ======
def run_qwen_vl_batch_score(llm, processor, batch_samples: list[dict], from_ceph=False, ceph_client=None) -> list[dict]:
    """
    处理QA数据批次评分
    """
    prompts = []
    image_batches = []
    results_list = []

    # QA数据：单图像 + question + answer + task_instruction
    for sample in batch_samples:
        # task_instruction = sample.get("task_instruction", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        img_paths = sample["img_paths"]  # 单图像路径列表
        qa_type = sample.get("qa_type", "general")  # general/spatial/grounding
        
        sample_result = {}
        
        # 根据qa_type选择对应的评分模板
        if qa_type == "general":
            template_key = "general_qa_score"
        elif qa_type == "spatial":
            template_key = "spatial_qa_score"
        elif qa_type == "grounding":
            template_key = "grounding_qa_score"
        else:
            template_key = "general_qa_score"  # 默认
        
        template = PROMPT_TEMPLATES[template_key]
        prompt_text = template.format(
            question=question,
            answer=answer
        )
        
        placeholders = [{"type": "image", "image": p} for p in img_paths]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [*placeholders, {"type": "text", "text": prompt_text}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        # 批量加载图片（单图像）
        image_batches.append([load_image(p, from_ceph=from_ceph, ceph_client=ceph_client) for p in img_paths])
        sample_result[template_key] = None
        results_list.append(sample_result)

    outputs = llm.generate(
        [{"prompt": p, "multi_modal_data": {"image": imgs}} for p, imgs in zip(prompts, image_batches)],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=512),
    )

    assert len(outputs) == len(prompts)
    idx = 0
    for result in results_list:
        # QA数据只有一个评分
        for key in result.keys():
            result[key] = outputs[idx].outputs[0].text.strip()
            idx += 1

    return results_list

# ====== 辅助函数：解析评分 ======
def parse_score(score_text: str) -> int:
    """从模型输出中解析评分（1-3）"""
    # 尝试提取数字
    match = re.search(r'\b([123])\b', score_text.strip())
    if match:
        return int(match.group(1))
    # 如果找不到，尝试其他模式
    score_text_lower = score_text.lower().strip()
    if '3' in score_text or 'three' in score_text_lower or 'good' in score_text_lower:
        return 3
    elif '2' in score_text or 'two' in score_text_lower or 'fair' in score_text_lower:
        return 2
    elif '1' in score_text or 'one' in score_text_lower or 'poor' in score_text_lower:
        return 1
    return 2  # 默认返回2（中等）

# ====== 辅助函数：从conversations提取QA ======
def extract_qa_from_conversations(conversations: list, item: dict = None) -> tuple:
    """从conversations中提取question和answer
    
    Args:
        conversations: 对话列表
        item: 可选的完整数据项，用于提取task_instruction等其他字段
    
    Returns:
        (question, answer, task_instruction)
    """
    question = None
    answer = None
    task_instruction = None
    
    # 尝试从item中提取task_instruction（如果存在）
    if item:
        # 尝试从item的task_instruction字段获取
        task_instruction = item.get("task_instruction") or item.get("task") or item.get("instruction")
    
    for conv in conversations:
        role = conv.get("from", "")
        value = conv.get("value", "")
        
        if role == "human":
            # 移除<image>标签
            text = value.replace("<image>", "").strip()
            if not question:
                question = text
            # 尝试提取task instruction（通常在第一个human消息中）
            if not task_instruction and text:
                task_instruction = text
        elif role == "gpt":
            if not answer:
                answer = value.strip()
    
    return question, answer, task_instruction

# ====== 主分布式批处理 ======
def batch_process(input_folder: str, image_root: str, output_root: str, batch_size: int = 4, 
                  from_ceph=False, qa_type="general", min_score=2):
    """
    两种输入形态：
    A) 目录：保持原有"按文件"切分的并行处理逻辑（每个rank处理一组文件，输出与输入同名，追加写）
    B) 单文件：新增"按行号范围"切分（每个rank处理不同行区间），各自写 part，rank0 收尾合并并清理 part

    断点续跑：
    - 目录模式：沿用原逻辑，读取已存在的输出文件以跳过已处理 id
    - 单文件模式：分片级断点（每个 rank 仅根据自己的 .part{rank}.jsonl 跳过已处理 id）
    
    参数：
    - qa_type: "general", "spatial", or "grounding" (默认"general")
    - min_score: 最低分数阈值，低于此分数的数据将被过滤（1-3，默认2）
    """
    try:
        from qwen_vl_utils import smart_resize
    except ImportError:
        smart_resize = None

    # 可选初始化 Ceph client
    ceph_client = build_ceph_client() if from_ceph else None

    # SLURM 并行信息
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))

    os.makedirs(output_root, exist_ok=True)

    # ====== 公共：模型与 processor 初始化（每个 rank 一次） ======
    # 注意：AutoProcessor 不需要 device；vLLM GPU 绑定用 CUDA_VISIBLE_DEVICES/调度控制
    model_path = "Qwen2.5-VL-7B-Instruct"
    device = rank  # 每个进程用不同 GPU
    processor = AutoProcessor.from_pretrained(model_path, device=device)

    # QA数据使用单图像
    if world_size == 1:
        engine_args = EngineArgs(
        model=model_path,
            max_model_len=4096 * 4,
            max_num_seqs=1024,
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enforce_eager=True,
        )
    else:
        engine_args = EngineArgs(
        model=model_path,
            max_model_len=4096 * 4,
            max_num_seqs=1024,
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enforce_eager=True,
        device=f"cuda:{device % 8}",
    )
    llm = LLM(**asdict(engine_args))

    # ====== 情况 A：输入为目录（文件级并行） ======
    if os.path.isdir(input_folder):
        input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jsonl')]

        total_files = len(input_files)
        base_files_per_rank = total_files // world_size
        extra_files = total_files % world_size

        if rank < extra_files:
            files_this_rank = base_files_per_rank + 1
            start_idx = rank * files_this_rank
        else:
            files_this_rank = base_files_per_rank
            start_idx = extra_files * (base_files_per_rank + 1) + (rank - extra_files) * base_files_per_rank

        end_idx = start_idx + files_this_rank
        files_to_process = input_files[start_idx:end_idx]

        processed_ids = set()

        for input_file in files_to_process:
            output_jsonl_path = os.path.join(output_root, os.path.basename(input_file))  # 与输入文件同名

            # 载入输入样本
            with open(input_file, "r", encoding="utf-8") as f:
                lines = []
                for l in f:
                    try:
                        lines.append(json.loads(l))
                    except Exception as e:
                        print(f"[rank {rank}] skip broken json in {input_file}: {e}")

            # 为不同分数创建不同的输出文件
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file_score1 = os.path.join(output_root, f"{base_name}_score1.jsonl")
            output_file_score2 = os.path.join(output_root, f"{base_name}_score2.jsonl")
            output_file_score3 = os.path.join(output_root, f"{base_name}_score3.jsonl")
            
            # resume：读已有输出以收集已处理 id（从所有三个分数文件中）
            for output_file in [output_file_score1, output_file_score2, output_file_score3]:
                if os.path.exists(output_file):
                    with open(output_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if "id" in data:
                                    processed_ids.add(data["id"])
                            except:
                                continue
            
            # 统计计数器
            count_score1 = 0
            count_score2 = 0
            count_score3 = 0
            
            # 打开所有输出文件（追加模式）
            fout_score1 = open(output_file_score1, "a", encoding="utf-8")
            fout_score2 = open(output_file_score2, "a", encoding="utf-8")
            fout_score3 = open(output_file_score3, "a", encoding="utf-8")
            
            buffer = []
            for item in tqdm(lines, desc=f"[rank {rank}] {os.path.basename(input_file)}"):
                item_id = item.get("id", None)
                if item_id is None or item_id in processed_ids:
                    continue

                # 基础合法性过滤
                # if any(x < 512 for x in item.get("width", [])) or any(y < 512 for y in item.get("height", [])):
                #     continue
                if isinstance(item["image"], str):
                    item["image"] = [item["image"]]
                # QA数据：单图像
                if "image" not in item or len(item["image"]) != 1:
                    continue

                # 构建图片路径
                if from_ceph:
                    img_paths = [os.path.join(image_root, name) for name in item["image"]]
                else:
                    img_paths = [os.path.join(image_root, name) for name in item["image"]]
                    if not all(os.path.exists(p) for p in img_paths):
                        continue

                # QA数据：从conversations提取question和answer
                question, answer, task_instruction = extract_qa_from_conversations(item.get("conversations", []), item)
                if not question or not answer:
                    continue
                buffer.append({
                    "item": item, 
                    "question": question,
                    "answer": answer,
                    "task_instruction": task_instruction or "",
                    "img_paths": img_paths,
                    "qa_type": qa_type
                })

                if len(buffer) >= batch_size:
                    try:
                        batch_input = [
                            {
                                "question": s["question"],
                                "answer": s["answer"],
                                "task_instruction": s["task_instruction"],
                                "img_paths": s["img_paths"],
                                "qa_type": s["qa_type"]
                            } for s in buffer
                        ]
                        
                        batch_results = run_qwen_vl_batch_score(
                            llm, processor, batch_input,
                            from_ceph=from_ceph, ceph_client=ceph_client
                        )
                        
                        for s, scores in zip(buffer, batch_results):
                            # 解析评分并过滤
                            score_key = list(scores.keys())[0] if scores else None
                            if score_key:
                                score_text = scores[score_key]
                                score = parse_score(score_text)
                                s["item"][score_key] = score_text
                                s["item"]["parsed_score"] = score
                                
                                # 过滤低分数据
                                if score < min_score:
                                    continue
                                
                                # 根据分数分类保存
                                if score == 1:
                                    fout_score1.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                                    count_score1 += 1
                                elif score == 2:
                                    fout_score2.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                                    count_score2 += 1
                                elif score == 3:
                                    fout_score3.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                                    count_score3 += 1
                                
                                processed_ids.add(s["item"]["id"])
                    except Exception as e:
                        print(f"[rank {rank}] Error during batch process: {e}")
                    buffer.clear()

            # 尾批
            if buffer:
                try:
                    batch_input = [
                        {
                            "question": s["question"],
                            "answer": s["answer"],
                            "task_instruction": s["task_instruction"],
                            "img_paths": s["img_paths"],
                            "qa_type": s["qa_type"]
                        } for s in buffer
                    ]
                    
                    batch_results = run_qwen_vl_batch_score(
                        llm, processor, batch_input,
                        from_ceph=from_ceph, ceph_client=ceph_client
                    )
                    
                    for s, scores in zip(buffer, batch_results):
                        # 解析评分并过滤
                        score_key = list(scores.keys())[0] if scores else None
                        if score_key:
                            score_text = scores[score_key]
                            score = parse_score(score_text)
                            s["item"][score_key] = score_text
                            s["item"]["parsed_score"] = score
                            
                            # 过滤低分数据
                            if score < min_score:
                                continue
                            
                            # 根据分数分类保存
                            if score == 1:
                                fout_score1.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                                count_score1 += 1
                            elif score == 2:
                                fout_score2.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                                count_score2 += 1
                            elif score == 3:
                                fout_score3.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                                count_score3 += 1
                            
                            processed_ids.add(s["item"]["id"])
                except Exception as e:
                    print(f"[rank {rank}] Error during final batch process: {e}")
            
            # 关闭文件
            fout_score1.close()
            fout_score2.close()
            fout_score3.close()
            
            # 输出统计信息
            print(f"[rank {rank}] File {os.path.basename(input_file)} statistics:")
            print(f"  Score 1: {count_score1} samples")
            print(f"  Score 2: {count_score2} samples")
            print(f"  Score 3: {count_score3} samples")
            print(f"  Total: {count_score1 + count_score2 + count_score3} samples")

        return  # 目录模式结束

    # ====== 情况 B：输入为单文件（行级并行 + 自动合并清理） ======
    if not os.path.isfile(input_folder):
        raise ValueError(f"Input path not found: {input_folder}")

    input_file = input_folder
    base_name = os.path.basename(input_file)

    # 每个 rank 写自己的分片输出，避免并发写
    part_out_score1 = os.path.join(output_root, f"{base_name}.part{rank:03d}_score1.jsonl")
    part_out_score2 = os.path.join(output_root, f"{base_name}.part{rank:03d}_score2.jsonl")
    part_out_score3 = os.path.join(output_root, f"{base_name}.part{rank:03d}_score3.jsonl")
    done_flag = os.path.join(output_root, f"{base_name}.part{rank:03d}.done")
    
    # 统计计数器
    count_score1 = 0
    count_score2 = 0
    count_score3 = 0

    # 读所有原始行，按行号切分
    with open(input_file, "r", encoding="utf-8") as f:
        lines_raw = f.readlines()

    total_lines = len(lines_raw)
    base_per_rank = total_lines // world_size
    extras = total_lines % world_size

    if rank < extras:
        start_idx = rank * (base_per_rank + 1)
        end_idx = start_idx + base_per_rank + 1
    else:
        start_idx = extras * (base_per_rank + 1) + (rank - extras) * base_per_rank
        end_idx = start_idx + base_per_rank

    # 解析本 rank 负责的行
    my_rows = []
    for i in range(start_idx, end_idx):
        try:
            my_rows.append(json.loads(lines_raw[i]))
        except Exception as e:
            print(f"[rank {rank}] skip broken json at line {i}: {e}")

    # 分片级 resume - 从所有分数文件中收集已处理的id
    processed_ids = set()
    for part_out in [part_out_score1, part_out_score2, part_out_score3]:
        if os.path.exists(part_out):
            with open(part_out, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "id" in data:
                            processed_ids.add(data["id"])
                    except:
                        continue

    # 打开所有分数文件
    fout_score1 = open(part_out_score1, "a", encoding="utf-8")
    fout_score2 = open(part_out_score2, "a", encoding="utf-8")
    fout_score3 = open(part_out_score3, "a", encoding="utf-8")
    
    buffer = []
    for item in tqdm(my_rows, desc=f"[rank {rank}] {base_name}[{start_idx}:{end_idx})"):
        item_id = item.get("id", None)
        if item_id is None or item_id in processed_ids:
            continue

        # 基础合法性过滤
        # if any(x < 512 for x in item.get("width", [])) or any(y < 512 for y in item.get("height", [])):
        #     continue
        
        # QA数据：单图像
        if isinstance(item["image"], str):
            item["image"] = [item["image"]]
        if "image" not in item or len(item["image"]) != 1:
            continue

        # 构建图片路径
        if from_ceph:
            img_paths = [os.path.join(image_root, name) for name in item["image"]]
        else:
            img_paths = [os.path.join(image_root, name) for name in item["image"]]
            if not all(os.path.exists(p) for p in img_paths):
                continue

        # QA数据：从conversations提取question和answer
        question, answer, task_instruction = extract_qa_from_conversations(item.get("conversations", []), item)
        if not question or not answer:
            continue
        buffer.append({
            "item": item, 
            "question": question,
            "answer": answer,
            "task_instruction": task_instruction or "",
            "img_paths": img_paths,
            "qa_type": qa_type
        })

        if len(buffer) >= batch_size:
            try:
                batch_input = [
                    {
                        "question": s["question"],
                        "answer": s["answer"],
                        "task_instruction": s["task_instruction"],
                        "img_paths": s["img_paths"],
                        "qa_type": s["qa_type"]
                    } for s in buffer
                ]
                
                batch_results = run_qwen_vl_batch_score(
                    llm, processor, batch_input,
                    from_ceph=from_ceph, ceph_client=ceph_client
                )
                
                for s, scores in zip(buffer, batch_results):
                    # 解析评分并过滤
                    score_key = list(scores.keys())[0] if scores else None
                    if score_key:
                        score_text = scores[score_key]
                        score = parse_score(score_text)
                        s["item"][score_key] = score_text
                        s["item"]["parsed_score"] = score
                        
                        # 过滤低分数据
                        if score < min_score:
                            continue
                        
                        # 根据分数分类保存
                        if score == 1:
                            fout_score1.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                            count_score1 += 1
                        elif score == 2:
                            fout_score2.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                            count_score2 += 1
                        elif score == 3:
                            fout_score3.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                            count_score3 += 1
                        
                        processed_ids.add(s["item"]["id"])
            except Exception as e:
                print(f"[rank {rank}] Error during batch process: {e}")
            buffer.clear()

    # 尾批
    if buffer:
        try:
            batch_input = [
                {
                    "question": s["question"],
                    "answer": s["answer"],
                    "task_instruction": s["task_instruction"],
                    "img_paths": s["img_paths"],
                    "qa_type": s["qa_type"]
                } for s in buffer
            ]
            
            batch_results = run_qwen_vl_batch_score(
                llm, processor, batch_input,
                from_ceph=from_ceph, ceph_client=ceph_client
            )
            
            for s, scores in zip(buffer, batch_results):
                # 解析评分并过滤
                score_key = list(scores.keys())[0] if scores else None
                if score_key:
                    score_text = scores[score_key]
                    score = parse_score(score_text)
                    s["item"][score_key] = score_text
                    s["item"]["parsed_score"] = score
                    
                    # 过滤低分数据
                    if score < min_score:
                        continue
                    
                    # 根据分数分类保存
                    if score == 1:
                        fout_score1.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                        count_score1 += 1
                    elif score == 2:
                        fout_score2.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                        count_score2 += 1
                    elif score == 3:
                        fout_score3.write(json.dumps(s["item"], ensure_ascii=False) + "\n")
                        count_score3 += 1
                    
                    processed_ids.add(s["item"]["id"])
        except Exception as e:
            print(f"[rank {rank}] Error during final batch process: {e}")
    
    # 关闭文件
    fout_score1.close()
    fout_score2.close()
    fout_score3.close()
    
    # 输出统计信息
    print(f"[rank {rank}] Part [{start_idx}:{end_idx}) statistics:")
    print(f"  Score 1: {count_score1} samples")
    print(f"  Score 2: {count_score2} samples")
    print(f"  Score 3: {count_score3} samples")
    print(f"  Total: {count_score1 + count_score2 + count_score3} samples")

    # === 单文件模式：标记完成 ===
    with open(done_flag, "w", encoding="utf-8") as _f:
        _f.write("")

    # === rank0 负责合并与清理 ===
    if rank == 0:
        # 等待所有 rank 完成
        expected_done = [os.path.join(output_root, f"{base_name}.part{r:03d}.done") for r in range(world_size)]
        while True:
            if all(os.path.exists(p) for p in expected_done):
                break
            time.sleep(5)

        # 为不同分数创建最终输出文件
        merged_out_score1 = os.path.join(output_root, f"{base_name}_score1.jsonl")
        merged_out_score2 = os.path.join(output_root, f"{base_name}_score2.jsonl")
        merged_out_score3 = os.path.join(output_root, f"{base_name}_score3.jsonl")
        
        # 统计总数
        total_count_score1 = 0
        total_count_score2 = 0
        total_count_score3 = 0
        
        # 合并每个分数的文件（按 rank 顺序），基于 id 去重
        for score, merged_out in [(1, merged_out_score1), (2, merged_out_score2), (3, merged_out_score3)]:
            seen_ids = set()
            with open(merged_out, "w", encoding="utf-8") as fout:
                for r in range(world_size):
                    part_path = os.path.join(output_root, f"{base_name}.part{r:03d}_score{score}.jsonl")
                    if not os.path.exists(part_path):
                        continue
                    with open(part_path, "r", encoding="utf-8") as fin:
                        for line in fin:
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            _id = obj.get("id", None)
                            if _id is None or _id in seen_ids:
                                continue
                            seen_ids.add(_id)
                            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            if score == 1:
                                total_count_score1 += 1
                            elif score == 2:
                                total_count_score2 += 1
                            elif score == 3:
                                total_count_score3 += 1

        # 输出总体统计信息
        print(f"\n[rank 0] Final statistics for {base_name}:")
        print(f"  Score 1: {total_count_score1} samples")
        print(f"  Score 2: {total_count_score2} samples")
        print(f"  Score 3: {total_count_score3} samples")
        print(f"  Total: {total_count_score1 + total_count_score2 + total_count_score3} samples")

        # 清理所有 part 与 done
        for r in range(world_size):
            for score in [1, 2, 3]:
                part_p = os.path.join(output_root, f"{base_name}.part{r:03d}_score{score}.jsonl")
                if os.path.exists(part_p):
                    try:
                        os.remove(part_p)
                    except Exception as e:
                        print(f"[rank 0] remove {part_p} failed: {e}")
            done_p = os.path.join(output_root, f"{base_name}.part{r:03d}.done")
            if os.path.exists(done_p):
                try:
                    os.remove(done_p)
                except Exception as e:
                    print(f"[rank 0] remove {done_p} failed: {e}")


# ====== 命令行入口 ======
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", 
        default="bridge/qa_datas_grounding/merged.jsonl", 
        help="Path to input folder containing JSONL files or a single JSONL file")
    parser.add_argument("--image_root", 
        default="bridge/qa_datas_grounding/images/", 
        help="Root directory of images (ignored for ceph)")
    parser.add_argument("--output_root", 
        default="bridge/qa_datas_grounding/filtered", 
        help="Root directory for output JSONL files")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default=4)")
    parser.add_argument("--from_ceph", default=True, action="store_true", help="If set, read images from Ceph using petrel_client")
    parser.add_argument("--qa_type", type=str, default="general", choices=["general", "spatial", "grounding"],
                        help="QA type: 'general', 'spatial', or 'grounding' (default=general)")
    parser.add_argument("--min_score", type=int, default=2, choices=[1, 2, 3],
                        help="Minimum score threshold for filtering (1-3, default=2). Data with score < min_score will be filtered out.")
    args = parser.parse_args()
    batch_process(
        args.input_folder, 
        args.image_root, 
        args.output_root, 
        args.batch_size, 
        from_ceph=args.from_ceph,
        qa_type=args.qa_type,
        min_score=args.min_score
    )
