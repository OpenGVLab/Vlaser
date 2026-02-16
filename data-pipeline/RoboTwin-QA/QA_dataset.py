# -*- coding: utf-8 -*-
import os
import glob
import csv
import json
import hashlib
import random
from typing import Dict, List, Tuple, Optional, Set

from PIL import Image, ImageFile
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断图片

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ========== 可选的指令映射（CSV）==========
def _load_instruction_map(csv_path: Optional[str]) -> Dict[tuple, str]:
    """
    CSV 支持两种格式：
      1) filepath,instruction   （优先级高）
      2) task,instruction
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    mp: Dict[tuple, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and "filepath" in header[0].lower():
            for row in reader:
                if not row:
                    continue
                path = os.path.normpath(row[0].strip())
                instr = ",".join(row[1:]).strip()
                mp[("path", path)] = instr
        else:
            for row in reader:
                if not row:
                    continue
                task = row[0].strip()
                instr = ",".join(row[1:]).strip()
                mp[("task", task)] = instr
    return mp


# ========== 路径解析辅助 ==========
def _infer_task_from_path(path: str) -> str:
    """
    期望结构：.../aloha-agilex/<task>/<domain>/frames/episode*/<images>
    从 'aloha-agilex' 后一个目录名作为 task。
    """
    parts = os.path.normpath(path).split(os.sep)
    if "aloha-agilex" in parts:
        i = parts.index("aloha-agilex")
        if i + 1 < len(parts):
            return parts[i + 1]
    # 兜底：往上两级
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def _infer_triplet_from_path(path: str, root: Optional[str] = None) -> Tuple[str, str, str]:
    """
    从图片路径推断 (task, domain, episode)。
    目标结构：.../aloha-agilex/<task>/<domain>/frames/<episode>/...
    """
    parts = os.path.normpath(path).split(os.sep)
    # 找到数据集根名（常见为 'aloha-agilex'）
    root_key = "aloha-agilex"
    if root:
        rk = os.path.basename(os.path.normpath(root))
        if rk:
            root_key = rk
    task = ""
    domain = ""
    episode = ""

    # task, domain
    if root_key in parts:
        i = parts.index(root_key)
        if i + 1 < len(parts):
            task = parts[i + 1]
        if i + 2 < len(parts):
            domain = parts[i + 2]

    # episode: 优先从 'frames' 后一段取；否则找以 'episode' 开头的段
    ep_idx = -1
    if "frames" in parts:
        j = parts.index("frames")
        if j + 1 < len(parts):
            episode = parts[j + 1]
            ep_idx = j + 1
    if not episode:
        for seg in parts:
            if seg.lower().startswith("episode"):
                episode = seg
                break

    # 兜底
    if not task:
        task = _infer_task_from_path(path)
    if not domain:
        # 从 task 目录下一个目录兜底
        try:
            i2 = parts.index(task)
            if i2 + 1 < len(parts):
                domain = parts[i2 + 1]
        except Exception:
            pass
    if not episode:
        # 再兜底从文件上级目录取
        episode = os.path.basename(os.path.dirname(path))
    return task, domain, episode


# ========== 主数据集类 ==========
class AlohaAgileXFolderDataset:
    """
    输出结构：
      - observation.image_primary: uint8 [1, H, W, 3]
      - task.language_instruction: bytes（utf-8），供主脚本 texts 使用
      - task.task_name: bytes（utf-8）
      - task.task_instruction: bytes（utf-8），来自 episode JSON 的 'seen' 随机一条
      - 其它键为占位或调试元信息

    为兼容外部使用方式，暴露 .dataset = self
    """

    def __init__(self, cfg_like, train: bool = True):
        # 根目录（图片）
        self.root: str = getattr(cfg_like, "images_root", None)
        if not (self.root and os.path.isdir(self.root)):
            raise FileNotFoundError(f"images_root 不存在：{self.root}")

        # 域白名单
        self.include_domains: List[str] = getattr(
            cfg_like, "include_domains",
            ["Aloha-AgileX", "Aloha-AgileX_domain_randomized"]
        )
        # 任务白/黑名单（可选）
        self.include_tasks: Optional[List[str]] = getattr(cfg_like, "include_tasks", None)
        self.exclude_tasks: Optional[List[str]] = getattr(cfg_like, "exclude_tasks", None)

        # 目录名：frames（可改）
        self.frames_dirname: str = getattr(cfg_like, "frames_dirname", "frames")
        # episode 目录通配
        self.episode_glob: str = getattr(cfg_like, "episode_glob", "episode*")

        # 读取上限（调试用）
        self.max_images: Optional[int] = getattr(cfg_like, "max_images", None)

        # 指令映射（CSV）
        self.instruction_csv: Optional[str] = getattr(cfg_like, "instruction_csv", None)
        self.inst_map: Dict[tuple, str] = _load_instruction_map(self.instruction_csv)
        self.default_instruction_tpl: str = getattr(
            cfg_like, "default_instruction_tpl",
            "Task: {task}. Describe the scene and how to safely complete it."
        )

        # episode 指令 JSON 根目录（你给的 /mnt/data/...）
        self.instructions_root: Optional[str] = getattr(cfg_like, "instructions_root", None)
        # 从 JSON 的哪个字段抽：'seen' / 'unseen' / 'any'
        self.instruction_source: str = getattr(cfg_like, "instruction_source", "seen").lower()
        assert self.instruction_source in ("seen", "unseen", "any")
        # 是否优先使用 JSON 指令来组成 language_instruction
        self.prefer_json_instruction: bool = bool(getattr(cfg_like, "prefer_json_instruction", True))
        # 稳定采样（同一个 episode 每次取到同一条）
        self.stable_sampling: bool = bool(getattr(cfg_like, "stable_sampling", True))
        self.stable_seed: Optional[int] = getattr(cfg_like, "stable_seed", None)

        # resize（None=原图；否则 (W, H)）
        self.resize_to: Optional[Tuple[int, int]] = getattr(cfg_like, "resize_to", (448, 448))

        # 可选：从 filelist 读（每行一个路径）
        self.filelist: Optional[str] = getattr(cfg_like, "filelist", None)

        # 收集文件
        self.files: List[str] = self._collect_files()
        if self.max_images:
            self.files = self.files[: int(self.max_images)]

        self._print_scan_summary()

        if not self.files:
            raise FileNotFoundError(
                f"在 {self.root} 下未找到图片；请检查 include_domains/episode_glob/扩展名/目录拼写（{self.frames_dirname}）"
            )

        # 兼容外部 .dataset 访问
        self.dataset = self

    # ---------- 扫描文件 ----------
    def _collect_files(self) -> List[str]:
        if self.filelist and os.path.isfile(self.filelist):
            files = self._collect_from_filelist(self.filelist)
        else:
            files = self._collect_from_hierarchy()
        files = [os.path.normpath(f) for f in files if self._is_valid_image_path(f)]
        files.sort()
        return files

    def _collect_from_filelist(self, flist: str) -> List[str]:
        out: List[str] = []
        with open(flist, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                if not os.path.isabs(p):
                    p = os.path.join(self.root, p)
                out.append(p)
        return out

    def _collect_from_hierarchy(self) -> List[str]:
        files: List[str] = []
        # 任务：.../aloha-agilex/<task>/
        task_dirs = sorted([d for d in glob.glob(os.path.join(self.root, "*")) if os.path.isdir(d)])

        # 任务白/黑名单过滤
        if self.include_tasks:
            allow: Set[str] = set(self.include_tasks)
            task_dirs = [d for d in task_dirs if os.path.basename(d) in allow]
        if self.exclude_tasks:
            deny: Set[str] = set(self.exclude_tasks)
            task_dirs = [d for d in task_dirs if os.path.basename(d) not in deny]

        for tdir in task_dirs:
            for dom in self.include_domains:
                dom_dir = os.path.join(tdir, dom, self.frames_dirname)
                if not os.path.isdir(dom_dir):
                    continue
                # episodes：frames/<episode_glob>/
                ep_dirs = sorted(glob.glob(os.path.join(dom_dir, self.episode_glob)))
                for ep in ep_dirs:
                    if not os.path.isdir(ep):
                        continue
                    # 递归抓取所有支持的图片扩展
                    for ext in VALID_EXTS:
                        files.extend(glob.glob(os.path.join(ep, f"**/*{ext}"), recursive=True))
        return files

    @staticmethod
    def _is_valid_image_path(p: str) -> bool:
        ext = os.path.splitext(p)[1].lower()
        return ext in VALID_EXTS and os.path.isfile(p)

    def _print_scan_summary(self):
        n = len(self.files)
        try:
            tasks = {_infer_task_from_path(p) for p in self.files}
            nt = len(tasks)
        except Exception:
            nt = 0
        print(f"[DAQI] root={self.root}")
        print(f"[DAQI] include_domains={self.include_domains}, frames_dirname={self.frames_dirname}, episode_glob={self.episode_glob}")
        if self.instructions_root:
            print(f"[DAQI] instructions_root={self.instructions_root}, source={self.instruction_source}, prefer_json_instruction={self.prefer_json_instruction}")
        print(f"[DAQI] tasks matched: {nt}, total images matched: {n}")
        if n > 0:
            print(f"[DAQI] sample path: {self.files[0]}")

    # ---------- episode 指令读取 ----------
    def _episode_json_path(self, task: str, domain: str, episode: str) -> Optional[str]:
        """
        规则：{instructions_root}/{task}/{domain}/instructions/{episode}.json
        """
        if not self.instructions_root:
            return None
        p = os.path.join(self.instructions_root, task, domain, "instructions", f"{episode}.json")
        return p if os.path.isfile(p) else None

    def _choose_from_list(self, items: List[str], key: str) -> Optional[str]:
        if not items:
            return None
        if self.stable_sampling:
            # 用 (stable_seed, key) 做稳定随机
            h = hashlib.md5(key.encode("utf-8")).hexdigest()
            base = int(h[:8], 16)
            seed = base if self.stable_seed is None else (base ^ int(self.stable_seed))
            rng = random.Random(seed)
            return rng.choice(items)
        else:
            return random.choice(items)

    def _get_episode_instruction(self, task: str, domain: str, episode: str) -> Optional[str]:
        """
        读取 episode JSON，从 self.instruction_source 指定的字段中取一条
        """
        jpath = self._episode_json_path(task, domain, episode)
        if not jpath:
            return None
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        src = self.instruction_source
        cand: List[str] = []
        if src == "seen":
            cand = list(data.get("seen", []))
        elif src == "unseen":
            cand = list(data.get("unseen", []))
        else:  # any
            cand = list(data.get("seen", [])) + list(data.get("unseen", []))
        cand = [c for c in cand if isinstance(c, str) and c.strip()]
        if not cand:
            return None
        key = f"{task}|{domain}|{episode}"
        return self._choose_from_list(cand, key)

    # ---------- Dataset 接口 ----------
    def __len__(self) -> int:
        return len(self.files)

    def _pil_to_uint8_tensor(self, img: Image.Image) -> torch.ByteTensor:
        img = img.convert("RGB")
        if self.resize_to:
            img = img.resize(self.resize_to, Image.BILINEAR)  # (W, H)
        w, h = img.size
        arr = torch.frombuffer(img.tobytes(), dtype=torch.uint8).view(h, w, 3).unsqueeze(0).contiguous()
        return arr

    def _resolve_instruction(self, path: str, task: str) -> Tuple[str, Optional[str], str, str]:
        """
        返回：
          - final_instr: 喂给大模型的最终文本（bytes 前身）
          - ep_instr: 从 JSON 选出的原始 episode 指令（可能为 None）
          - domain, episode
        逻辑优先级：
          1) CSV 文件级（覆盖所有）
          2) prefer_json_instruction=True 时，用 JSON seen/unseen 组合成 "Task: {task}. {ep_instr}"
          3) CSV 任务级
          4) 若 2) 没用过，再尝试 JSON 兜底
          5) 默认模板
        """
        task2, domain, episode = _infer_triplet_from_path(path, self.root)

        # 1) CSV: 文件级
        key = ("path", os.path.normpath(path))
        if key in self.inst_map:
            return self.inst_map[key], None, domain, episode

        ep_instr: Optional[str] = None

        # 2) JSON 优先
        if self.prefer_json_instruction:
            ep_instr = self._get_episode_instruction(task2, domain, episode)
            if ep_instr:
                return f"Task: {task2}. {ep_instr}", ep_instr, domain, episode

        # 3) CSV: 任务级
        key = ("task", task2)
        if key in self.inst_map:
            return self.inst_map[key], None, domain, episode

        # 4) JSON 兜底
        if not ep_instr:
            ep_instr = self._get_episode_instruction(task2, domain, episode)
            if ep_instr:
                return f"Task: {task2}. {ep_instr}", ep_instr, domain, episode

        # 5) 默认模板
        return self.default_instruction_tpl.format(task=task2), None, domain, episode

    def __getitem__(self, idx: int):
        path = self.files[idx]
        task = _infer_task_from_path(path)

        with Image.open(path) as im:
            img_tensor = self._pil_to_uint8_tensor(im)
            w, h = (self.resize_to if self.resize_to else im.size)

        final_instr, ep_instr, domain, episode = self._resolve_instruction(path, task)

        return {
            "observation": {
                "image_primary": img_tensor,  # uint8 [1,H,W,3]
            },
            "task": {
                "language_instruction": final_instr.encode("utf-8"),     # 给主脚本/大模型看的文本
                "task_name": task.encode("utf-8"),                       # 任务名
                "task_instruction": (ep_instr or "").encode("utf-8"),    # ★ episode JSON 的原句（seen里选一条）
            },
            # 占位
            "action": torch.zeros(1, 1, 1, 4, dtype=torch.float32),
            "action_pad_mask": torch.ones(1, 1, 1, 4, dtype=torch.float32),
            # 元信息便于调试/追踪
            "meta": {
                "filepath": path,
                "task": task,
                "domain": domain,
                "episode": episode,
                "size": [w, h],
                "instructions_json": self._episode_json_path(task, domain, episode) if self.instructions_root else None,
                "instruction_source": self.instruction_source,
            },
        }
