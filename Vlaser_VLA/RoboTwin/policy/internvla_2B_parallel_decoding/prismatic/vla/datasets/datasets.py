"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from torchvision import transforms
from .augmentation import  random_crop_arr
import itertools

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def image_transform(image_size=224):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform
@dataclass
class RLDSBatchTransform:
    action_tokenizer: Any
    base_tokenizer: PreTrainedTokenizerBase
    act_token_start_idx: int
    # prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    image_size: int = 224

    pixel_values_dtype: torch.dtype = torch.float32
    img_history_size: int = 1
    action_chunk_size: int = 1
    image_size: int = 224

    image_transform = image_transform(image_size)



    def __call__(self, meta, actions, imgs) -> Dict[str, Any]:
        # process each instance
        dataset_name = meta["dataset_name"]
        assert len(imgs)==self.img_history_size+1
        assert len(actions)==self.img_history_size*self.action_chunk_size+1

        # imgs: size: img_history_size, actions: size: img_history_size*action_chunk_size+1
        pixel_values_steps = [self.image_transform(img) for img in imgs]

        lang = meta["instruction"].lower()
        input_text = lang

        input_text_ids = np.array(self.base_tokenizer(input_text).input_ids)





        # 将actions转化为action tokens
        action_steps = np.array_split(actions[1:], indices_or_sections=self.img_history_size, axis=0)
        action_steps = np.stack(action_steps)  # [img_history_size, action_chunk_size, 14], 是同一个trajectory
        action_steps_ids = self.action_tokenizer(action_steps) # list[list]

        action_end_idx = [0]
        action_end_idx.extend([len(action_steps_id) for action_steps_id in action_steps_ids])
        action_end_idx = list(itertools.accumulate(action_end_idx))  # 记录每个action在整个list中的结束index

        # 平铺所有的action_steps_ids
        action_steps_ids = sum(action_steps_ids, [])  # list[list]-> [list]
        action_steps_ids = np.array(action_steps_ids) + self.act_token_start_idx



        init_state = actions[:1]
        # init state也需要过一遍action tokenizer, 因为init state和action的representation相同
        # state_ids = self.action_tokenizer(init_state)
        state_ids = self.action_tokenizer(init_state)[:1]  # 因为输入的init_state的长度为1，action_tokenizer会自动将长度复制为2，最终是2个相同的输出，我们只用1个表示state

        # 使用 itertools.accumulate 进行累加
        state_end_idx = [0]
        state_end_idx.extend([len(state) for state in state_ids])
        state_end_idx = list(itertools.accumulate(state_end_idx))  # 记录每个state在list中的结束index
        state_ids = sum(state_ids, [])  # list[list]-> [list]

        state_ids = np.array(state_ids) + self.act_token_start_idx


        # action_steps_ids = []
        # for action_step in action_steps:
        #     action_steps_id = self.action_tokenizer(np.array(action_step)) + self.act_token_start_idx
        #     action_steps_ids.append(action_steps_id)
        # action_steps_ids = self.base_tokenizer(action_steps).input_ids
        # action_tokens = self.base_tokenizer.batch_decode(action_steps_ids)
        return dict(pixel_values_steps=pixel_values_steps, input_text_ids=input_text_ids,
                    action_steps_ids=action_steps_ids, action_end_idx=action_end_idx,
                    state_ids=state_ids, state_end_idx=state_end_idx,
                    dataset_name=dataset_name, )



class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        step_window_size: int = 1
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=step_window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


# class DummyDataset(Dataset):
#     def __init__(
#         self,
#         action_tokenizer: ActionTokenizer,
#         base_tokenizer: PreTrainedTokenizerBase,
#         image_transform: ImageTransform,
#         prompt_builder_fn: Type[PromptBuilder],
#     ) -> None:
#         self.action_tokenizer = action_tokenizer
#         self.base_tokenizer = base_tokenizer
#         self.image_transform = image_transform
#         self.prompt_builder_fn = prompt_builder_fn
#
#         # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
#         # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
#         self.dataset_statistics = {
#             "dummy_dataset": {
#                 "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
#             }
#         }
#
#     def __len__(self):
#         # TODO =>> Replace with number of elements in your dataset!
#         return 10000
#
#     def __getitem__(self, idx):
#         # TODO =>> Load image, action and instruction from disk -- we use dummy values
#         image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
#         action = np.asarray(np.random.rand(7), dtype=np.float32)
#         instruction = "do something spectacular"
#
#         # Add instruction to VLA prompt
#         prompt_builder = self.prompt_builder_fn("openvla")
#         conversation = [
#             {"from": "human", "value": f"What action should the robot take to {instruction}?"},
#             {"from": "gpt", "value": self.action_tokenizer(action)},
#         ]
#         for turn in conversation:
#             prompt_builder.add_turn(turn["from"], turn["value"])
#
#         # Tokenize (w/ `base_tokenizer`)
#         input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
#         labels = list(input_ids)
#
#         # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
#         #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
#         input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
#         pixel_values = self.image_transform(image)
#
#         # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
#         labels[: -(len(action) + 1)] = IGNORE_INDEX
#
#         return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
