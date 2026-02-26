"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# from prismatic.vla.action_tokenizer import ActionTokenizer
from .data_transform import RLDSBatchTransform
# from .load_data.hdf5_vla_dataset import HDF5VLADataset
from .load_data.hdf5_vla_dataset_read_all_data import HDF5VLADataset

import numpy as np
from transformers import AutoProcessor


from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100



def pad_sequences_to_length(sequences, target_length, batch_first=True, padding_value=0):
    # sequences 是一个 list，包含许多不同长度的1D张量
    # target_length 是你希望填充到的目标长度

    # 注意：确保每个序列长度 <= target_length，否则会截断

    padded_sequences = []
    for seq in sequences:
        # 如果序列长度小于目标长度，则填充序列
        if len(seq) < target_length:
            padded_seq = torch.cat([seq, torch.full((target_length - len(seq),), padding_value)])
        else:
            # 如果序列长度超过目标长度，截断
            padded_seq = seq[:target_length]

        padded_sequences.append(padded_seq)

    # 使用 pad_sequence 来处理最终格式化
    padded_sequences = pad_sequence(padded_sequences, batch_first=batch_first, padding_value=padding_value)

    return padded_sequences


# # 示例用法
# sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
# target_length = 5
# padded_sequences = pad_sequences_to_length(sequences, target_length, batch_first=True, padding_value=0)
#
# print(padded_sequences)










@dataclass
class PaddedCollatorForImageActionPrediction:

    state_max: float = 3.0
    state_min: float = -2.5
    state_vocab_size: int=256
    state_token_start_idx: int=None
    pad_token_id: int=None
    target_length: int=None

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        input_text_ids = [torch.from_numpy(instance["input_text_ids"]).long() for instance in instances]
        pixel_values_steps = torch.stack([torch.stack(instance["pixel_values_steps"]) for instance in instances])

        action_steps = torch.stack([torch.from_numpy(instance["action_steps_ids"]) for instance in instances])


        # 处理states
        states = []
        # np.linspace会生成包含self.state_min, self.state_max在内的self.state_vocab_size个数
        bins = np.linspace(self.state_min, self.state_max, self.state_vocab_size)
        for instance in instances:
            state = np.clip(instance["state_ids"], self.state_min, self.state_max)  # 截断state的取值范围
            discretized_state = np.digitize(state, bins)+self.state_token_start_idx-1  # np.digitize的结果的索引是从1开始
            states.append(torch.from_numpy(discretized_state))
        state_ids = torch.stack(states)



        input_text_ids = pad_sequences_to_length(input_text_ids, self.target_length,
                                                 batch_first=True, padding_value=self.pad_token_id)



        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        output = dict(
            pixel_values_steps=pixel_values_steps,
            input_text_ids=input_text_ids,
            state_ids=state_ids,
            action_steps_ids=action_steps,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output



def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    # image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    # prompt_builder_fn: Type[PromptBuilder],
    # default_image_resolution: Tuple[int, int, int],
    act_token_start_idx: int,
    state_token_start_idx: int,
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    img_history_size: int = 1,
    action_chunk_size: int = 1,
    image_size: int = 224,
    state_dim: int = 14,
    state_max: float=3.0,
    state_min: float=-2.5,
    state_vocab_size: int=256,
    target_length: int=30,
    instruction_path: str = None

) -> Tuple[Dataset, PaddedCollatorForImageActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    # action_tokenizer = AutoProcessor.from_pretrained("./data_utils/dwt_trained_robotwin", trust_remote_code=True)

    batch_transform = RLDSBatchTransform(
        # action_tokenizer,
        tokenizer, image_size=image_size, predict_stop_token=predict_stop_token,
        act_token_start_idx=act_token_start_idx, img_history_size=img_history_size, action_chunk_size=action_chunk_size
    )
    collator = PaddedCollatorForImageActionPrediction(
        state_min=state_min, state_max=state_max, state_vocab_size=state_vocab_size,
        state_token_start_idx=state_token_start_idx, pad_token_id=tokenizer.pad_token_id,
        target_length=target_length
    )


    dataset = HDF5VLADataset(
        data_root_dir,
        action_chunk_size,
        img_history_size,
        state_dim,
        batch_transform,
        instruction_path
        # dataset_num
        # resize_resolution=default_image_resolution[1:],
    )

    return dataset, collator