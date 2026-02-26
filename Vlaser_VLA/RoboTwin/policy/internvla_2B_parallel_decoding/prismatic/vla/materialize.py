"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForImageActionPrediction
# from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset
from prismatic.load_data.hdf5_vla_dataset import HDF5VLADataset

import numpy as np
from transformers import AutoProcessor




# def get_vla_dataset_and_collator(
#     data_root_dir: Path,
#     data_mix: str,
#     # image_transform: ImageTransform,
#     tokenizer: PreTrainedTokenizerBase,
#     # prompt_builder_fn: Type[PromptBuilder],
#     default_image_resolution: Tuple[int, int, int],
#     act_token_start_idx: int,
#     padding_side: str = "right",
#     predict_stop_token: bool = True,
#     shuffle_buffer_size: int = 100_000,
#     train: bool = True,
#     episodic: bool = False,
#     image_aug: bool = False,
#     step_window_size: int = 1,
#     action_chunk_size: int = 1,
#     image_size: int = 224
# ) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
#     """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
#     action_tokenizer = ActionTokenizer(tokenizer, act_token_start_idx)
#     batch_transform = RLDSBatchTransform(
#         action_tokenizer, tokenizer, image_size= image_size, predict_stop_token=predict_stop_token
#     )
#     collator = PaddedCollatorForImageActionPrediction(
#         tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side, step_window_size = step_window_size, action_chunk_size=action_chunk_size
#     )
#
#     # Build RLDS Iterable Dataset
#     cls = RLDSDataset if not episodic else EpisodicRLDSDataset
#     dataset = cls(
#         data_root_dir,
#         data_mix,
#         batch_transform,
#         resize_resolution=default_image_resolution[1:],
#         shuffle_buffer_size=shuffle_buffer_size,
#         train=train,
#         image_aug=image_aug,
#         step_window_size = step_window_size,
#     )
#
#     return dataset, action_tokenizer, collator



def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    # image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    # prompt_builder_fn: Type[PromptBuilder],
    # default_image_resolution: Tuple[int, int, int],
    act_token_start_idx: int,
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
) -> Tuple[Dataset, AutoProcessor, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    action_tokenizer = AutoProcessor.from_pretrained("./prismatic/vla/dwt_trained_robotwin", trust_remote_code=True)

    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_size=image_size, predict_stop_token=predict_stop_token,
        act_token_start_idx=act_token_start_idx, img_history_size=img_history_size, action_chunk_size=action_chunk_size
    )
    collator = PaddedCollatorForImageActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side, img_history_size=img_history_size, action_chunk_size=action_chunk_size
    )


    dataset = HDF5VLADataset(
        data_root_dir,
        action_chunk_size,
        img_history_size,
        state_dim,
        batch_transform,
        # resize_resolution=default_image_resolution[1:],
    )

    return dataset, action_tokenizer, collator