"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from internvl.util.data_utils import PaddedCollatorForActionPrediction
from internvl.vla.action_tokenizer import ActionTokenizer
from internvl.vla.datasets import RLDSBatchTransform, RLDSDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    default_image_resolution: Tuple[int, int],
    # padding_side: str = "right",
    predict_stop_token: bool = True,
    # shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    use_wrist_image: bool = True,
    use_proprio: bool = True,
    num_image_token: int = 64,
    shuffle_buffer_size: int = 50000,
    # num_new_tokens: int = 0,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        predict_stop_token=predict_stop_token,
        use_wrist_image=use_wrist_image,
        use_proprio=use_proprio,
        is_train=train,
        image_size=default_image_resolution,
        num_image_token=num_image_token,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, 
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset
    dataset = cls(
        data_root_dir=data_root_dir,
        data_mix=dataset_name,
        batch_transform=batch_transform,
        resize_resolution=default_image_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    return dataset, action_tokenizer, collator
