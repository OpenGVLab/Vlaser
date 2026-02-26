from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
# from .internvl_process_img import load_image

import math
import random
import numpy as np
from PIL import Image

from torchvision import transforms
import itertools

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])






def image_transform(image_size=384):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    return transform

@dataclass
class RLDSBatchTransform:
    # action_tokenizer: Any
    base_tokenizer: PreTrainedTokenizerBase
    act_token_start_idx: int
    # prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    image_size: int = 224

    pixel_values_dtype: torch.dtype = torch.float32
    img_history_size: int = 1
    action_chunk_size: int = 1

    image_transform = image_transform(image_size=image_size)



    def __call__(self, meta, actions, imgs, state) -> Dict[str, Any]:
        # process each instance
        dataset_name = meta["dataset_name"]
        assert len(imgs)==self.img_history_size
        assert len(actions)==self.img_history_size*self.action_chunk_size

        # imgs: size: img_history_size, actions: size: img_history_size*action_chunk_size+1
        pixel_values_steps = [self.image_transform(img) for img in imgs]

        lang = meta["instruction"].lower()
        input_text = lang

        input_text_ids = np.array(self.base_tokenizer(input_text).input_ids)





        # 将actions转化为action tokens
        action_steps = np.array_split(actions, indices_or_sections=self.img_history_size, axis=0)
        action_steps = np.stack(action_steps)  # [img_history_size, action_chunk_size, 14], 是同一个trajectory



        init_state = state

        return dict(pixel_values_steps=pixel_values_steps, input_text_ids=input_text_ids,
                    action_steps_ids=action_steps, action_end_idx=[0],
                    state_ids=init_state, state_end_idx=[0],
                    dataset_name=dataset_name, )