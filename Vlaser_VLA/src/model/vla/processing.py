from typing import List
from typing import List, Dict, Any, Callable, Optional

import torch
from transformers import AutoProcessor

IMAGENET_STANDARD_MEAN = torch.tensor([0.5, 0.5, 0.5])
IMAGENET_STANDARD_STD = torch.tensor([0.5, 0.5, 0.5])


def add_image_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    image_seq_len,
    image_token,
):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: torch.LongTensor,
    scale: float,
) -> torch.FloatTensor:
    rescaled_image = image * scale
    return rescaled_image


def normalize(
    image: torch.LongTensor,
    mean: torch.FloatTensor,
    std: torch.FloatTensor,
) -> torch.FloatTensor:
    assert image.ndim == 5, f"Expected 5D tensor, got {image.ndim}D tensor."
    assert (
        image.shape[2] == 3
    ), f"Expected 3 channels at axis 2, got {image.shape[2]} channels."
    mean = mean[None, None, :, None, None]  # add batch and spatial dimensions
    std = std[None, None, :, None, None]
    image = (image - mean) / std
    image = image.flatten(0,1)
    return image


def process_images(
    images: torch.LongTensor,
    rescale_factor: float,
    image_mean: torch.FloatTensor,
    image_std: torch.FloatTensor,
) -> torch.FloatTensor:
    # Rescale the pixel values to be in the range [0, 1]
    images = rescale(images, scale=rescale_factor)

    # Normalize the images to have mean 0 and standard deviation 1
    images = normalize(images, mean=image_mean, std=image_std)

    return images


class VLAProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        tokenizer_padding: str = "max_length",  #  # instead of truncating to longest
    ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.max_seq_len = max_seq_len
        self.tokenizer_padding = tokenizer_padding

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: torch.LongTensor,
        actions: torch.Tensor = None, 
        truncation: bool = True,
    ) -> dict:
        assert len(images) == len(
            text
        ), f"Received {len(images)} images for {len(text)} prompts."
        assert (
            images.dtype == torch.uint8
        ), f"Expected uint8 tensor for images, got {images.dtype}."

        pixel_values = process_images(
            images,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,
        )
        output = {"pixel_values": pixel_values, **inputs}
        return output



class InternVLAProcessor_old:
    IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        tokenizer_padding: str = "max_length",  #  # instead of truncating to longest
    ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.max_seq_len = max_seq_len
        self.tokenizer_padding = tokenizer_padding

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        # tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        # tokenizer.add_special_tokens(tokens_to_add)
        # EXTRA_TOKENS = [
        #     f"<loc{i:04d}>" for i in range(1024)
        # ]  # These tokens are used for object detection (bounding boxes)
        # EXTRA_TOKENS += [
        #     f"<seg{i:03d}>" for i in range(128)
        # ]  # These tokens are used for object segmentation
        # tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        # tokenizer.add_bos_token = False
        # tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: torch.LongTensor,
        truncation: bool = True,
    ) -> dict:
        assert len(images) == len(
            text
        ), f"Received {len(images)} images for {len(text)} prompts."
        assert (
            images.dtype == torch.uint8
        ), f"Expected uint8 tensor for images, got {images.dtype}."



        MEAN = torch.tensor([0.4850, 0.4560, 0.4060])
        STD = torch.tensor([0.2290, 0.2240, 0.2250])

        pixel_values = process_images(
            images,
            rescale_factor=1 / 255.0,
            image_mean=MEAN,
            image_std=STD,
        )

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        
        
        import sys
        import os
        current_dir = os.getcwd()
        sys.path.append(os.path.join(current_dir, "src/model/internvl_chat"))

        
        from internvl.train.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN
        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.image_seq_length}{IMG_END_TOKEN}'
        input_strings = [image_tokens + prompt for prompt in text]

       
        
        from internvl.train.dataset import preprocess_internvl2_5

        conversation = [[
                        # {"from": "system", "value": "You are required to control the robot by discreted actions. Specifically, given the current image observation, you should output the next action (x, y, z, roll, pitch, yaw, gripper)"},
                        {"from": "human", "value": '<image>\n' + prompt},
                        {"from": "gpt", "value": ''},] for prompt in text]
        conversation = [[{"from": "human", "value": '<image>\n' + prompt},{"from": "gpt", "value": ''},] for prompt in text]
        # import ipdb;ipdb.set_trace()
        # ret = preprocess_internvl2_5(template_name = 'internvl2_5',sources=conversation, tokenizer = self.tokenizer, num_image_token_list = [64 for item in text], ds_name='', num_image = 1)
        ret_list = []
        self.tokenizer.model_max_length = self.max_seq_len
        num_image_token = 64
        import os
        if 'IMAGE_448' in os.environ:
            num_image_token = 256
        for i in range(len(conversation)):
            ret = preprocess_internvl2_5(template_name = 'internvl2_5',sources=[conversation[i]], tokenizer = self.tokenizer, num_image_token_list = [num_image_token], ds_name='', num_image = 1)
            ret_list.append(ret)
        inputs = {'input_ids': torch.cat([item['input_ids'] for item in ret_list]),
                        'labels': torch.cat([item['labels'] for item in ret_list]),
                        'attention_mask': torch.cat([item['attention_mask'] for item in ret_list]),
                }        
        
        
        
        # hard code
        img_content = ''.join(['<IMG_CONTEXT>' for iii in range(num_image_token)])
        query = ["<|im_start|>system\nNone<|im_end|>\n<|im_start|>user\n<img>{}</img>\n{}<|im_end|>\n<|im_start|>assistant\n".format(img_content, prompt) for prompt in text]
        
        output = {"pixel_values": pixel_values, **inputs}
        return output


class InternVLAProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        actions: torch.Tensor = None,
        tokenizer_padding: str = "max_length",  #  # instead of truncating to longest
        num_images = 1,
    ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.max_seq_len = max_seq_len
        self.tokenizer_padding = tokenizer_padding

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        # tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        # tokenizer.add_special_tokens(tokens_to_add)
        # EXTRA_TOKENS = [
        #     f"<loc{i:04d}>" for i in range(1024)
        # ]  # These tokens are used for object detection (bounding boxes)
        # EXTRA_TOKENS += [
        #     f"<seg{i:03d}>" for i in range(128)
        # ]  # These tokens are used for object segmentation
        # tokenizer.add_tokens(EXTRA_TOKENS)
        # import ipdb;ipdb.set_trace()
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        # tokenizer.add_bos_token = False
        # tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
        self.num_images = num_images

    def __call__(
        self,
        text: List[str],
        images: torch.LongTensor,
        truncation: bool = True,
        actions: torch.Tensor = None, 
    ) -> dict:
        assert len(images) == len(
            text
        ), f"Received {len(images)} images for {len(text)} prompts."
        assert (
            images.dtype == torch.uint8
        ), f"Expected uint8 tensor for images, got {images.dtype}."

        # print(images.shape)

        MEAN = torch.tensor([0.4850, 0.4560, 0.4060])
        STD = torch.tensor([0.2290, 0.2240, 0.2250])

        pixel_values = process_images(
            images,
            rescale_factor=1 / 255.0,
            image_mean=MEAN,
            image_std=STD,
        )

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        
        # input_strings = [
        #     add_image_tokens_to_prompt(
        #         prefix_prompt=prompt,
        #         bos_token='\n',
        #         image_seq_len=self.image_seq_length,
        #         image_token=self.IMAGE_TOKEN,
        #     )
        #     for prompt in text
        # ]
        import sys
        import os
        current_dir = os.getcwd()
        sys.path.append(os.path.join(current_dir, "src/model/internvl_chat"))

        
        from internvl.train.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN
        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.image_seq_length}{IMG_END_TOKEN}'
        input_strings = [image_tokens + prompt for prompt in text]

        

        conversation = [[
                        # {"from": "system", "value": "You are required to control the robot by discreted actions. Specifically, given the current image observation, you should output the next action (x, y, z, roll, pitch, yaw, gripper)"},
                        {"from": "human", "value": '<image>\n' + prompt},
                        {"from": "gpt", "value": ''},] for prompt in text]
        conversation = [[{"from": "human", "value": '<image>\n' + prompt},{"from": "gpt", "value": ''},] for prompt in text]
        # import ipdb;ipdb.set_trace()
        # ret = preprocess_internvl2_5(template_name = 'internvl2_5',sources=conversation, tokenizer = self.tokenizer, num_image_token_list = [64 for item in text], ds_name='', num_image = 1)
        ret_list = []
        self.tokenizer.model_max_length = self.max_seq_len
        num_image_token = 64 * self.num_images
        import os
        if 'IMAGE_448' in os.environ:
            num_image_token = 256 * self.num_images
        
       
        
        
        # hard code
        img_content = ''.join(['<IMG_CONTEXT>' for iii in range(num_image_token)])
        if 'DEBUG_prompt' in os.environ:
            query = ["<|im_start|>system\nNone<|im_end|>\n<|im_start|>user\n<img>{}</img>\nThe task is to {}, show me the position of the target object, and how to achieve the task.<|im_end|>\n<|im_start|>assistant\n".format(img_content, prompt) for prompt in text]
        else:
            query = ["<|im_start|>system\nNone<|im_end|>\n<|im_start|>user\n<img>{}</img>\n{}<|im_end|>\n<|im_start|>assistant\n".format(img_content, prompt) for prompt in text]
        
        inputs = self.tokenizer(query, return_tensors='pt',
        max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,)
       
        output = {"pixel_values": pixel_values, **inputs}
        return output



def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format.
    Action token 0 is mapped to the string <robot_action_0>  ... and so on 
    """
    return ''.join([f"<robot_action_{token}>" for token in tokens])

def process_example(example: Dict[str, Any], fast_tokenizer: AutoProcessor) -> Dict[str, Any]:
    """Processes a single example from the dataset."""
    pixel_values = example['image']
    action = example['action']
    lang = example['lang']
    if action != None:
        fast_tokens = fast_tokenizer(action)
        vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])
    else:
        vlm_action = ''

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pixel_values},
                {"type": "text", "text": lang},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": vlm_action},
            ],
        },
    ]
    return messages

def collate_fn(examples, processor, fast_tokenizer):
        messages = [process_example(example,fast_tokenizer) for example in examples]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = processor(text=text, images=image_inputs, videos=video_inputs, padding='max_length', return_tensors="pt", max_length=384,)
        # import ipdb;ipdb.set_trace()
        processor.tokenizer.padding_side = 'right'
        action_token_min = 151665
        action_token_max = 153712
        labels = batch_input['input_ids'].clone()
        # For each sequence in the batch, find the first occurrence of an action token.
        
        for i in range(labels.size(0)):
            seq = labels[i]
            # Create a mask for tokens within the action token range.
            mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
            nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
            if nonzero_indices.numel() > 0:
                first_action_index = nonzero_indices[0].item()
                # Mask out all tokens before the first action token.
                seq[:first_action_index] = -100

            else:
                # If no action token is found, mask the entire sequence.
                seq[:] = -100
        
        labels[labels == processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
        batch_input['labels'] = labels
        return batch_input

