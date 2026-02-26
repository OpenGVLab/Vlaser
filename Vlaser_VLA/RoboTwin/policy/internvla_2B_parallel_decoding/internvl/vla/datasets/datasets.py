"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, Iterator, Tuple, Type

import numpy as np
from copy import deepcopy
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms as T
from transformers import PreTrainedTokenizerBase
from torchvision.transforms.functional import InterpolationMode
import transformers
import tensorflow as tf

from internvl.conversation import get_conv_template
from internvl.util.data_utils import tree_map
from internvl.vla.action_tokenizer import ActionTokenizer
from internvl.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD, SIGLIP_MEAN, SIGLIP_STD, IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN, QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN, REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN
from internvl.vla.constants import PROPRIO_START_TOKEN, PROPRIO_END_TOKEN, PROPRIO_CONTEXT_TOKEN
from internvl.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from internvl.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_robotwin_dataset_kwargs, get_oxe_dataset_kwargs_and_weights
import io

def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    is_train: bool = True
    image_size: Tuple[int, int] = (224, 224)
    num_image_token: int = 64

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by InternVL models."""
        dataset_name = rlds_batch["dataset_name"]
        actions = rlds_batch["action"]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        instr = rlds_batch["task"]["language_instruction"]

        if isinstance(instr, (np.ndarray, list, tuple)):
            instr = random.choice(instr)

        if isinstance(instr, bytes):
            instr = instr.decode("utf-8")

        assert isinstance(instr, str), f"Unexpected type for language_instruction: {type(instr)}"
        lang = instr.lower()
        transform = self.get_transform()

        imgs = []
        imgs.append(img)
        num_tiles = [1]
        if self.use_wrist_image:
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    imgs.append(img_wrist)
                    num_tiles.append(1)

        num_images = len(imgs)
        pixel_values = [transform(img) for img in imgs]
        # data augmentation together
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        action_chunk_string = ''.join(self.action_tokenizer(actions))
        action_chunk_token = np.array(self.action_tokenizer.encode_actions_to_token_ids(actions))
        
        human_prompt_value = "<image>\n" * num_images

        human_prompt_value += "What action should the robot take to " + lang + "?"
        
        proprio_available = self.use_proprio and "proprio" in rlds_batch["observation"]
        if proprio_available:
            human_prompt_value += "<proprio>\n"

        conversations = [
            {"from": "human", "value": human_prompt_value},
            {"from": "gpt", "value": action_chunk_string}
        ]

        # Construct InternVL prompt using the Conversation class
        # conv = get_conv_template("internvl_zh") # maybe something else??
        # need to add the image to the prompt??
        # human_prompt = f"<image>\nWhat action should the robot take to {lang}?"
        # human_prompt = f"What action should the robot take to {lang}?"
        # conv.append_message(conv.roles[0], human_prompt)
        # conv.append_message(conv.roles[1], action_chunk_string)
        # prompt = conv.get_prompt()

        preprocess_function = preprocess_internvl2_5

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function("internvl2_5", [deepcopy(conversations)],
                                  self.base_tokenizer, num_image_tokens, num_image=num_images, action_chunk_token=deepcopy(action_chunk_token))

        #input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
        # labels = list(input_ids)
        # labels = ret['labels']

        # labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        # use another conv template to get the prompt up to the bot's answer
        # conv_for_masking = get_conv_template("internvl_zh")
        # conv_for_masking.append_message(conv_for_masking.roles[0], human_prompt)
        # conv_for_masking.append_message(conv_for_masking.roles[1], None) # empty message
        # prompt_until_bot_answer = conv_for_masking.get_prompt() # end with '<bot>:'
        
        labels = ret['labels'][0]

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        # if not self.predict_stop_token:
        #     labels[-1] = IGNORE_INDEX

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        # input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # pixel_values = self.image_transform(img)

        # return_dict = dict(
        #     pixel_values=pixel_values,
        #     input_ids=input_ids,
        #     labels=labels,
        #     dataset_name=dataset_name,
        #     actions=actions
        # )
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=labels,
            # attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            dataset_name=dataset_name,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            actions=actions
        )

        if proprio_available:
            proprio = rlds_batch["observation"]["proprio"]
            ret["proprio"] = torch.from_numpy(proprio) if isinstance(proprio, np.ndarray) else proprio

        return ret

        # if self.use_wrist_image:
        #     all_wrist_pixels = []
        #     for k in rlds_batch["observation"].keys():
        #         if "wrist" in k:
        #             img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
        #             pixel_values_wrist = self.image_transform(img_wrist)
        #             all_wrist_pixels.append(pixel_values_wrist)
        #     return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)

        # if self.use_proprio and "proprio" in rlds_batch["observation"]:
        #     proprio = rlds_batch["observation"]["proprio"]
        #     return_dict["proprio"] = proprio

        # return return_dict
    
    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size)
        return transform
    
def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform

def preprocess_internvl2_5(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        action_chunk_token,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    # if conversations[0]['from'] == 'system':
    #     system_prompt = conversations[0]['value']
    #     conversations = conversations[1:]  # remove system prompt
    # else:
    #     conv = get_conv_template(template_name)
    #     system_prompt = conv.system_message
        # system_prompt = None

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        current_proprio_idx = 0 

        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for _ in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    num_tokens = num_image_token_list[current_image_idx]
                    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_tokens}{IMG_END_TOKEN}'
                    # image_tokens = f'{IMG_CONTEXT_TOKEN * num_tokens}'
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1

                proprio_cnt = conversation['value'].count('<proprio>')
                for _ in range(proprio_cnt):
                    proprio_tokens = f'{PROPRIO_START_TOKEN}{PROPRIO_CONTEXT_TOKEN}{PROPRIO_END_TOKEN}'
                    # proprio_tokens = f'{PROPRIO_CONTEXT_TOKEN}'
                    # Replace the placeholder (e.g., '<proprio>') with the full token sequence
                    conversation['value'] = conversation['value'].replace('<proprio>', proprio_tokens, 1)
                    current_proprio_idx += 1

            new_conversations.append(conversation)
        
        conversations = new_conversations
        
        # --- Final Assertions (Updated) ---
        assert current_image_idx == num_image, f'Image count mismatch: used {current_image_idx}, expected {num_image}'
        assert current_proprio_idx == 1, f'Proprio count mismatch: used {current_proprio_idx}, expected 1'

    batches, roles = [], []
    # if system_prompt is not None:
    #     batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
    #     roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('gpt')
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids
    
    

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id in zip(roles, input_ids):
        if role == 'system' or role == 'human':
            final_targets.append(np.full(input_id.shape, IGNORE_INDEX))  # ignore
            final_input_ids.append(input_id)
        elif role == 'gpt':
            ids = np.concatenate([tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0],
                                  action_chunk_token.flatten(), 
                                  tokenizer('<|im_end|>\n', return_tensors='np').input_ids[0]])
            target = ids.copy()
            target[:ignore_len] = IGNORE_INDEX  # ignore loss for `<|im_start|>assistant\n`
            # target[-1:] = IGNORE_INDEX  # ignore loss for `\n`b[]
            target[-2:] = IGNORE_INDEX  # ignore both `<|im_end|>` and `\n`
            final_targets.append(target)
            final_input_ids.append(ids)
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]

    # padding = False if group_by_length or use_packed_ds else True
    # if padding:
    #     current_length = input_ids.size(0)
    #     padding_length = tokenizer.model_max_length - current_length
    #     input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
    #     targets = F.pad(targets, (0, padding_length), value=IGNORE_INDEX)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )



class RLDS_single_Dataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        dataset_name: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000, 
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        # self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.batch_transform = batch_transform

        # Configure RLDS Dataset(s)
        # if self.data_mix in OXE_NAMED_MIXTURES:
        #     mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        # else:
        #     # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
        #     mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in dataset_name:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")
        
        # dataset_kwargs = dict(
        #     name=dataset_name,
        #     data_dir=str(data_root_dir),
        #     image_obs_keys={view: view for view in load_camera_views},
        #     state_obs_keys=["state"],
        #     action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        #     num_parallel_calls=tf.data.AUTOTUNE,
        #     num_parallel_reads=tf.data.AUTOTUNE,
        # )
        
        dataset_kwargs = get_robotwin_dataset_kwargs(
            data_root_dir,
            dataset_name=dataset_name,
            load_camera_views=load_camera_views,
            load_depth=False,  
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        dataset_kwargs["num_parallel_calls"] = tf.data.AUTOTUNE
        dataset_kwargs["num_parallel_reads"] = tf.data.AUTOTUNE
        
        
            # num_parallel_calls=tf.data.AUTOTUNE,
            # num_parallel_reads=tf.data.AUTOTUNE,

        traj_transform_kwargs = dict(
            window_size=1,
            future_action_window_size=NUM_ACTIONS_CHUNK - 1,
            skip_unlabeled=True,
            goal_relabeling_strategy="uniform",
        )
        
        frame_transform_kwargs = dict(
            resize_size=resize_resolution,
            num_parallel_calls=16,
        )
        # per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
        #     self.data_root_dir,
        #     mixture_spec,
        #     load_camera_views=load_camera_views,
        #     load_depth=False,
        #     load_proprio=True,
        #     load_language=True,
        #     action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        # )
        # rlds_config = dict(
        #     traj_transform_kwargs=dict(
        #         window_size=1,                                      # If we wanted to feed / predict more than one step
        #         future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
        #         skip_unlabeled=True,                                # Skip trajectories without language labels
        #         goal_relabeling_strategy="uniform",                 # Goals are currently unused
        #     ),
        #     frame_transform_kwargs=dict(
        #         resize_size=resize_resolution,
        #         num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
        #     ),
        #     dataset_kwargs_list=per_dataset_kwargs,
        #     shuffle_buffer_size=shuffle_buffer_size,
        #     sample_weights=weights,
        #     balance_weights=True,
        #     traj_transform_threads=len(mixture_spec),
        #     traj_read_threads=len(mixture_spec),
        #     train=train,
        # )

        # If applicable, enable image augmentations
        if image_aug:
            # Define common augmentation parameters (without random_resized_crop)
            base_augment_kwargs = dict(
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_brightness", "random_contrast",
                    "random_saturation", "random_hue",
                ],
            )
            
            # Define primary camera augmentation (with random_resized_crop)
            primary_augment_kwargs = dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop", "random_brightness", "random_contrast",
                    "random_saturation", "random_hue",
                ],
            )
            
            # Create per-camera augmentation configuration
            camera_specific_augment = {}
            for view in load_camera_views:
                if view == "primary":
                    camera_specific_augment[view] = primary_augment_kwargs
                else:
                    # For wrist cameras, use base augmentation without random_resized_crop
                    camera_specific_augment[view] = base_augment_kwargs
            
            frame_transform_kwargs["image_augment_kwargs"] = camera_specific_augment
        # fmt: on

        # Initialize RLDS Dataset
        dataset, num_trajectories, statistics = make_single_dataset(
            dataset_kwargs=dataset_kwargs,
            train=train,
            traj_transform_kwargs=traj_transform_kwargs,
            frame_transform_kwargs=frame_transform_kwargs,
            shuffle_buffer_size=shuffle_buffer_size, 
        )

        self.dataset = dataset
        self.dataset_length = num_trajectories
        self.dataset_statistics = statistics

    # def make_dataset(
    #     self,
    #     dataset_kwargs: dict,
    #     train: bool,
    #     traj_transform_kwargs: dict,
    #     frame_transform_kwargs: dict
    # ) -> None:
    #     dataset, statistics = make_single_dataset(
    #         dataset_kwargs=dataset_kwargs,
    #         train=train,
    #         traj_transform_kwargs=traj_transform_kwargs,
    #         frame_transform_kwargs=frame_transform_kwargs,
    #     )
    #     return dataset, statistics.get("num_trajectories", 0), statistics

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


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
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
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
            # Define common augmentation parameters (without random_resized_crop)
            base_augment_kwargs = dict(
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_brightness", "random_contrast",
                    "random_saturation", "random_hue",
                ],
            )
            
            # Define primary camera augmentation (with random_resized_crop)
            primary_augment_kwargs = dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop", "random_brightness", "random_contrast",
                    "random_saturation", "random_hue",
                ],
            )
            
            # Create per-camera augmentation configuration
            camera_specific_augment = {}
            for view in load_camera_views:
                if view == "primary":
                    camera_specific_augment[view] = primary_augment_kwargs
                else:
                    # For wrist cameras, use base augmentation without random_resized_crop
                    camera_specific_augment[view] = base_augment_kwargs
            
            rlds_config["frame_transform_kwargs"]["image_augment_kwargs"] = camera_specific_augment

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

        # Shard dataset across distributed ranks to avoid identical batches on each process
        try:
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                # tf.data.Dataset supports sharding natively
                self.dataset = self.dataset.shard(num_shards=world_size, index=rank)
        except Exception:
            # If torch.distributed is unavailable or sharding not supported, proceed without sharding
            pass

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


# class EpisodicRLDSDataset(RLDSDataset):
#     """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

#     def make_dataset(self, rlds_config):
#         per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
#         assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

#         return make_single_dataset(
#             per_dataset_kwargs[0],
#             train=rlds_config["train"],
#             traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
#             frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
#         )

#     def __iter__(self) -> Dict[str, Any]:
#         for rlds_batch in self.dataset.as_numpy_iterator():
#             out = [
#                 self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
#                 for i in range(rlds_batch["action"].shape[0])
#             ]
#             yield out


