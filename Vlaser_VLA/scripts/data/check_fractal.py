import time
import os
'''
if you use a s3 bucket, please modify the following variables

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["AWS_ACCESS_KEY_ID"] = YOUR_AWS_ACCESS_KEY_ID
# os.environ["AWS_SECRET_ACCESS_KEY"] = YOUR_AWS_SECRET_ACCESS_KEY
# os.environ["S3_ENDPOINT"] = YOUR_S3_ENDPOINT
# os.environ["S3_USE_HTTPS"] = "0"
# os.environ["S3_VERIFY_SSL"] = "0"
'''


from transformers import AutoTokenizer
import torch.distributed as dist

import tensorflow_io as tfio
import tensorflow as tf
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import make_interleaved_dataset
from src.data.dataset_torch import TorchRLDSDataset
from src.data.oxe import make_oxe_dataset_kwargs_and_weights
from src.utils.geometry import quat2euler
from copy import deepcopy
import numpy as np

tf.config.set_visible_devices([], "GPU")
import sys
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, "../../src/model/internvl_chat"))
from internvl.train.dataset import build_transform,preprocess_internvl2_5

def bin_actions(action_list, num_bins=256):
    """
    将动作列表进行等宽分箱
    
    Args:
        action_list: 动作列表，每个元素是一个动作向量
        num_bins: 分箱数量
    
    Returns:
        binned_actions: 分箱后的动作列表
    """
    action_array = np.array(action_list)
    
    # 创建分箱边界，从-1到1等分
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    
    # 对每个动作向量进行分箱
    binned_actions = []
    for action in action_array:
        # 使用digitize进行分箱，返回每个值对应的bin索引
        binned_action = np.digitize(action, bin_edges) - 1  # 减1使索引从0开始
        # 确保索引在有效范围内
        binned_action = np.clip(binned_action, 0, num_bins - 1)
        binned_actions.append(binned_action.tolist())
    
    return binned_actions

# 使用示例
# action_list = [[0.19681906700134277, 0.14072728157043457, 0.028178930282592773, -0.1145966649055481, 0.02161252498626709, 0.07884299755096436, 0.0], 
#                [0.16158580780029297, 0.10756051540374756, 0.046381354331970215, -0.11462771892547607, 0.09642493724822998, 0.0919419527053833, 0.0], 
#                [0.0771629810333252, 0.08209776878356934, 0.07732462882995605, -0.11249476671218872, 0.1623162031173706, 0.169647216796875, 0.0], 
#                [-0.1634252667427063, 0.2141416072845459, 0.15296542644500732, -0.07508552074432373, 0.19027602672576904, 0.47648441791534424, 0.0]]

# binned_result = bin_actions(action_list, num_bins=256)

def process_batch(batch, tokenizer, template_name, transform, 
                dynamic_image_size=False, min_dynamic_patch=1, max_dynamic_patch=12,
                image_size=448, use_thumbnail=False, num_image_token=256,
                group_by_length=False, use_packed_ds=False, ds_name=None):

    # 从batch中提取数据
    images = batch["observation"]["image_primary"]
    proprios = batch["observation"]["proprio"]
    actions = batch["action"].squeeze(1)
    texts = [text.decode("utf-8") for text in batch["task"]["language_instruction"]]
    
    batch_size = len(texts)
    ret_list = []
    
    for i in range(batch_size):
        # 构建conversations格式
        conversation_text = texts[i]
        
        # 确保第一个对话包含图像占位符
        if '<image>' not in conversation_text:
            conversation_text = '<image>\n' + conversation_text
        
        # 构建data_item格式
        data_item = {
            'conversations': [
                {
                    'from': 'human',
                    'value': conversation_text
                },
                {
                    'from': 'gpt', 
                    'value': f"Action: {bin_actions(actions[i].tolist())}"  # 将动作转换为字符串格式
                }
            ]
        }

        # import ipdb;ipdb.set_trace()
        # 获取对应的图像
        image = images[i]  # 假设images[i]已经是PIL Image或numpy array格式
        
        # 动态图像预处理
        processed_images = [image]
        
        # 应用变换并堆叠成张量
        pixel_values = [transform(Image.fromarray(img[0].cpu().numpy())) for img in processed_images]
        
        pixel_values = torch.stack(pixel_values)
        
        # 确保补丁数量正确
        num_patches = pixel_values.size(0)
        if not dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        

        # 选择适当的预处理函数
        preprocess_function = preprocess_internvl2_5 
        # 预处理对话并生成返回字典
        ret = preprocess_function(template_name, [deepcopy(data_item['conversations'])],
                                tokenizer, [num_image_token * num_patches],
                                group_by_length=group_by_length,
                                use_packed_ds=use_packed_ds, ds_name=ds_name)
        
        # 添加图像数据到ret中 
        ret['pixel_values'] = pixel_values
        # ret['num_patches'] = num_patches
        ret['image_flags'] =torch.tensor([1] * num_patches)
        
        
        ret_list.append(ret)
    ret_new = {}
    
    for k in ret_list[0].keys():
        ret_new[k] = torch.stack([item[k] for item in ret_list])
    return ret_new


if __name__ == "__main__":
    import argparse
    import os

    import einops

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default=f"your data path"
    )
    parser.add_argument("--mix", type=str, default="fractal")
    parser.add_argument("--camera_views", nargs="*", default=("primary",))
    parser.add_argument(
        "--skip_norm", action="store_true", help="Use raw actions and proprio"
    )
    args = parser.parse_args()

    # config
    start_time = time.time()
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        args.mix,
        args.data_path,
        load_depth=False,
        load_language=True,
        load_proprio=True,
        load_camera_views=args.camera_views,
        skip_norm=args.skip_norm,
    )
    transform = build_transform(is_train=True, input_size=448,
                                pad2square=False, normalize_type='imagenet')    

    # dataset - fractal has 82851 trajectories and 3786400 transitions
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=True,
        split="train[:95%]",  # fractal does not have validation split
        shuffle_buffer_size=10000,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
        batch_size=None,  # batching will be handles in PyTorch Dataloader object
        balance_weights=True,
        traj_transform_kwargs=dict(  # no neeed for goal relabeling
            window_size=1,
            action_horizon=4,
            subsample_length=100,
            skip_unlabeled=False,  # skip ones without language annotation
            # max_action_from_stats=True,
            # max_proprio_from_stats=True,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(
                        scale=[0.8, 1.0],
                        ratio=[0.9, 1.1],
                    ),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
                "wrist": dict(
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            },
            resize_size=dict(
                primary=(224, 224),
                wrist=(224, 224),
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=48,
        traj_read_threads=48,
    )

    # convert for torch
    pytorch_dataset = TorchRLDSDataset(dataset)
    print("Dataset length (traj):", len(pytorch_dataset))
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=16,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )
    prep_time = time.time()
    print(f"Preparation time: {prep_time - start_time:.2f}s")

    print("Starting dataloader")
    cnt_batch = 0
    for _, _sample in tqdm.tqdm(enumerate(dataloader)):
        # _sample: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
        # observation: 'image_primary' (torch.Size([16, 2, 256, 256, 3]), 'image_wrist', 'timestep' (torch.Size([16, 2])), 'pad_mask_dict', 'timestep_pad_mask', 'task_completed' (torch.Size([16, 2, 4]), 'proprio' (fractal: torch.Size([16, 2, 8]))
        # task: 'language_instruction', 'pad_mask_dict', 'image_primary', 'image_wrist', 'timestep' (torch.Size([16]))
        # action (torch.Size([16, 2, 4, 7])
        # dataset_name
        # action_pad_mask (torch.Size([16, 2, 4, 7]))

        # timestep_pad_mask: which observations at the beginning of the trajectory are padding --- repeat the first observation at the beginning of the trajectory rather than going out of bounds
        # action_pad_mask: mark actions past the goal timestep as padding --- repeat the last action at the end of the trajectory rather than going out of bounds
        # task_completed should correspond to action_pad_mask
        # timestep should correspond to timestep_pad_mask (e.g., timestep [0, 0] for a datapoint indicates padding the first observation)
        images = _sample["observation"]["image_primary"]
        images = einops.rearrange(
            images, "B T H W C -> B (T C) H W"
        )  # remove cond_steps dimension
        texts = [
            text.decode("utf-8") for text in _sample["task"]["language_instruction"]
        ]
        actions = _sample["action"]
        proprios = _sample["observation"]["proprio"]  # pos, quat, gripper;

        tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-2B', add_eos_token=False, trust_remote_code=True, use_fast=False)
        # import ipdb;ipdb.set_trace()
        ret = process_batch(_sample, tokenizer, 'internvl2_5', transform, 
                dynamic_image_size=False, min_dynamic_patch=1, max_dynamic_patch=12,
                image_size=448, use_thumbnail=False, num_image_token=256,
                group_by_length=False, use_packed_ds=False, ds_name=None)
        sample_quat = torch.cat(
            (proprios[0, -1, -2:-1], proprios[0, -1, -5:-2])
        )  # quat [x, y, z, w] to [w, x, y, z]
        sample_rpy = quat2euler(sample_quat)
        num_unlabled_texts = len([t for t in texts if t == ""])
        print(num_unlabled_texts)

        # quat is in [x, y, z, w], and relative to robot base (unlike bridge that is relative to a top-down rotation). z is pointing forward/downward from the fingers, green is pointing left to the finger (sideway), and red is pointing away from the palm (pointing behind)

        # save an image
        img = Image.fromarray(images[0, :3].numpy().transpose(1, 2, 0))
        import ipdb;ipdb.set_trace()
        img.save("temp/fractal_sample_img_first1.png")
        img = Image.fromarray(images[0, -3:].numpy().transpose(1, 2, 0))
        img.save("temp/fractal_sample_img_last1.png")
        print(texts[0])
        print(actions[0, -1].numpy())
        print("w x y z", sample_quat.numpy())
        breakpoint()

        # check padding
        if not _sample["observation"]["timestep_pad_mask"].all():
            print("Padding for history obs past trajectory start")
        if not _sample["action_pad_mask"].all():
            print("Padding for action chunks past trajectory end")

        # verify the normalization
        if not args.skip_norm and (actions.abs().max() > 1 or proprios.abs().max() > 1):
            breakpoint()
        cnt_batch += 1
    load_time = time.time()
    print(f"Iterative over {cnt_batch} batches: {load_time - prep_time:.2f}s")
