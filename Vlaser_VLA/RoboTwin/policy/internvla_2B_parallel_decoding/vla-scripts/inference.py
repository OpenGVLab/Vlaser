import os
import pickle

# Sane Defaults


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['NCCL_ALGO'] = 'Tree'  # for A800
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys

sys.path.append(sys.path[0] + '/..')
import torch.nn.functional as F

import torch
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from transformers import AutoTokenizer

from models import Showo, ShowoInf, VQ_models, UnitModel, StateProjector, ActionProjector, ActionVQVAE, ImageProjector

from einops import rearrange
import json
import h5py

import cv2
import numpy as np
import imageio
from PIL import Image

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torchvision import transforms
from transformers import AutoProcessor
import itertools
from collections import defaultdict
from data_utils.internvl_process_img import load_image


def image_transform(image_size=224):
    transform = transforms.Compose([
        # transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform


def get_action_img_tokens(tokens, tokenizer, img_vq_model, action_vqvae_model, img_token_start_idx, act_token_start_idx, img_history_size=1):
    soi_token_id = tokenizer.convert_tokens_to_ids('<soi>')
    eoi_token_id = tokenizer.convert_tokens_to_ids('<eoi>')
    soa_token_id = tokenizer.convert_tokens_to_ids('<soa>')
    eoa_token_id = tokenizer.convert_tokens_to_ids('<eoa>')

    # 先得到 soa, eoa, soi, eoi在tokens中的位置，判断位置是否符合要求

    # <soa>***<eoa><soi>***<eoi>|<soa>***<eoa><soi>***<eoi>|<soa>***<eoa><soi>***<eoi>|<soa>***<eoa><soi>***<eoi>
    # 1. start with soa
    # 2. 0-th soi after 0-th eoa
    # 3. 1-th soa after 0-th eoi
    # 4. 256 tokens between soi and eoi

    special_positions = defaultdict(list)
    tokens = [i.item() for i in tokens]
    for idx, token in enumerate(tokens):
        if token in [soi_token_id, eoi_token_id, soa_token_id, eoa_token_id]:
            special_positions[token].append(idx)

    soa_positions = special_positions[soa_token_id]
    eoa_positions = special_positions[eoa_token_id]
    soi_positions = special_positions[soi_token_id]
    eoi_positions = special_positions[eoi_token_id]
    if tokens[0] != soa_token_id:
        return False, None, None
    for i in range(img_history_size):
        if eoa_positions[i] + 1 != soi_positions[i]:
            return False, None, None
        if eoa_positions[i] !=soa_positions[i]+img_history_size*4*16+1:  # 4代表action chunks 16压缩4倍，16是左臂和右臂各6
            return False, None, None
        # if i <= 2:
        #     if eoi_positions[i] + 1 != soa_positions[i + 1]:
        #         return False, None, None
        if eoi_positions[i] - soi_positions[i] != 256 + 1:
            return False, None, None

    action_token_lists = []
    img_token_lists = []
    # 通过所有条件，然后我们从中得到action tokens和image tokens
    for i in range(img_history_size):
        action_token_lists.append(tokens[soa_positions[i] + 1:eoa_positions[i]])
        img_token_lists.append(tokens[soi_positions[i] + 1:eoi_positions[i]])

    imgs, actions = decode_tokens(action_token_lists, img_token_lists, img_vq_model, action_vqvae_model, img_token_start_idx,
                                  act_token_start_idx)
    return True, imgs, actions



def get_action_tokens_only(tokens, tokenizer, img_vq_model, action_vqvae_model, img_token_start_idx,
                                  act_token_start_idx):
    tokenizer.left_arm_soa_token_id = tokenizer.convert_tokens_to_ids('<left_arm_soa>')
    tokenizer.left_arm_eoa_token_id = tokenizer.convert_tokens_to_ids('<left_arm_eoa>')
    tokenizer.right_arm_soa_token_id = tokenizer.convert_tokens_to_ids('<right_arm_soa>')
    tokenizer.right_arm_eoa_token_id = tokenizer.convert_tokens_to_ids('<right_arm_eoa>')

    # <left_arm_soa>act<left_arm_eoa>|<left_arm_soa>act<left_arm_eoa>|<left_arm_soa>act<left_arm_eoa>|<left_arm_soa>act<left_arm_eoa>|<right_arm_soa>act<right_arm_eoa>...
    special_positions = defaultdict(list)
    for idx, token in enumerate(tokens):
        token = token.item()
        if token in [tokenizer.left_arm_soa_token_id, tokenizer.left_arm_eoa_token_id,
                     tokenizer.right_arm_soa_token_id, tokenizer.right_arm_eoa_token_id]:
            special_positions[token].append(idx)

    for i in range(4):
        assert special_positions[tokenizer.left_arm_eoa_token_id][i]-special_positions[tokenizer.left_arm_soa_token_id][i] == 8+1
        assert special_positions[tokenizer.right_arm_eoa_token_id][i]-special_positions[tokenizer.right_arm_soa_token_id][i] == 8+1

    left_action_tokens = [tokens[special_positions[tokenizer.left_arm_soa_token_id][i]+1:special_positions[tokenizer.left_arm_eoa_token_id][i]-1]  for i in range(4)]
    right_action_tokens = [tokens[special_positions[tokenizer.right_arm_soa_token_id][i]+1:special_positions[tokenizer.right_arm_eoa_token_id][i]-1]  for i in range(4)]

    left_action_tokens = torch.tensor(left_action_tokens).to(device).unsqueeze(0) - act_token_start_idx
    right_action_tokens = torch.tensor(right_action_tokens).to(device).unsqueeze(0) - act_token_start_idx



    left_actions = action_vqvae_model.idx_to_action(left_action_tokens)
    right_actions = action_vqvae_model.idx_to_action(right_action_tokens)
    final_actions = torch.cat((left_actions, right_actions), dim=-1)
    return True, None, final_actions


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def decode_tokens(action_token_lists, img_token_lists, img_vq_model, action_vqvae_model, img_token_start_idx, act_token_start_idx):
    # action_token_list = [[token - act_token_start_idx for token in action_token_list] for action_token_list in
    #                      action_token_lists]
    img_token_lists = [[token - img_token_start_idx for token in img_token_list] for img_token_list in img_token_lists]
    act_token_lists = [[token - act_token_start_idx for token in act_token_list] for act_token_list in action_token_lists]

    re_normalize = NormalizeInverse(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    pil_transform = transforms.ToPILImage()
    imgs = []
    if not os.path.exists("predicted_img"):
        os.mkdir("predicted_img")
    for idx, img_token_list in enumerate(img_token_lists):
        _, img = img_vq_model.decode_code(img_token_list, (1, 40, 16, 16))
        img = img.float()  # numpy doesn't support bf16
        img = pil_transform(re_normalize(img[0]))
        imgs.append(img)
        img.save(f"predicted_img/{idx}_step.png")
    act_token_lists = torch.tensor(act_token_lists).to(img_vq_model.device).view(1, 4, 16)
    left_actions = action_vqvae_model.idx_to_action(act_token_lists[:, :, :8])
    right_actions = action_vqvae_model.idx_to_action(act_token_lists[:, :, 8:])
    final_actions = torch.cat((left_actions, right_actions), dim=-1)
    return imgs, final_actions


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    vla_model_path = "/oss/tmp_data_weights/openvla-main_mine_delta_action_pai0/action_siglip384_v1/saved_ckpt/checkpoint-80000/unwrapped_model"


    # checkpoint_1 = torch.load("/oss/openvla-main_mine_v8_vla/show-o-tuning-stage1/checkpoint-110000/unwrapped_model/pytorch_model.bin", map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained(vla_model_path, padding_side="left")
    print(f"there are {len(tokenizer)} tokens in the trained model")

    tokenizer.soi_token_id = tokenizer.convert_tokens_to_ids('<soi>')
    tokenizer.eoi_token_id = tokenizer.convert_tokens_to_ids('<eoi>')
    tokenizer.sot_token_id = tokenizer.convert_tokens_to_ids('<sot>')
    tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids('<eot>')
    tokenizer.soa_token_id = tokenizer.convert_tokens_to_ids('<soa>')
    tokenizer.eoa_token_id = tokenizer.convert_tokens_to_ids('<eoa>')
    tokenizer.left_arm_soa_token_id = tokenizer.convert_tokens_to_ids('<left_arm_soa>')
    tokenizer.left_arm_eoa_token_id = tokenizer.convert_tokens_to_ids('<left_arm_eoa>')
    tokenizer.right_arm_soa_token_id = tokenizer.convert_tokens_to_ids('<right_arm_soa>')
    tokenizer.right_arm_eoa_token_id = tokenizer.convert_tokens_to_ids('<right_arm_eoa>')


    tokenizer.left_arm_sost_token_id = tokenizer.convert_tokens_to_ids('<left_arm_sost>')
    tokenizer.left_arm_eost_token_id = tokenizer.convert_tokens_to_ids('<left_arm_eost>')
    tokenizer.right_arm_sost_token_id = tokenizer.convert_tokens_to_ids('<right_arm_sost>')
    tokenizer.right_arm_eost_token_id = tokenizer.convert_tokens_to_ids('<right_arm_eost>')


    tokenizer.ignore_id = -100

    # action_vq_model_codebook_size = 64 # delte
    # act_token_start_idx = tokenizer(["<act_1>"]).input_ids[0][0]
    # act_token_end_idx = tokenizer([f"<act_{action_vq_model_codebook_size}>"]).input_ids[0][0]

    state_token_start_idx = tokenizer(["<state_1>"]).input_ids[0][0]


    # TODO 
    # img encoder
    img_encoder = SiglipVisionModel.from_pretrained("/mnt/workspace/hf_models/siglip-so400m-patch14-384").to(device, dtype=torch.bfloat16)



    img_encoder.requires_grad_(False)



    # action tokenizer
    # action_vqvae_model = ActionVQVAE(
    #     action_dim=7,
    #     quantizer='multicodebook',
    #     codebook_size=64,
    #     codebook_dim=2048,
    #     n_latent_dims=512,
    #     down_t=2,
    #     stride_t=2,
    #     width=512,
    #     depth=3,
    #     dilation_growth_rate=3,
    #     norm=None,
    #     activation="relu",
    #     num_codebooks=8,
    #     quant_proj='attn',
    # )
    # action_vqvae_model.load_state_dict(torch.load(
    #     "/oss/trained_action_tokenizer/action_tokenizer_mine_v1/DWT_VAE_new_v5_v5/checkpoints/DWT_VAE_new_v5_v5/trained_vqvae_24.pt",
    #         weights_only=False))
    # action_vqvae_model.eval().to(device, dtype=torch.bfloat16)
    # action_vqvae_model.requires_grad_(False)

    
    vlm = Qwen2ForCausalLM.from_pretrained(
        vla_model_path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=False,  # if set low_cpu_mem_usage to True, there will be something wrong with resize_token_embeddings https://github.com/huggingface/accelerate/issues/1620#issuecomment-2413317047
        # attn_implementation="flash_attention_2" #"eager", # "flash_attention_2"
    )
    # vlm.resize_token_embeddings(len(tokenizer))
    vla = Showo(vlm)



    state_action_projector_weights = torch.load(vla_model_path + "/projector.pth", map_location="cpu", weights_only=False)
    action_projector_weights = state_action_projector_weights['action_projector']
    image_projector_weights = state_action_projector_weights['state_projector']

    image_projector = ImageProjector(input_dim=1152, output_dim=896)
    action_projector = ActionProjector(input_dim=256, output_dim=896)

    # state_projector.load_state_dict(state_projector_weights)
    action_projector.load_state_dict(action_projector_weights)
    image_projector.load_state_dict(image_projector_weights)

    model = UnitModel(
        vla,
        action_projector,
        image_projector
    )

    model.to(device, dtype=torch.bfloat16).eval()



    test_input_files = json.load(open("action_vae_val_files.json", "r"))[0:1]
    # instructions = ["Arm movement to pick up container and place on plate."]
    instructions = ["Use arm movement to pick up hammer and hit block on table."]
    image_transform = image_transform(384)
    idx = 0

    for file, instruction in zip(test_input_files, instructions):
        with h5py.File(file, 'r') as f:
            qpos = f['observations']['qpos'][:]
            img = f['observations']['images']["cam_high"][idx]
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        img.save("org.png")

        print(qpos.shape[0])
        # imgs: size: img_history_size, actions: size: img_history_size*action_chunk_size+1
        pixel_values_steps = [image_transform(img)]

        input_text = instruction.lower()

        input_text_ids = np.array(tokenizer([input_text]).input_ids)

        init_state = qpos[idx]
        # init state也需要过一遍action tokenizer, 因为init state和action的representation相同
        # state_ids = action_tokenizer(init_state)
        # state_ids = action_tokenizer(init_state)[
        #             :1]  # 因为输入的init_state的长度为1，action_tokenizer会自动将长度复制为2，最终是2个相同的输出，我们只用1个表示state
        #
        # # 使用 itertools.accumulate 进行累加
        # state_end_idx = [0]
        # state_end_idx.extend([len(state) for state in state_ids])
        # state_end_idx = list(itertools.accumulate(state_end_idx))  # 记录每个state在list中的结束index
        # state_ids = sum(state_ids, [])  # list[list]-> [list]
        #
        # state_ids = np.array(state_ids) + act_token_start_idx

        input_text_ids = torch.from_numpy(input_text_ids).long().to(device)
        pixel_values_steps = torch.stack(pixel_values_steps).to(device, dtype=torch.bfloat16)

        # 处理states
        states = []
        # np.linspace会生成包含state_min, state_max在内的state_vocab_size-1个数
        bins = np.linspace(-2.5, 3.0, 256 - 1)
        state = np.clip(init_state, -2.5, 3.0)  # 截断state的取值范围
        # discretized_state后的结果不-1，因为我们做了截断处理，state的最小值经过np.digitize后的index是0，最大值对应的index则是state_vocab_size
        discretized_state = np.digitize(state, bins) + state_token_start_idx
        states.append(torch.from_numpy(discretized_state))
        state_ids = torch.stack(states).long()  # init state ids

        left_arm_state_start = torch.ones((1, 1)).long() * tokenizer.left_arm_sost_token_id
        left_arm_state_end = torch.ones((1, 1)).long() * tokenizer.left_arm_eost_token_id
        right_arm_state_start = torch.ones((1, 1)).long() * tokenizer.right_arm_sost_token_id
        right_arm_state_end = torch.ones((1, 1)).long() * tokenizer.right_arm_eost_token_id
        # 增加state分隔符
        state_ids = torch.cat(
            [
                left_arm_state_start,
                state_ids[:, :7],
                left_arm_state_end,
                right_arm_state_start,
                state_ids[:, 7:],
                right_arm_state_end
            ], dim=1
        )
        state_ids = state_ids.to(device, non_blocking=True)


        # state_end_idx = torch.tensor([state_end_idx])  # 记录init state ids的长度
        batch_size = 1
        img_input_len = 1
        # with torch.no_grad():
        #     image_features, _, infos = vq_model.encode(
        #         pixel_values_steps)  # [batch*img_input_len, 40, 16,16], img_input_len是1
        #
        #     images_feat = rearrange(image_features, 'b c h w -> b (h w) c')  # [batch*img_input_len, 256, 40]
        #     images_discrete_tokens = infos[-1]  # # [batch*img_input_len, 16， 16]
        #     images_discrete_tokens = rearrange(images_discrete_tokens, 'b h w -> b (h w)')  # [batch*img_input_len, 256]
        #     # check whether images_discrete_tokens idx start from 0
        #     images_discrete_tokens = (images_discrete_tokens + img_token_start_idx).contiguous().view(1, -1,
        #                                                                                               images_discrete_tokens.size()[
        #                                                                                                   -1])  # # [batch, img_input_len, 256]
        #     observation_img_tokens = images_discrete_tokens[:, 0]  # [batch, 256]

        # 将下面的tensor全部转换到cpu上进行cat操作，是因为对于toch.cat, cpu比gpu快很多
        img_embeddings = img_encoder(pixel_values_steps, output_hidden_states=True).hidden_states[-2]
        img_embeddings = model.image_projector(img_embeddings)

        text_embeddings = model.vla.model.model.embed_tokens(input_text_ids)  # [batch, len, 896]
        img_embeddings = img_embeddings.contiguous().view(
            batch_size, img_input_len, -1,
            text_embeddings.shape[-1]) # [batch

        state_embeddings = model.vla.model.model.embed_tokens(state_ids)  # [batch, 14, 896]

        # state_embeddings = model.vlwa.showo.model.embed_tokens(state_ids)  # [batch, len, 896]

        observation_img_embeddings = img_embeddings[:, 0]  # [batch, 256, 896]

        text_start_embeddings = model.vla.model.model.embed_tokens(
            torch.ones(batch_size, 1).long().to(device) * tokenizer.sot_token_id)
        text_end_embeddings = model.vla.model.model.embed_tokens(
            torch.ones(batch_size, 1).long().to(device) * tokenizer.eot_token_id)

        img_start_embeddings = model.vla.model.model.embed_tokens(
            torch.ones(batch_size, 1).long().to(device) * tokenizer.soi_token_id)
        img_end_embeddings = model.vla.model.model.embed_tokens(
            torch.ones(batch_size, 1).long().to(device) * tokenizer.eoi_token_id)

        instance_start_embeddings = model.vla.model.model.embed_tokens(
            torch.ones(batch_size, 1).long().to(device) * tokenizer.bos_token_id)
        instance_end_embeddings = model.vla.model.model.embed_tokens(
            torch.ones(batch_size, 1).long().to(device) * tokenizer.eos_token_id)






        #  batch为1，就不需要再考虑padding的问题
        input_embedding = torch.cat([
            instance_start_embeddings,  # [1, 1, dim],
            text_start_embeddings,  # [1, 1, dim]
            text_embeddings,  # [1, len, dim]
            # :input_text_length代表padding之前的text embedding
            text_end_embeddings,
            state_embeddings,
            img_start_embeddings,
            observation_img_embeddings,
            img_end_embeddings,
        ], dim=1).to(device)  # [1, len, dim]
        attention_mask = torch.ones(batch_size, input_embedding.size(1)).to(input_embedding.device)
        retry_num = 5
        pred_img = False
        if pred_img:
            eot_token = [tokenizer.eos_token_id,
                         tokenizer.pad_token_id]  # 如果不包含tokenizer.eoa_token_id，即需要生成image
        else:
            eot_token=[tokenizer.eos_token_id,
                       ]  # 这里我们只需要生成action即可，不需要生成image

        
        
        # TODO : change to mmu_generate_img_actions_parallel_decoding

        out_tokens = model.vla.mmu_generate_img_actions(input_embeddings=input_embedding, attention_mask=attention_mask,
                                             temperature=0.2, max_new_tokens=2000,
                                             eot_token=eot_token,
                                             img_token_start_idx=None,
                                             img_token_end_idx=None,
                                             img_embedding_weight=None,
                                             action_vqvae_model=action_vqvae_model,
                                             act_token_start_idx=act_token_start_idx,
                                             act_token_end_idx=act_token_end_idx,
                                             action_num_codebooks=8,
                                             action_projector=model.action_projector
                                             )
        if pred_img:
            suc, imgs, action_token_list = get_action_img_tokens(
                out_tokens, tokenizer, None, action_vqvae_model, None,
                act_token_start_idx)
        else:
            suc, imgs, action_token_list = get_action_tokens_only(
                out_tokens, tokenizer, None, action_vqvae_model, None,
                act_token_start_idx)
        print(action_token_list)