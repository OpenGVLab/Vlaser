from transformers import AutoTokenizer
# from internvl.vla.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
#                                       IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
#                                       IMG_START_TOKEN, QUAD_END_TOKEN,
#                                       QUAD_START_TOKEN, REF_END_TOKEN,
#                                       REF_START_TOKEN, PROPRIO_CONTEXT_TOKEN, PROPRIO_START_TOKEN, PROPRIO_END_TOKEN)
from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from internvl.model.internvl_chat import InternVLA_Model, InternVLAForActionPrediction, InternVLChatModel
import torch
from transformers import PreTrainedTokenizerBase


def load(model_args, data_args):

    if model_args.model_name_or_path is not None:
        print('Loading InternVLA_Model...')
        # print(f"!!! DEBUG: 正在从以下路径加载模型: {model_args.model_name_or_path}")
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            print('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            print('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        # or InternVLAModel
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    else:
        raise ValueError('model_name_or_path is required')

    assert model.config.downsample_ratio == data_args.down_sample_ratio
    
    # if model_args.mlp_path is not None:
    #     print('Loading pretrained MLP projector...')
    #     state_dict = torch.load(model_args.mlp_path, map_location='cpu')
    #     message = model.mlp1.load_state_dict(state_dict)
    #     print(f'MLP loading result: {message}')

    # === Setup model dimensions and tokenizer ===
    patch_size = model.config.vision_config.patch_size
    if model.config.vision_config.image_size != data_args.force_image_size:
        print(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))


    return model


# def load_internvla(model_args, data_args):
#     # Load pretrained model, tokenizer, and image processor
#     print(f"Loading VLA Model from checkpoint: {model_args.model_name_or_path}")
#     tokenizer_path = model_args.model_name_or_path
#     print(f'Loading Tokenizer: {tokenizer_path}')
#     tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
#         tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=model_args.use_fast_tokenizer)
#     tokenizer.pad_token_id = 0
#     tokenizer.tokenizer_path = tokenizer_path
#     tokenizer.model_max_length = data_args.max_seq_length
#     img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
#     proprio_context_token_id = tokenizer.convert_tokens_to_ids(PROPRIO_CONTEXT_TOKEN)

#     if model_args.model_name_or_path is not None:
#         print('Loading InternVLA_Model...')
#         config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
#         config.vision_config.drop_path_rate = model_args.drop_path_rate
#         if config.llm_config.model_type == 'internlm2':
#             config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
#             print('Using flash_attention_2 for InternLM')
#         else:
#             config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
#             print('Using flash_attention_2 for LLaMA')
#         config.template = data_args.conv_style
#         config.select_layer = model_args.vision_select_layer
#         config.dynamic_image_size = data_args.dynamic_image_size
#         config.use_thumbnail = data_args.use_thumbnail
#         config.ps_version = model_args.ps_version
#         config.min_dynamic_patch = data_args.min_dynamic_patch
#         config.max_dynamic_patch = data_args.max_dynamic_patch
#         model = InternVLAForActionPrediction.from_pretrained(
#             model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
#     else:
#         raise ValueError('model_name_or_path is required')
#     model.img_context_token_id = img_context_token_id
#     model.proprio_context_token_id = proprio_context_token_id

#     assert model.config.downsample_ratio == data_args.down_sample_ratio
    
#     if model_args.mlp_path is not None:
#         print('Loading pretrained MLP projector...')
#         state_dict = torch.load(model_args.mlp_path, map_location='cpu')
#         message = model.mlp1.load_state_dict(state_dict)
#         print(f'MLP loading result: {message}')

#     # === Setup model dimensions and tokenizer ===
#     patch_size = model.config.vision_config.patch_size
#     if model.config.vision_config.image_size != data_args.force_image_size:
#         print(f'Resizing position embedding from '
#                     f'{model.config.vision_config.image_size} '
#                     f'to {data_args.force_image_size}...')
#         model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
#                                                  new_size=data_args.force_image_size,
#                                                  patch_size=patch_size)
#         model.config.vision_config.image_size = data_args.force_image_size
#     model.config.force_image_size = data_args.force_image_size
#     model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))


#     # === Setup training configurations ===
#     model.language_model.config.use_cache = False
#     model.vision_model.gradient_checkpointing = True
#     model.vision_model.encoder.gradient_checkpointing = True

#     # Set device
#     return model, tokenizer
