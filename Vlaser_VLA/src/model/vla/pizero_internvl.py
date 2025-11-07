"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""

import logging
from typing import Optional, Tuple

import hydra
import torch
from torch import nn
import sys

from src.model.vla.processing import InternVLAProcessor
from src.model.vla.mixture import MixtureAttention
from src.model.kv_cache import KVCache
from src.model.vla.modules import (
    ActionEncoder,
    SinusoidalPosEmb,
)
import os
from src.utils.decorator import NoSyncBase
from src.utils.monitor import log_execution_time
from transformers import AutoConfig
log = logging.getLogger(__name__)
import copy

def get_internvl3(pretrained_model_path='OpenGVLab/InternVL3-2B', model_size="2B", image_448 = True, debug_joint=False, debug_1536=False):
    print(pretrained_model_path)
    import sys
    
    current_dir = os.getcwd()
    sys.path.append(os.path.join(current_dir, "src/model/internvl_chat"))

    from internvl.model.internvl_chat import InternVLChatModel
    from transformers import AutoTokenizer
    from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig        
    tokenizer_path = pretrained_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=False,  trust_remote_code=True, use_fast=False,use_flash_attn=False, )

    if not debug_joint:
        max_new_tokens = 256
        action_token_list = ['<a{}>'.format(i) for i in range(max_new_tokens)]
        num_new_tokens = tokenizer.add_tokens(action_token_list, special_tokens=True)
        
    from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                    IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                    IMG_START_TOKEN, QUAD_END_TOKEN,
                                    QUAD_START_TOKEN, REF_END_TOKEN,
                                    REF_START_TOKEN)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = 256

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    
    sep_id = tokenizer.convert_tokens_to_ids(',')
    

    if 'InternVL3_5' in pretrained_model_path:
        config = AutoConfig.from_pretrained(pretrained_model_path, trust_remote_code=True)
        from transformers import Qwen3ForCausalLM as Qwen2ForCausalLM
    else:
        print(pretrained_model_path)
        config = InternVLChatConfig.from_pretrained(pretrained_model_path)
        from transformers import Qwen2ForCausalLM
    
    config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
    
    config.template = 'internvl2_5'
    config.select_layer = -1
    config.dynamic_image_size = False
    config.use_thumbnail = False
    config.ps_version = 'v2'
    config.llm_config.action_expert_hidden_ratio = 4
    config.n_action_steps = 4 + 1
    
    vlm = InternVLChatModel.from_pretrained(
        pretrained_model_path, torch_dtype=torch.bfloat16, config=config, use_flash_attn=False, ignore_mismatched_sizes=False)

    if not debug_joint:
        vlm.language_model.resize_token_embeddings(len(tokenizer))
    

    vlm.config.llm_config.vocab_size = len(tokenizer)
    vlm.language_model.config.vocab_size = len(tokenizer)

    # vlm = vlm.to(torch.float32)
    vlm.language_model.config.use_cache = False
    vlm.vision_model.gradient_checkpointing = True
    vlm.vision_model.encoder.gradient_checkpointing = True

    vlm.img_context_token_id = img_context_token_id

    patch_size = vlm.config.vision_config.patch_size
    # import ipdb;ipdb.set_trace()
    if not image_448:
        vlm.vision_model.resize_pos_embeddings(old_size=448,
                                                    new_size=224,
                                                    patch_size=patch_size)
    
    
    vlm.language_model._set_gradient_checkpointing()


        
    print('use pretraining internvl-------------------------', flush=True)


    # from transformers import Qwen2ForCausalLM
    # import ipdb;ipdb.set_trace()
    
    vlm.action_expert_config = copy.deepcopy(vlm.config.llm_config)
    # if 'DEBUG' in os.environ: import ipdb;ipdb.set_trace()
    
    if model_size == "2B":
        if not debug_1536:
            vlm.action_expert_config.hidden_size = 768
            vlm.action_expert_config.intermediate_size = 8960
            vlm.action_expert_config.head_dim=128
        else:
            vlm.action_expert_config.hidden_size = 768
            vlm.action_expert_config.intermediate_size = 4096
            vlm.action_expert_config.head_dim=128
    elif model_size == "8B":
        vlm.action_expert_config.hidden_size = 768
        vlm.action_expert_config.intermediate_size = 5600
        vlm.action_expert_config.head_dim = 128
    
    
    vlm.action_expert = Qwen2ForCausalLM(vlm.action_expert_config)
    vlm.action_expert.model.embed_tokens = None




    vlm.action_in_proj = torch.nn.Linear(7, vlm.action_expert_config.hidden_size).to(torch.bfloat16)
    vlm.action_time_mlp_in = torch.nn.Linear(vlm.action_expert_config.hidden_size*2, vlm.action_expert_config.hidden_size).to(torch.bfloat16)
    vlm.action_time_mlp_out = torch.nn.Linear(vlm.action_expert_config.hidden_size, vlm.action_expert_config.hidden_size).to(torch.bfloat16)
    vlm.action_out_proj = torch.nn.Linear(vlm.action_expert_config.hidden_size, 7).to(torch.bfloat16)

    vlm.action_in_proj = None
    vlm.action_time_mlp_in = None
    vlm.action_time_mlp_out = None
    vlm.action_out_proj = None
    
    model = vlm
    model.tokenizer = tokenizer
    return model

class PiZero(nn.Module, NoSyncBase):
    @log_execution_time(log)
    def __init__(self, cfg, use_ddp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_ddp = use_ddp  # used in NoSyncBase
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.use_lm_head = cfg.get("use_lm_head", False)
        self.integration_method = cfg.get("integration_method", "euler")


        self.imgfeat = cfg.get("indi_imgfeat", False)
        if 'DEBUG_IMGFEAT' in os.environ:
            self.imgfeat = True
        else:
            self.imgfeat = False
        if 'DEBUG_VLM' in os.environ:
            self.debug_vlm = True
        else:
            self.debug_vlm = False
        if 'DEBUG_CAUSAL' in os.environ:
            self.debug_causal = True
        else:
            self.debug_causal = False
        if 'IMAGE_448' in os.environ:
            self.image_448 = True
        else:
            self.image_448 = False
        if 'DEBUG_JOINT' in os.environ:
            self.debug_joint = True
        else:
            self.debug_joint = False
            
        if 'NO_CAUSAL_IMG' in os.environ:
            self.NO_CAUSAL_IMG = True
        else:
            self.NO_CAUSAL_IMG = False
        if 'NO_IMG' in os.environ:
            self.no_img = True
        else:
            self.no_img = False
        if 'DEBUG_1536' in os.environ:
            self.debug_1536 = True
        else:
            self.debug_1536 = False
        if 'BUG' in os.environ:
            self.bug = True
        else:
            self.bug = False
        

        self.max_image_text_tokens = cfg.max_image_text_tokens
        self.num_proprio_tokens = cfg.cond_steps
        self.num_proprio_tokens = 1 # hard code, if muiltiple proprio, remove this and modify other hard code in mask building
        self.num_action_tokens = cfg.horizon_steps + cfg.cond_steps - 1
        self.total_num_tokens = (
            self.max_image_text_tokens
            + self.num_proprio_tokens
            + self.num_action_tokens
        )

        self.image_text_hidden_size = cfg.mixture.vlm.hidden_size
        self.proprio_hidden_size = cfg.mixture.proprio.hidden_size
        self.action_hidden_size = cfg.mixture.action.hidden_size

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps + cfg.cond_steps - 1
        self.action_dim = cfg.action_dim
        self.proprio_dim = cfg.proprio_dim
        self.final_action_clip_value = cfg.final_action_clip_value
        self.flow_sig_min = cfg.get("flow_sig_min", 0.001)

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

       
        
        internvl_model = get_internvl3(cfg.pretrained_model_path, model_size=cfg.get('model_size', "2B"), debug_joint=self.debug_joint,image_448=self.image_448, debug_1536=self.debug_1536)
        
        
        
        self.vision_tower = hydra.utils.instantiate(cfg.vision)
        self.multi_modal_projector = hydra.utils.instantiate(cfg.vision_projector)
        
        total_params = sum(p.numel() for p in internvl_model.action_expert.parameters())
        print(f"Number of parameters in action expert: {total_params / 1024 / 1024} M")
        trainable_params = sum(
            p.numel() for p in internvl_model.action_expert.parameters() if p.requires_grad
        )
        print(f"Trainable params in action expert: {trainable_params / 1024 / 1024}M")
    
        # Mixtures
        self.joint_model = hydra.utils.instantiate(cfg.joint)
        self.joint_model.mixtures.vlm.layers = internvl_model.language_model.model.layers
        self.joint_model.mixtures.proprio.layers = internvl_model.action_expert.model.layers
        self.joint_model.mixtures.action.layers = internvl_model.action_expert.model.layers


        self.joint_model.mixtures.vlm.norm = internvl_model.language_model.model.norm
        # self.joint_model.mixtures.vlm.lm_head = internvl_model.language_model.lm_head
        self.joint_model.mixtures.proprio.norm = internvl_model.action_expert.model.norm
        self.joint_model.mixtures.action.norm = internvl_model.action_expert.model.norm
        
          
        if not self.debug_vlm:
            internvl_model.language_model.model.layers = None
            internvl_model.action_expert.model.layers = None

        
        # internvl_model.proprio_expert.model.lm_head = None
        self.multi_modal_projector = internvl_model.mlp1
        if self.imgfeat:
           
            self.multi_modal_projector1 = copy.deepcopy(self.multi_modal_projector)
            self.multi_modal_projector1 = torch.nn.Sequential(self.multi_modal_projector1[0], self.multi_modal_projector1[1], self.multi_modal_projector1[2], torch.nn.Linear(in_features=1536, out_features=768, bias=True),)
            
            pass
        
        if not self.debug_joint and not self.debug_vlm:
            internvl_model.mlp1 = None
        
        
        self.vision_tower.vision_model = internvl_model.vision_model
        
        if self.imgfeat:
            self.vision_tower1 = copy.deepcopy(self.vision_tower)
        self.embed_tokens = internvl_model.language_model.model.embed_tokens
        self.internvl_model = internvl_model
        self.num_image_token = internvl_model.num_image_token
        
        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if cfg.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(
                cfg.time_hidden_size, cfg.time_max_period
            )
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(
                self.action_hidden_size, cfg.time_max_period
            )
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Action decoder
        self.action_decoder = nn.Linear(
            self.action_hidden_size,
            self.action_dim,
        )
        if self.debug_joint or self.debug_vlm:
            self.use_lm_head = True

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights
            
            self.lm_head = internvl_model.language_model.lm_head
            self.lm_head = self.internvl_model.language_model.lm_head
        self.vocab_size = self.internvl_model.language_model.config.vocab_size
        self.num_images = cfg.cond_steps
        

    @property
    def action_expert_parameters_only(self):
        if self.imgfeat:
            return (
                list(self.action_encoder.parameters())
                + list(self.action_decoder.parameters())
                + list(self.proprio_encoder.parameters())
                + list(self.joint_model.mixtures["action"].parameters())
                + list(self.multi_modal_projector1[-1].parameters())
            )  # note: action and proprio share weights
            
        else:
            return (
                list(self.action_encoder.parameters())
                + list(self.action_decoder.parameters())
                + list(self.proprio_encoder.parameters())
                + list(self.joint_model.mixtures["action"].parameters())
            )  # note: action and proprio share weights


    @property
    def action_expert_parameters(self):
        if self.imgfeat:
            return (
                list(self.action_encoder.parameters())
                + list(self.action_decoder.parameters())
                + list(self.proprio_encoder.parameters())
                + list(self.joint_model.mixtures["action"].parameters())
                # + list(self.multi_modal_projector1[-1].parameters())
            )  # note: action and proprio share weights
            
        else:
            return (
                list(self.action_encoder.parameters())
                + list(self.action_decoder.parameters())
                + list(self.proprio_encoder.parameters())
                + list(self.joint_model.mixtures["action"].parameters())
            )  # note: action and proprio share weights
            
    @property
    def action_expert_parameters_debug(self):
        print(self.joint_model.mixtures["action"])
        if self.imgfeat:
            return (
                list(self.action_encoder.parameters())
                + list(self.action_decoder.parameters())
                + list(self.proprio_encoder.parameters())
                + list(self.joint_model.mixtures["action"].parameters())
                + list(self.vision_tower.parameters())
                + list(self.multi_modal_projector.parameters())
                + list(self.multi_modal_projector1.parameters())
                # + list(self.multi_modal_projector1[-1].parameters())
            )  # note: action and proprio share weights
            
        else:
            return (
                list(self.action_encoder.parameters())
                + list(self.action_decoder.parameters())
                + list(self.proprio_encoder.parameters())
                + list(self.joint_model.mixtures["action"].parameters())
                + list(self.vision_tower.parameters())
                + list(self.multi_modal_projector.parameters())
                # + list(self.joint_model.mixtures["vlm"].parameters())
            )  # note: action and proprio share weights

    @property
    def trainable_vlm_parameters(self):
        
        return list(self.internvl_model.parameters())
    
    @property
    def trainable_vision_parameters(self):
        if self.imgfeat:
            return (
                list(self.vision_tower1.parameters())
                + list(self.multi_modal_projector1.parameters())
            )
        else:
            return (
                list(self.vision_tower.parameters())
                + list(self.multi_modal_projector.parameters())
            )

    @property
    def lora_trainable_vlm_parameters(self):
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        for name, param in self.multi_modal_projector.named_parameters():
            if "lora_" in name:
                params.append(param)
        params.extend(self.trainable_lora_gemma_parameters)
        return params

    @property
    def trainable_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            # if not self._check_gemma_unused_parameter_by_name(name):
            gemma_parameters.append(param)
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters

    @log_execution_time(log)
    def load_pretrained_weights(self):
        return 
        

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

    def freeze_non_lora_weights_in_vlm(self):
        """Keep all bias frozen"""
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in vision tower")

        for name, param in self.multi_modal_projector.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in lm part of the joint model")

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        # for name, param in self.joint_model.mixtures["vlm"].named_parameters():
        #     if self._check_gemma_unused_parameter_by_name(name):
        #         param.requires_grad = False
        last_layer_num_id = len(self.joint_model.mixtures["vlm"].layers) - 1
        self.joint_model.mixtures["vlm"].layers[last_layer_num_id].post_attention_layernorm.weight.requires_grad = False
        self.joint_model.mixtures["vlm"].layers[last_layer_num_id].mlp.down_proj.weight.requires_grad = False
        self.joint_model.mixtures["vlm"].layers[last_layer_num_id].mlp.up_proj.weight.requires_grad = False
        self.joint_model.mixtures["vlm"].layers[last_layer_num_id].mlp.gate_proj.weight.requires_grad = False
        self.joint_model.mixtures["vlm"].layers[last_layer_num_id].self_attn.o_proj.weight.requires_grad = False
        self.internvl_model.action_expert.lm_head.weight.requires_grad = False
        # self.internvl_model.action_expert.model.norm.weight.requires_grad = False
        self.internvl_model.language_model.lm_head.weight.requires_grad = False
        self.internvl_model.language_model.model.norm.weight.requires_grad = False


    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    def build_text_cache(self):
        return KVCache()

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

                 img/text img/text img/text (padding) proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        proprio     x        x        x                 x
        action      x        x        x                 x       x      x
        action      x        x        x                 x       x      x
        """
        bsz = attention_mask.size(0)
        proprio_start = self.max_image_text_tokens
        proprio_end = self.max_image_text_tokens + self.num_proprio_tokens
        action_start = proprio_end
        image_text_token_cnts = torch.sum(attention_mask, dim=1)

        if not self.debug_causal:
            causal_mask = torch.full(
                (bsz, attention_mask.shape[-1] + self.num_action_tokens + 1, attention_mask.shape[-1] + self.num_action_tokens + 1),
                torch.finfo(dtype).min,
                dtype=dtype,
            )  # smallest value, avoid using inf for softmax nan issues with padding
            for idx, cnt in enumerate(image_text_token_cnts):
                causal_mask[idx, :cnt, :cnt] = 0  # image/text attend to itself
                causal_mask[idx, proprio_start:, :cnt] = (
                    0  # proprio/action attend to image/text
                )
            causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
                0  # proprio attend to itself
            )
            causal_mask[:, action_start:, proprio_start:] = (
                0  # action attend to itself and proprio
            )

            # add the head dimension
            # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
            causal_mask = causal_mask.unsqueeze(1)
        else:
            # L = attention_mask.shape[-1] + self.num_action_tokens + 1
            # causal_mask = torch.triu(
            #     torch.full((L, L), torch.finfo(dtype).min, dtype=dtype),
            #     diagonal=1
            # )
            # causal_mask = causal_mask.unsqueeze(0).expand(bsz, -1, -1)
            # causal_mask = causal_mask.unsqueeze(1)
            causal_mask = torch.cat([attention_mask, 
                                     torch.ones([len(attention_mask), self.num_action_tokens + 1], 
                                                dtype=attention_mask.dtype, device=attention_mask.device)], dim=-1)
            # if 'DEBUG_FP' in os.environ:
            #     tttt = attention_mask
            #     causal_mask = torch.cat([attention_mask, torch.full(
            #     (bsz, self.num_action_tokens + 1), 1, dtype=dtype,).to(attention_mask.device)], dim=-1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens + 1).repeat(
            bsz, 1
        )
        proprio_position_ids = torch.arange(1, self.num_proprio_tokens + 1).repeat(
            bsz, 1
        )
        action_position_ids = torch.arange(
            self.num_proprio_tokens + 1,
            self.num_proprio_tokens + self.num_action_tokens + 1,
        ).repeat(bsz, 1)
        # since proprio and action share the same mixture weights, makes sense to use [1 (proprio), 2 (action), 3 (action), ...] instead of [1 (proprio), 1 (action), 2 (action), ...]
        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        if len(causal_mask.shape) == 4:
            image_text_proprio_mask = causal_mask[
                ...,
                : self.max_image_text_tokens + self.num_proprio_tokens,
                : self.max_image_text_tokens + self.num_proprio_tokens,
            ]
            action_mask = causal_mask[..., -self.num_action_tokens :, :]
            return image_text_proprio_mask, action_mask
        else:
            # import ipdb;ipdb.set_trace()
            return causal_mask[:, : self.max_image_text_tokens + self.num_proprio_tokens,], causal_mask[:,  :]



    def build_causal_mask_and_position_ids_for_text(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        dtype, device = attention_mask.dtype, attention_mask.device

        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token, because we're in the prefill phase
            # assume no padding
            causal_mask = torch.full((bsz, q_len, q_len), 0, dtype=dtype, device=device)
        else:
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            # also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # this only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, kv_len), 0, dtype=dtype, device=device
            )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # use the last location
            position_ids = attention_mask.cumsum(-1)[:, -1:]
        else:
            # create position_ids based on the size of the attention_mask
            # for padded tokens, use number 1
            position_ids = (attention_mask.cumsum(-1)).masked_fill_(
                (attention_mask == 0), 1
            )
        return causal_mask, position_ids



    def build_causal_mask_and_position_ids_for_text1(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        dtype, device = attention_mask.dtype, attention_mask.device
        # import ipdb;ipdb.set_trace()
        bsz = attention_mask.shape[0]
        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token, because we're in the prefill phase
            # assume no padding
            causal_mask = torch.full((bsz, q_len, q_len), 0, dtype=dtype, device=device)
        else:
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            # also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # this only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, kv_len), 0, dtype=dtype, device=device
            )
        cache_position = torch.arange(
                0, 0 + q_len, device=causal_mask.device
            )
        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # use the last location
            position_ids = attention_mask.cumsum(-1)[:, -1:]
        else:
            # create position_ids based on the size of the attention_mask
            # for padded tokens, use number 1
            position_ids = (attention_mask.cumsum(-1)).masked_fill_(
                (attention_mask == 0), 1
            )
        min_dtype = torch.finfo(torch.float32).min
        
        causal_mask = torch.full((q_len, q_len), fill_value=min_dtype, dtype=torch.float32, device=device)
        
        diagonal_attend_mask = torch.arange(q_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.shape[-1] > q_len:
                attention_mask = attention_mask[:, :q_len]
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                causal_mask.device
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
        # import ipdb;ipdb.set_trace()
        return causal_mask, position_ids

    # ---------- Inference ----------#

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        out_img_feat = False,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        # import ipdb;ipdb.set_trace()
        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)
        # import ipdb;ipdb.set_trace()
        # image features from siglip and projector
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        selected_image_feature = self.vision_tower(pixel_values)
        # import ipdb;ipdb.set_trace()
        # b, n, dim = selected_image_feature.shape
        # selected_image_feature = selected_image_feature.reshape(b // self.num_images, self.num_images, n,dim).flatten(1,2)
        # pooler_output
        
        img_feats = selected_image_feature['last_hidden_state'][:, 1:]
        if self.image_448:
            img_feats = img_feats.reshape(img_feats.shape[0], 32, 32, img_feats.shape[-1])
        else:
            img_feats = img_feats.reshape(img_feats.shape[0], 16, 16, img_feats.shape[-1])
        selected_image_feature = self.pixel_shuffle(img_feats)

        image_features = self.multi_modal_projector(selected_image_feature).to(pixel_values.dtype)

        image_features = image_features.reshape(image_features.shape[0], image_features.shape[1] * image_features.shape[2], -1)
        # normalize the image features
        _, _, embed_dim = image_features.shape
        bsz, seq_len = input_ids.shape
        # import ipdb;ipdb.set_trace()
        scaled_image_features = image_features 
        # / (self.image_text_hidden_size**0.5)
        embed_dim = inputs_embeds.shape[-1]

        # put embedding together - image, text, padding
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), 0, dtype=dtype, device=device
        )
        # if 'DEBUG_JOINT' in os.environ:
        #     import ipdb;ipdb.set_trace()
        # [Batch_Size, Seq_Len]
        # import ipdb;ipdb.set_trace()
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        final_embedding[text_mask] = inputs_embeds[text_mask]
        if self.imgfeat:
            if self.no_img:
                pass
            else:
                final_embedding[image_mask] = scaled_image_features.flatten(0, 1).detach()
           
            
            selected_image_feature = self.vision_tower1(pixel_values)
            
            img_feats = selected_image_feature['last_hidden_state'][:, 1:]
            if self.image_448:
                img_feats = img_feats.reshape(img_feats.shape[0], 32, 32, img_feats.shape[-1])
            else:
                img_feats = img_feats.reshape(img_feats.shape[0], 16, 16, img_feats.shape[-1])
            selected_image_feature = self.pixel_shuffle(img_feats)            

            scaled_image_features = self.multi_modal_projector1(selected_image_feature).to(pixel_values.dtype)

            scaled_image_features = scaled_image_features.reshape(scaled_image_features.shape[0], scaled_image_features.shape[1] * scaled_image_features.shape[2], -1)            
            
            pass
        else:
            final_embedding[image_mask] = scaled_image_features.flatten(0, 1)
       
        if out_img_feat:
            return final_embedding, scaled_image_features
        else:
            return final_embedding

    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0) // self.num_images
        # if "BUG" in os.environ:
        #     import ipdb;ipdb.set_trace()
        kv_caches = self.joint_model.build_mixture_caches()

        if self.imgfeat:
            # merge the text tokens and the image tokens
            inputs_embeds, selected_image_feats = self._forward_siglip_and_text_embedding(input_ids, pixel_values, out_img_feat=True)
        else:
            # merge the text tokens and the image tokens
            inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)
        if self.imgfeat:
            # import ipdb;ipdb.set_trace()
            img_position_ids = torch.arange(1, 257).repeat(len(proprio_position_ids), 1).to(proprio_position_ids.device)
            proprio_position_ids = proprio_position_ids + 256 
            proprio_position_ids = torch.cat([img_position_ids, proprio_position_ids], dim=1)
            action_position_ids = action_position_ids + 256
            proprio_embeds = torch.cat([selected_image_feats, proprio_embeds], dim=1)

            act_propri_token_num = self.num_proprio_tokens + self.num_action_tokens
            # import ipdb;ipdb.set_trace()

        
            if not self.debug_causal:
                new_causal_mask = torch.triu(torch.full((len(proprio_embeds),  action_mask.shape[-1] + 256,  action_mask.shape[-1] + 256),  torch.finfo(action_mask.dtype).min, dtype=action_mask.dtype, device=action_mask.device), diagonal=1)[:, None,]

                new_causal_mask[:, :, - act_propri_token_num+1:, - act_propri_token_num:] = 0
                # image_text_proprio_mask = new_causal_mask[:,:, :641, :641]
                # image_text_proprio_mask, action_mask = self.split_full_mask_into_submasks(new_causal_mask)
                if self.NO_CAUSAL_IMG:
                    new_causal_mask[:, :, new_causal_mask.shape[-1] - act_propri_token_num - 256: new_causal_mask.shape[-1] - act_propri_token_num, 
                                                            new_causal_mask.shape[-1] - act_propri_token_num - 256 :new_causal_mask.shape[-1] - act_propri_token_num] = 0            
                
                image_text_proprio_mask = new_causal_mask[..., : -4, : -4,]
                action_mask = new_causal_mask[..., -self.num_action_tokens :, :]
            else:
                causal_mask = torch.ones([len(image_text_proprio_mask), action_mask.shape[-1] + 256], 
                                                dtype=action_mask.dtype, device=action_mask.device)
                image_text_proprio_mask, action_mask = causal_mask[:, :-4,], causal_mask[:,  -4:]
            # causal_mask = new_causal_mask
            # import ipdb;ipdb.set_trace()
        position_embeddings_vlm = self.internvl_model.language_model.model.rotary_emb(inputs_embeds, vlm_position_ids)
        position_embeddings_proprio = self.internvl_model.action_expert.model.rotary_emb(proprio_embeds, proprio_position_ids)
        # position_embeddings_action = self.internvl_model.action_expert.model.rotary_emb(action_embeds, action_position_ids)

        # forward pass thru the vlm and proprio, cache the kv
        _, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            position_embeddings_all = {
                'vlm': position_embeddings_vlm,
                'proprio': position_embeddings_proprio,
                # 'action': position_embeddings_action,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            position_embeddings_action = self.internvl_model.action_expert.model.rotary_emb(action_embeds, action_position_ids)
            action_embeds = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                position_embeddings_all = {
                    # 'vlm': position_embeddings_vlm,
                    # 'proprio': position_embeddings_proprio,
                    'action': position_embeddings_action,
                },
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            if self.integration_method == "euler":
                action_vel = self.action_decoder(action_embeds)
                action += delta_t * action_vel
            else:
                def model_step(x, tt):
                    # 这里可以重用解码逻辑
                    action_embeds_local = self.action_decoder(action_embeds)
                    return action_embeds_local

                # 根据选择的方法更新 action
                action = integration_step(
                    action, t, delta_t, model_step, method=self.integration_method
                )

            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        # cfg.horizon_steps
        action = action[:, -self.cfg.horizon_steps:, ]
        # if 'DEBUGX' in os.environ: import ipdb;ipdb.set_trace()
        return action

    def infer_action_naive(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # encode proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- run vlm in each step, which is unnecessary
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                    "action": action_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds.clone(),  # clone needed due to modified in-place
                    "proprio": proprio_embeds.clone(),
                    "action": action_embeds,
                },
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="no_append",  # no new tokens
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def infer_text(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        q_len = input_ids.size(1)

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # build causal mask and position ids for text
        (
            causal_mask,
            position_ids,
        ) = self.build_causal_mask_and_position_ids_for_text1(
            q_len, attention_mask, kv_cache
        )
        position_ids = position_ids-1
        position_embeddings_vlm = self.internvl_model.language_model.model.rotary_emb(inputs_embeds, position_ids)
        # position_embeddings_proprio = self.internvl_model.action_expert.model.rotary_emb(proprio_embeds, proprio_position_ids)
        # if 'DEBUG111' in os.environ: import ipdb;ipdb.set_trace()
        hidden_states = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": position_ids},
            embeds_all={"vlm": inputs_embeds},
            kv_caches={"vlm": kv_cache},
            position_embeddings_all = {
                'vlm': position_embeddings_vlm,
                # 'action': position_embeddings_action,
            },
            cache_mode="append",  # new tokens for the active mixture
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )["vlm"]
        logits = self.lm_head(hidden_states)
        output = {
            "logits": logits,
        }
        if kv_cache is not None:
            output["kv_cache"] = kv_cache
        return output

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        # import ipdb;ipdb.set_trace()
        if t.ndim == 4:
            t = t[:, None, None, None]  # (B, 1, 1,1)
        else:
             t = t[:,None, None]
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.ByteTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t: torch.FloatTensor,
        labels= None, #for vlm
        is_vlm= False,
        **kwargs,
    ) -> torch.FloatTensor:


        # import ipdb;ipdb.set_trace()
        if is_vlm:
            # import ipdb;ipdb.set_trace()
            res = self.internvl_model(pixel_values=pixel_values, input_ids=input_ids, labels=labels, **kwargs)
            # print(res)
            return res.loss
           
        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)
        # import ipdb;ipdb.set_trace()
        # text tokens + image tokens
        if self.imgfeat:
            # merge the text tokens and the image tokens
            inputs_embeds, selected_image_feats = self._forward_siglip_and_text_embedding(input_ids, pixel_values, out_img_feat=True)
            # import ipdb;ipdb.set_trace()
        else:
            # merge the text tokens and the image tokens
            inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)
        # proprio
        proprio_embeds = self.proprio_encoder(proprios)
        if self.imgfeat:

            # import ipdb;ipdb.set_trace()
            img_position_ids = torch.arange(1, 257).repeat(len(proprio_position_ids), 1).to(proprio_position_ids.device)
            proprio_position_ids = proprio_position_ids + 256 
            proprio_position_ids = torch.cat([img_position_ids, proprio_position_ids], dim=1)
            action_position_ids = action_position_ids + 256
            proprio_embeds = torch.cat([selected_image_feats, proprio_embeds], dim=1)

            act_propri_token_num = self.num_proprio_tokens + self.num_action_tokens
        
            if not self.debug_causal:
                new_causal_mask = torch.triu(torch.full((len(proprio_embeds), 
                                            causal_mask.shape[-1] + 256, 
                                            causal_mask.shape[-1] + 256), 
                                            torch.finfo(causal_mask.dtype).min, dtype=causal_mask.dtype, device=causal_mask.device),
                                            diagonal=1)[:, None,]
                # new_causal_mask = new_causal_mask[:, None, :, :]
                # new_causal_mask[:, :, :causal_mask.shape[-1] - act_propri_token_num, :causal_mask.shape[-1] - act_propri_token_num] = causal_mask[:, :, :-act_propri_token_num, :-act_propri_token_num]
                # new_causal_mask[:, :, causal_mask.shape[-1] - act_propri_token_num: - act_propri_token_num, :causal_mask.shape[-1] - act_propri_token_num] = 0
                
                # new_causal_mask[:, :, - act_propri_token_num:-act_propri_token_num + 1, - act_propri_token_num: - act_propri_token_num + 1] = 0
                # new_causal_mask[:, :, - act_propri_token_num+1:, - act_propri_token_num:] = 0
                # causal_mask = new_causal_mask
                # import ipdb;ipdb.set_trace()

                # new_causal_mask = new_causal_mask[:, None, :, :]
                
                if self.NO_CAUSAL_IMG:
                    new_causal_mask[:, :, new_causal_mask.shape[-1] - act_propri_token_num - 256: new_causal_mask.shape[-1] - act_propri_token_num, 
                                                            new_causal_mask.shape[-1] - act_propri_token_num - 256 :new_causal_mask.shape[-1] - act_propri_token_num] = 0
                
                # new_causal_mask[:, :, - act_propri_token_num:-act_propri_token_num + 1, - act_propri_token_num: - act_propri_token_num + 1] = 0
                new_causal_mask[:, :, - act_propri_token_num+1:, - act_propri_token_num:] = 0
                causal_mask = new_causal_mask
            else:
                causal_mask = torch.ones([len(causal_mask), causal_mask.shape[-1] + 256], 
                                                dtype=causal_mask.dtype, device=causal_mask.device)
                     

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
                # create position embeddings to be shared across the decoder layers
        # import ipdb;ipdb.set_trace()
        
        # from IPython import embed
        # embed()
        position_embeddings_vlm = self.internvl_model.language_model.model.rotary_emb(inputs_embeds, vlm_position_ids)
        position_embeddings_proprio = self.internvl_model.action_expert.model.rotary_emb(proprio_embeds, proprio_position_ids)
        position_embeddings_action = self.internvl_model.action_expert.model.rotary_emb(action_embeds, action_position_ids)

        joint_outputs = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
                "action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            position_embeddings_all = {
                'vlm': position_embeddings_vlm,
                'proprio': position_embeddings_proprio,
                'action': position_embeddings_action,
            },
            final_layer_post_attn_skip_names=['proprio', ],
            time_cond=time_cond,
            kv_caches={},  # no caching during training
        )
        action_embeds = joint_outputs['action']


            # if ignore_flag:
            #     loss = loss * 0.0
            
        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        loss_action= torch.mean((v_psi - d_psi) ** 2)
        loss_all = loss_action
        
        return loss_all


    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.ByteTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t: torch.FloatTensor,
        labels= None, #for vlm
        **kwargs,
    ) -> torch.FloatTensor:
        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        # x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        # x1 = actions
        # psi_t = self.psi_t(x0, x1, t)

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # proprio
        

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        # time_cond = self.time_embedding(t)
        time_cond = None
        # [Batch_Size, Horizon_Steps, Embed_Dim]
       
        position_embeddings_vlm = self.internvl_model.language_model.model.rotary_emb(inputs_embeds, vlm_position_ids)
       

        joint_outputs = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                # "proprio": proprio_position_ids,
                # "action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                # "proprio": proprio_embeds,
                # "action": action_embeds,
            },
            position_embeddings_all = {
                'vlm': position_embeddings_vlm,
                # 'proprio': position_embeddings_proprio,
                # 'action': position_embeddings_action,
            },
            final_layer_post_attn_skip_names=['proprio', 'action'],
            time_cond=time_cond,
            kv_caches={},  # no caching during training
        )
        # action_embeds = joint_outputs['action']


            # if ignore_flag:
            #     loss = loss * 0.0
            
        # [Batch_Size, Horizon_Steps, Action_Dim]
        
        vlm_hidden_states = joint_outputs['vlm']
        logits = self.lm_head(vlm_hidden_states)
        # labels = input_ids
        # logits = logits[:]
        # logits = logits[..., :, :]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.internvl_model.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        
        loss_vlm = loss_fct(shift_logits[shift_labels > 0], shift_labels[shift_labels > 0])
        # print(loss_vlm, flush=True)
       
        # import ipdb;ipdb.set_trace()
        return loss_vlm


class PiZeroInference(PiZero):
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return super().infer_action(
            input_ids,
            pixel_values,
            image_text_proprio_mask,
            action_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
            proprios,
        )

def integration_step(action, t, delta_t, model_step, method="euler"):
    """
    单步数值积分器
    Args:
        action: 当前状态 (tensor)
        t: 当前时间 (tensor)
        delta_t: 时间步长 (float)
        model_step: 一个函数 f(action, t) → action_vel
        method: "euler", "heun", "rk4"
    """
    if method == "euler":
        # 一阶显式欧拉
        action_vel = model_step(action, t)
        action_next = action + delta_t * action_vel

    elif method == "heun":
        # 二阶 Heun 方法（改进欧拉）
        k1 = model_step(action, t)
        k2 = model_step(action + delta_t * k1, t + delta_t)
        action_next = action + 0.5 * delta_t * (k1 + k2)

    elif method == "rk4":
        # 四阶 Runge-Kutta 方法
        k1 = model_step(action, t)
        k2 = model_step(action + 0.5 * delta_t * k1, t + 0.5 * delta_t)
        k3 = model_step(action + 0.5 * delta_t * k2, t + 0.5 * delta_t)
        k4 = model_step(action + delta_t * k3, t + delta_t)
        action_next = action + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    else:
        raise ValueError(f"Unknown integration method: {method}")

    return action_next

if __name__ == "__main__":
    import argparse
    import time

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import AutoTokenizer

    from src.model.vla.processing import VLAProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--load_pretrained_weights", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--loss_only", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert not (args.text_only and args.loss_only)

    torch.manual_seed(args.seed)

    # config = OmegaConf.load("config/train/bridge_internvl_3_5_448.yaml")
    config = OmegaConf.load("config/train/bridge_debug.yaml")
    # 
    
    if args.text_only:
        config.use_lm_head = True
        config.mixture.vlm.use_final_norm = True
    device = "cpu" if args.cpu else "cuda"
    model = PiZero(config)
    model.tie_action_proprio_weights()
    # if args.load_pretrained_weights:
    #     model.load_pretrained_weights()
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    model.to(device)
    model.to(dtype)
    model.eval()
    print(f"Using {device} and {dtype}...")
    from IPython import embed
    embed()
    # dummy image --- replace the first image with a real one
    bsz = 1 if args.text_only else 2
    dummy_images = torch.randint(
        0, 256, (bsz, 3, 448, 448), dtype=torch.uint8
    )  # not used if text_only
    real_image_path = "media/maniskill_pp.png"
    real_image = Image.open(real_image_path).convert("RGB")
    real_image_t = torch.as_tensor(
        np.array(real_image.resize((448, 448))).transpose(2, 0, 1)
    )
    dummy_images[0] = real_image_t
    dummy_images = dummy_images.unsqueeze(0)
    # text and proprio
    dummy_texts = [
        "this image shows , please say something",
        "this is a nice portrait of London because ",
    ][:bsz]
    dummy_proprio = torch.rand(bsz, config.cond_steps, config.action_dim)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, padding_side="right"
    )
    assert tokenizer.padding_side == "right"

    # processor
    num_image_tokens = config.vision.config.num_image_tokens
    processor = InternVLAProcessor(tokenizer, num_image_tokens, config.max_seq_len)

    # process image and text
    model_inputs = processor(text=dummy_texts, images=dummy_images)
    input_ids = model_inputs["input_ids"][:, :]
    attention_mask = model_inputs["attention_mask"][:, :]
    pixel_values = model_inputs["pixel_values"].to(dtype)

    # inference
    start_time = time.time()
    if args.text_only:  # no sampling
        kv_cache = model.build_text_cache()
        num_tokens_to_generate = 100
        print(f"Generating text of maximum {num_tokens_to_generate} tokens...")

        stop_token = processor.tokenizer.eos_token_id
        generated_tokens = []
        
        # set the max number of tiles in `max_num`
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        question = '<image>\n' + dummy_texts[0]
        # import ipdb;ipdb.set_trace()
        response = model.internvl_model.chat(tokenizer, pixel_values.cuda(), question, generation_config)
        print(response)
        # import ipdb;ipdb.set_trace()
        for _ in range(num_tokens_to_generate):
            with torch.inference_mode():
                outputs = model.infer_text(
                    input_ids=input_ids.to(device),
                    pixel_values=pixel_values.to(device),
                    attention_mask=attention_mask.to(device),
                    kv_cache=kv_cache,
                )
            next_token_logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)  # remove batch dimension
            generated_tokens.append(next_token)
            # stop if the stop token has been generated
            # import ipdb;ipdb.set_trace()
            if next_token.item() == stop_token:
                break
            # only input the new token the next time since using cache
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=-1
            )
        generated_tokens = torch.cat(generated_tokens, dim=-1)
        decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("\n\n=========================")
        print("Image path:", real_image_path)
        print("Prompt:", dummy_texts[0])
        print(decoded)
    elif args.loss_only:
        dummy_actions = torch.randn(bsz, config.horizon_steps, config.action_dim)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        attention_mask = torch.cat([attention_mask, torch.ones([2, 5], dtype=torch.int64)], dim=-1)
        t = torch.rand(bsz)

        pixel_values = pixel_values.to(torch.bfloat16)
        dummy_proprio = dummy_proprio.to(torch.bfloat16)
        dummy_actions = dummy_actions.to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(
                input_ids=input_ids.to(device),
                pixel_values=pixel_values.to(dtype).to(device),
                causal_mask=attention_mask.to(device),
                vlm_position_ids=vlm_position_ids.to(device),
                proprio_position_ids=proprio_position_ids.to(device),
                action_position_ids=action_position_ids.to(device),
                proprios=dummy_proprio.to(dtype).to(device),
                actions=dummy_actions.to(dtype).to(device),
                t=t.to(dtype).to(device),
            )
        print("\n\n=========================")
        print("Loss:", loss)
    else:  # dummy action generation
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        
        with torch.inference_mode():
            actions = model.infer_action(
                input_ids=input_ids.to(device),
                pixel_values=pixel_values.to(dtype).to(device),
                image_text_proprio_mask=image_text_proprio_mask.to(device),
                action_mask=action_mask.to(device),
                vlm_position_ids=vlm_position_ids.to(device),
                proprio_position_ids=proprio_position_ids.to(device),
                action_position_ids=action_position_ids.to(device),
                proprios=dummy_proprio.to(dtype).to(device),
            )
        print("\n\n=========================")
        print("Final action dimensions:", actions.shape)
        print("Final action values:", actions)
    print("Time taken:", time.time() - start_time)
    print("============================\n\n")
