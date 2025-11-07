"""
Wrapper around the mixtures (without encoder / decoder)

Agnostic to the mixture setup

KV caches --- There are a few different modes depending on the setting:
    - text generation, only vlm active, use vlm cache --- append active (mode="append")
    - action naive inference, all active, use vlm and proprio cache --- no new tokens for the active mixture (mode="no_append")
    - action inference, no cache during vlm and proprio forward, then use vlm and proprio cache --- append, non-active (mode="append_non_active")
    - action flow matching training, all active, no cache (mode does not matter)
"""

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


from src.model.utils import repeat_kv
from src.model.kv_cache import KVCache
from src.model.vla.mixture import Mixture


def forward_mixture_layers(
    mixtures: nn.ModuleDict,
    attention_mask: torch.Tensor,
    position_ids_all: dict[torch.LongTensor],
    embeds_all: dict[torch.FloatTensor],
    layer_idx: int,
    post_attn_skip_names: Tuple[str, ...] = ("vlm", "proprio"),
    kv_caches: dict[KVCache] = {},
    cache_mode: str = "append_non_active",
    position_embeddings_all = {},
    time_cond: Optional[torch.FloatTensor] = None,
    backbone_type = 'default',
    stop_grad = False,
    is_training = True,
) -> dict[torch.FloatTensor]:
    """the usual norm + attn + res + norm + mlp + res"""
    active_mixture_names = list(embeds_all.keys())

    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_attn = embeds_all
    hidden_states_input_norm = {}
    

    for name in active_mixture_names:
        hidden_states_input_norm[name] = mixtures[name].layer_func(
            "forward_norm",
            layer_idx,
            "input_layernorm",
            embeds_all[name],
            time_cond,
        )  # a bit convoluted
    hidden_states_pre_attn = hidden_states_input_norm

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_attn = forward_mixture_attn(
        mixtures,
        hidden_states_all=hidden_states_pre_attn,
        attention_mask=attention_mask,
        position_ids_all=position_ids_all,
        layer_idx=layer_idx,
        post_attn_skip_names=post_attn_skip_names,
        kv_caches=kv_caches,
        cache_mode=cache_mode,
    )
    hidden_states_pre_res = hidden_states_post_attn

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_res = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_post_res[name] = None
        else:
            hidden_states_pre_res[name] = mixtures[name].layer_func(
                "forward_adaptive_scale",
                layer_idx,
                "post_attn",
                hidden_states_pre_res[name],
                time_cond,
            )
            hidden_states_post_res[name] = (
                residuals_pre_attn[name] + hidden_states_pre_res[name]
            )
    hidden_states_pre_post_attn = hidden_states_post_res

    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_post_attn = hidden_states_pre_post_attn
    hidden_states_post_post_attn = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_post_post_attn[name] = None
        else:
            hidden_states_post_post_attn[name] = mixtures[name].layer_func(
                "forward_norm",
                layer_idx,
                "post_attention_layernorm",
                hidden_states_pre_post_attn[name],
                time_cond,
            )
    hidden_states_pre_mlp = hidden_states_post_post_attn

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_pos_mlp = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_pos_mlp[name] = None
        else:
            hidden_states_pos_mlp[name] = mixtures[name].layer_func(
                "mlp",
                layer_idx,
                hidden_states_pre_mlp[name],
            )
    hidden_states_pre_final_res = hidden_states_pos_mlp

    
    hidden_states_final = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_final[name] = None
        else:
            hidden_states_pre_final_res[name] = mixtures[name].layer_func(
                "forward_adaptive_scale",
                layer_idx,
                "final",
                hidden_states_pre_final_res[name],
                time_cond,
            )
            hidden_states_final[name] = (
                residuals_pre_post_attn[name] + hidden_states_pre_final_res[name]
            )
    return hidden_states_final


def forward_mixture_layers_internvl(
    mixtures: nn.ModuleDict,
    attention_mask: torch.Tensor,
    position_ids_all: dict[torch.LongTensor],
    embeds_all: dict[torch.FloatTensor],
    layer_idx: int,
    post_attn_skip_names: Tuple[str, ...] = ("vlm", "proprio"),
    kv_caches: dict[KVCache] = {},
    position_embeddings_all = {},
    cache_mode: str = "append_non_active",
    time_cond: Optional[torch.FloatTensor] = None,
    backbone_type = 'INTERNVL',
    use_flash_attention=False,
    stop_grad=False,
    is_training=True,
) -> dict[torch.FloatTensor]:
    """the usual norm + attn + res + norm + mlp + res"""
    active_mixture_names = list(embeds_all.keys())

    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_attn = embeds_all
    hidden_states_input_norm = {}
    
    for name in active_mixture_names:
        
        
        hidden_states_input_norm[name] = mixtures[name].layers[layer_idx].input_layernorm(embeds_all[name])
        
    hidden_states_pre_attn = hidden_states_input_norm

    
    forward_mixture_attn_func = forward_mixture_attn_internvl
    
    hidden_states_post_attn = forward_mixture_attn_func(
        mixtures,
        hidden_states_all=hidden_states_pre_attn,
        attention_mask=attention_mask,
        position_ids_all=position_ids_all,
        layer_idx=layer_idx,
        post_attn_skip_names=post_attn_skip_names,
        position_embeddings_all=position_embeddings_all,
        kv_caches=kv_caches,
        cache_mode=cache_mode,
        use_flash_attention=use_flash_attention,
        stop_grad=stop_grad,
        is_training=is_training,
    )
    hidden_states_pre_res = hidden_states_post_attn

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_res = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_post_res[name] = None
        else:
            
            hidden_states_post_res[name] = (
                residuals_pre_attn[name] + hidden_states_pre_res[name]
            )
    hidden_states_pre_post_attn = hidden_states_post_res

    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_post_attn = hidden_states_pre_post_attn
    hidden_states_post_post_attn = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_post_post_attn[name] = None
        else:
           
            hidden_states_post_post_attn[name] = mixtures[name].layers[layer_idx].post_attention_layernorm(hidden_states_pre_post_attn[name])
    hidden_states_pre_mlp = hidden_states_post_post_attn

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_pos_mlp = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_pos_mlp[name] = None
        else:
            
            hidden_states_pos_mlp[name] = mixtures[name].layers[layer_idx].mlp(hidden_states_pre_mlp[name])
    hidden_states_pre_final_res = hidden_states_pos_mlp

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_final = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_final[name] = None
        else:
            
            hidden_states_final[name] = (
                residuals_pre_post_attn[name] + hidden_states_pre_final_res[name]
            )
    return hidden_states_final


def forward_mixture_attn(
    mixtures: nn.ModuleDict,
    attention_mask: torch.Tensor,
    position_ids_all: dict[torch.LongTensor],
    hidden_states_all: dict[torch.FloatTensor],
    layer_idx: int,
    post_attn_skip_names: Tuple[str, ...] = ("vlm", "proprio"),
    kv_caches: dict[KVCache] = {},
    cache_mode: str = "append_non_active",

    attn_softclamp: float = 50.0,  # default in gemma
    attention_dropout: float = 0.0,
) -> dict[torch.FloatTensor]:
    """Assume all mixtures have the same head dim"""
    assert cache_mode in [
        "no_append",
        "append",
        "append_non_active",
    ], f"Invalid cache mode: {cache_mode}"
    bsz = len(attention_mask)
    q_lens = [hidden_states.size(1) for hidden_states in hidden_states_all.values()]
    active_mixture_names = list(hidden_states_all.keys())

    # always re-compute queries
    query_states_all = {}
    for name in active_mixture_names:
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = mixtures[name].attn_func("forward_q_proj", layer_idx, hidden_states_all[name])
        query_states_all[name] = query_states

    # use kv caches from non-active mixtures
    key_states_all = {}
    value_states_all = {}
    if cache_mode == "append_non_active":
        for name, kv_cache in kv_caches.items():
            if name not in active_mixture_names:
                key_states_all[name], value_states_all[name] = kv_cache.get(layer_idx)

    # the caching logic below can be much simplified if we ignore the "no_append" mode, which is only used in the naive action inference mode
    for name in active_mixture_names:
        # prepare rope
        query_states = query_states_all[name]
        rope_cos, rope_sin = mixtures[name].attn_func(
            "forward_rotary_emb", layer_idx, query_states, position_ids_all[name]
        )


        # always use kv cache if it has the current layer
        flag_cached_mixture = name in kv_caches and kv_caches[name].has_item(layer_idx)
        if flag_cached_mixture:
            key_states_cached, value_states_cached = kv_caches[name].get(
                layer_idx
            )  # note: rope already applied before they were cached

        # always add to cache in append mode, or kv cache does not have the layer yet (in no_append mode)
        flag_to_cache_mixture = (
            name in kv_caches and not kv_caches[name].has_item(layer_idx)
        ) or cache_mode == "append"

        # calculate kv for new tokens if in append mode or this layer is not cached
        key_states_new, value_states_new = None, None
        flag_calc_new_kv = not flag_cached_mixture or cache_mode == "append"
        assert flag_cached_mixture or flag_calc_new_kv, (
            "Cannot skip new kv calculation while also not using cache!"
        )
        if flag_calc_new_kv:
            hidden_states = hidden_states_all[name]
            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            key_states_new = mixtures[name].attn_func("forward_k_proj", layer_idx, hidden_states)
            value_states_new = mixtures[name].attn_func(
                "forward_v_proj", layer_idx, hidden_states
            )
            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            key_states_new = mixtures[name].attn_func(
                "forward_apply_rotary_emb",
                layer_idx,
                key_states_new,
                rope_cos,
                rope_sin,
            )
            if flag_to_cache_mixture:
                kv_caches[name].update(
                    key_states_new,
                    value_states_new,
                    layer_idx,
                )

        # always apply rope to Q
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = mixtures[name].attn_func(
            "forward_apply_rotary_emb", layer_idx, query_states, rope_cos, rope_sin
        )
        query_states_all[name] = query_states

        # assign K and V carefully for this active mixture
        if flag_cached_mixture:
            key_states = key_states_cached
            value_states = value_states_cached
            if key_states_new is not None:
                key_states = torch.cat((key_states, key_states_new), dim=-2)
            if value_states_new is not None:
                value_states = torch.cat((value_states, value_states_new), dim=-2)
        else:
            key_states = key_states_new
            value_states = value_states_new
        key_states_all[name] = key_states
        value_states_all[name] = value_states


    # Repeat the key and values to match the number of heads of the query
    for name in key_states_all:
        key_states, value_states = mixtures[name].attn_func(
            "repeat_kv",
            layer_idx,
            key_states_all[name],
            value_states_all[name],
        )
        key_states_all[name] = key_states
        value_states_all[name] = value_states
        
    # Concatenate all the blocks along sequence
    # [Batch_Size, Num_Heads_Q / Num_Heads_KV, Full_Seq_Len, Head_Dim]
    query_states = torch.cat(tuple(query_states_all.values()), dim=-2)
    key_states = torch.cat(tuple(key_states_all.values()), dim=-2)
    value_states = torch.cat(tuple(value_states_all.values()), dim=2)

    # Perform the calculation as usual, Q * K^T / sqrt(head_dim)
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        mixtures[active_mixture_names[0]].head_dim
    )

    # Soft capping
    attn_weights = attn_weights / attn_softclamp
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * attn_softclamp

    # Apply the softmax / dropout
    attn_weights = attn_weights + attention_mask
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights,
        p=attention_dropout,
        training=mixtures[active_mixture_names[0]].training,
    )
    # Multiply by the values. [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len] x [Batch_Size, Num_Heads_KV, Full_Seq_Len, Head_Dim] -> [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim]
    attn_output = torch.matmul(attn_weights, value_states)

    # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
    # Concatenate all the heads together. [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q * Head_Dim]
    attn_output = attn_output.view(bsz, sum(q_lens), -1)

    # Split into the different mixtures
    attn_outputs = torch.split(attn_output, q_lens, dim=1)
    attn_outputs = {
        key: value for key, value in zip(active_mixture_names, attn_outputs)
    }

    # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
    attn_outputs_final = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            attn_outputs_final[name] = None
        else:
            attn_outputs_final[name] = mixtures[name].attn_func(
                "forward_o_proj", layer_idx, attn_outputs[name]
            )
    return attn_outputs_final


def forward_mixture_attn_internvl(
    mixtures: nn.ModuleDict,
    attention_mask: torch.Tensor,
    position_ids_all: dict[torch.LongTensor],
    hidden_states_all: dict[torch.FloatTensor],
    layer_idx: int,
    post_attn_skip_names: Tuple[str, ...] = ("vlm", "proprio"),
    kv_caches: dict[KVCache] = {},
    cache_mode: str = "append_non_active",
    position_embeddings_all={},
    attn_softclamp: float = 50.0,  # default in gemma
    attention_dropout: float = 0.0,
    use_flash_attention=False,
    stop_grad = False,
    is_training = True,
) -> dict[torch.FloatTensor]:
    """Assume all mixtures have the same head dim"""
    assert cache_mode in [
        "no_append",
        "append",
        "append_non_active",
    ], f"Invalid cache mode: {cache_mode}"
    
    bsz = len(attention_mask)
    q_lens = [hidden_states.size(1) for hidden_states in hidden_states_all.values()]
    active_mixture_names = list(hidden_states_all.keys())

    attn_head_dim = 128
    # always re-compute queries
    query_states_all = {}
    for name in active_mixture_names:
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        # import ipdb;ipdb.set_trace()
        # from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        # query_states = mixtures[name].attn_func(  
        #     "forward_q_proj", layer_idx, hidden_states_all[name]
        # )
        # import ipdb;ipdb.set_trace()

        if not 'Qwen3' in str(type(mixtures[name].layers[layer_idx].self_attn)):
            query_states = mixtures[name].layers[layer_idx].self_attn.q_proj(hidden_states_all[name]).view((hidden_states_all[name].shape[0], hidden_states_all[name].shape[1],  -1,  attn_head_dim)).transpose(1, 2)
        else:
            query_states = mixtures[name].layers[layer_idx].self_attn.q_norm(mixtures[name].layers[layer_idx].self_attn.q_proj(hidden_states_all[name]).view((hidden_states_all[name].shape[0], hidden_states_all[name].shape[1],  -1,  attn_head_dim))).transpose(1, 2)

        
            
        query_states_all[name] = query_states

    # use kv caches from non-active mixtures
    key_states_all = {}
    value_states_all = {}
    if cache_mode == "append_non_active":
        for name, kv_cache in kv_caches.items():
            if name not in active_mixture_names:
                key_states_all[name], value_states_all[name] = kv_cache.get(layer_idx)
    
    # the caching logic below can be much simplified if we ignore the "no_append" mode, which is only used in the naive action inference mode
    for name in active_mixture_names:
        # prepare rope
        query_states = query_states_all[name]
       
        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(value, cos, sin, position_ids=None, unsqueeze_dim=1):
            """Applies Rotary Position Embedding to the query and key tensors.

            Args:
                q (`torch.Tensor`): The query tensor.
                k (`torch.Tensor`): The key tensor.
                cos (`torch.Tensor`): The cosine part of the rotary embedding.
                sin (`torch.Tensor`): The sine part of the rotary embedding.
                position_ids (`torch.Tensor`, *optional*):
                    Deprecated and unused.
                unsqueeze_dim (`int`, *optional*, defaults to 1):
                    The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                    sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                    that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                    k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                    cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                    the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
            Returns:
                `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
            """
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            # if "BUG" in os.environ:
            #     import ipdb;ipdb.set_trace()
            q_embed = (value * cos) + (rotate_half(value) * sin)
            # k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed
        
        def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
            """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

            Explanation:
                Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
                sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
                vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
                Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
                For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
                height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
                difference with modern LLMs.

            Args:
                q (`torch.Tensor`): The query tensor.
                k (`torch.Tensor`): The key tensor.
                cos (`torch.Tensor`): The cosine part of the rotary embedding.
                sin (`torch.Tensor`): The sine part of the rotary embedding.
                position_ids (`torch.Tensor`):
                    The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                    used to pass offsetted position ids when working with a KV-cache.
                mrope_section(`List(int)`):
                    Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
                unsqueeze_dim (`int`, *optional*, defaults to 1):
                    The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                    sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                    that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                    k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                    cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                    the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
            Returns:
                `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
            """
            mrope_section = mrope_section * 2
            cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
                unsqueeze_dim
            )
            sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
                unsqueeze_dim
            )

            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed        

        cos, sin = position_embeddings_all[name]
        # apply_rotary_pos_emb(query_states, None, )

        # always use kv cache if it has the current layer
        flag_cached_mixture = name in kv_caches and kv_caches[name].has_item(layer_idx)
        if flag_cached_mixture:
            key_states_cached, value_states_cached = kv_caches[name].get(
                layer_idx
            )  # note: rope already applied before they were cached

        # always add to cache in append mode, or kv cache does not have the layer yet (in no_append mode)
        flag_to_cache_mixture = (
            name in kv_caches and not kv_caches[name].has_item(layer_idx)
        ) or cache_mode == "append"

        # calculate kv for new tokens if in append mode or this layer is not cached
        key_states_new, value_states_new = None, None
        flag_calc_new_kv = not flag_cached_mixture or cache_mode == "append"
        assert flag_cached_mixture or flag_calc_new_kv, (
            "Cannot skip new kv calculation while also not using cache!"
        )
        if flag_calc_new_kv:
            hidden_states = hidden_states_all[name]
            
            if not 'Qwen3' in str(type(mixtures[name].layers[layer_idx].self_attn)):
                key_states_new = mixtures[name].layers[layer_idx].self_attn.k_proj(hidden_states_all[name]).view((hidden_states_all[name].shape[0],hidden_states_all[name].shape[1], -1, 128)).transpose(1, 2)
            else:
                key_states_new = mixtures[name].layers[layer_idx].self_attn.k_norm(mixtures[name].layers[layer_idx].self_attn.k_proj(hidden_states_all[name]).view((hidden_states_all[name].shape[0],hidden_states_all[name].shape[1], -1, 128))).transpose(1, 2)
            # key_states_new = mixtures[name].layers[layer_idx].self_attn.k_proj(hidden_states_all[name]).view(hidden_states_all[name].shape).transpose(1, 2)
            value_states_new = mixtures[name].layers[layer_idx].self_attn.v_proj(hidden_states_all[name]).view((hidden_states_all[name].shape[0],hidden_states_all[name].shape[1], -1, 128)).transpose(1, 2)
            
            
            try:
               
                key_states_new = apply_rotary_pos_emb(key_states_new, cos, sin)
            except:
                
                import ipdb;ipdb.set_trace()

            if flag_to_cache_mixture:
                kv_caches[name].update(
                    key_states_new,
                    value_states_new,
                    layer_idx,
                )

        # always apply rope to Q
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
       
        query_states_all[name] = query_states

        # assign K and V carefully for this active mixture
        if flag_cached_mixture:
            key_states = key_states_cached
            value_states = value_states_cached
            if key_states_new is not None:
                key_states = torch.cat((key_states, key_states_new), dim=-2)
            if value_states_new is not None:
                value_states = torch.cat((value_states, value_states_new), dim=-2)
        else:
            key_states = key_states_new
            value_states = value_states_new
        key_states_all[name] = key_states
        value_states_all[name] = value_states

    num_key_value_groups = 1

       
   
            
    # Concatenate all the blocks along sequence
    # [Batch_Size, Num_Heads_Q / Num_Heads_KV, Full_Seq_Len, Head_Dim]
    query_states = torch.cat(tuple(query_states_all.values()), dim=-2)
    key_states = torch.cat(tuple(key_states_all.values()), dim=-2)
    value_states = torch.cat(tuple(value_states_all.values()), dim=2)

    attn_module = mixtures[active_mixture_names[0]].layers[layer_idx].self_attn
    

    impl_attention_forward = None
    if not 'qwen3' in str(type(attn_module)):
        from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
        from transformers.integrations.flash_attention import flash_attention_forward
        if use_flash_attention:
            #  or 'DEBUG_JOINT' in os.environ and len(active_mixture_names) == 1
            impl_attention_forward = flash_attention_forward
            attn_output, attn_weights = impl_attention_forward(attn_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
                    scaling=attn_module.scaling,
                    sliding_window=attn_module.config.sliding_window,  # main diff with Llama
                )
        else:
            impl_attention_forward = eager_attention_forward
            
            attn_output, attn_weights = impl_attention_forward(attn_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
                    scaling=attn_module.scaling,
                    sliding_window=attn_module.config.sliding_window,  # main diff with Llama
                )
    else:
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        impl_attention_forward = eager_attention_forward
        
        attn_output, attn_weights = impl_attention_forward(attn_module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
                scaling=attn_module.scaling,
                sliding_window=attn_module.config.sliding_window,  # main diff with Llama
            )
        
    
    attn_output = attn_output.view(bsz, sum(q_lens), -1)

    
    attn_outputs = torch.split(attn_output, q_lens, dim=1)
    attn_outputs = {
        key: value for key, value in zip(active_mixture_names, attn_outputs)
    }

   
    attn_outputs_final = {}
    for name in active_mixture_names:
        if "DEBUG_init" in os.environ:
            # from IPython import embed
            # embed()
            if name != "vlm":
                x_downsampled = F.interpolate(attn_outputs[name], size=896, mode='linear', align_corners=True)
                attn_outputs[name] = x_downsampled
            
        if name in post_attn_skip_names:
            attn_outputs_final[name] = None
        else:
           
            attn_outputs_final[name] = mixtures[name].layers[layer_idx].self_attn.o_proj(attn_outputs[name])
            
    return attn_outputs_final





class JointModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # import ipdb;ipdb.set_trace()
        self.num_hidden_layers = config.num_hidden_layers
        self.num_mixture = len(config.mixture)
        self.cache_names = [
            name for name in config.mixture if config.mixture[name].cache
        ]  # name of the mixtures that use cache during generation; no cache during training

        # Mixtures
        self.mixtures = nn.ModuleDict()
        for mixture_name, mixture_config in config.mixture.items():
            mixture_config = OmegaConf.merge(config, mixture_config)
            self.mixtures[mixture_name] = Mixture(mixture_config)
        self.mixture_names = list(config.mixture.keys())
        self.gradient_checkpointing = False
        
        self.backbone_type = 'default'
        
        if 'INTERNVL' in os.environ or 'QWENVL' in os.environ:
            self.use_internvl=True
            self.backbone_type = 'INTERNVL'
        
            
        self.use_flash_attention = False
        if 'DEBUG_CAUSAL' in os.environ:
            self.use_flash_attention = True
        
        self.stop_grad_to_vlm = False
        if 'STOP_GRAD' in os.environ:
            self.stop_grad_to_vlm = True
            

    def build_mixture_caches(self):
        return {name: KVCache() for name in self.cache_names}

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids_all: dict[torch.LongTensor],
        embeds_all: dict[torch.FloatTensor],
        time_cond: Optional[torch.FloatTensor] = None,
        final_layer_post_attn_skip_names: Tuple[str, ...] = ("vlm", "proprio"),
        kv_caches: dict[KVCache] = {},
        position_embeddings_all= {},
        cache_mode: str = "append_non_active",
        return_caches: bool = False,
    ) -> dict[torch.FloatTensor]:
        """
        Assume attention_mask is in the right block attention form

        embeds_all and position_ids_all need to be in the correct order, e.g., {"vlm": ..., "proprio": ..., "action": ...}
        """
        active_mixture_names = list(embeds_all.keys())

        # normalization
        # [Batch_Size, Seq_Len, Hidden_Size]
        

        if self.backbone_type == 'default':
            for name in active_mixture_names:
                hidden_size = embeds_all[name].shape[-1]
                normalizer = torch.tensor(
                    hidden_size**0.5,
                    dtype=embeds_all[name].dtype,
                    device=embeds_all[name].device,
                )
                embeds_all[name] *= normalizer
                if name == 'vlm':
                    if self.stop_grad_to_vlm:
                        embeds_all[name] = embeds_all[name].detach()
       
        for layer_idx in range(self.num_hidden_layers):
            is_final_layer = layer_idx == self.num_hidden_layers - 1
            func_forward_mixture = forward_mixture_layers_internvl if self.backbone_type != 'default' else forward_mixture_layers
             
            embeds_all = func_forward_mixture(
                self.mixtures,
                attention_mask,
                position_ids_all,
                embeds_all,
                layer_idx=layer_idx,
                position_embeddings_all = position_embeddings_all,
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode=cache_mode,
                post_attn_skip_names=final_layer_post_attn_skip_names
                if is_final_layer
                else [],
                backbone_type = self.backbone_type,
                use_flash_attention=self.use_flash_attention,
                stop_grad = self.stop_grad_to_vlm,
                is_training = self.training,
            )
            
            
           

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_all = {}
        for name in active_mixture_names:
            if name not in final_layer_post_attn_skip_names:
                hidden_states_all[name] = self.mixtures[name].forward_norm(
                    embeds_all[name], time_cond
                )
                # if name == 'vlm':
                #     if self.stop_grad_to_vlm:
                #         embeds_all['vlm'] = embeds_all['vlm'].detach() 
        if return_caches:
            return hidden_states_all, kv_caches
        return hidden_states_all


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/train/bridge.yaml")
    model = JointModel(cfg.joint.config)

    # dummy inputs
    dummy_num_image_tokens = 7
    q_lens = [
        dummy_num_image_tokens,
        cfg.cond_steps,
        cfg.horizon_steps,
    ]  # not considering text (padding)
    total_len = sum(q_lens)
    inputs_embeds = torch.randn(
        1,
        dummy_num_image_tokens,
        cfg.mixture.vlm.hidden_size,
    )  # no history
    proprio_embeds = torch.randn(
        1,
        cfg.cond_steps,
        cfg.mixture.proprio.hidden_size,
    )
    action_embeds = torch.randn(
        1,
        cfg.horizon_steps,
        cfg.mixture.action.hidden_size,
    )
    time_cond = None
    if cfg.action_expert_adaptive_mode:
        time_cond = torch.randn(1, cfg.time_hidden_size)

    kv_caches = model.build_mixture_caches()
    position_ids_all = {
        "vlm": torch.arange(dummy_num_image_tokens)[None],
        "proprio": torch.arange(cfg.cond_steps)[None],
        "action": torch.arange(cfg.horizon_steps)[None],
    }  # add batch dim

    # block attention
    proprio_start = dummy_num_image_tokens
    proprio_end = dummy_num_image_tokens + 1
    action_start = proprio_end
    causal_mask = torch.full(
        (1, total_len, total_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )  # smallest value, avoid using inf for softmax nan issues with padding
    causal_mask[:, :dummy_num_image_tokens, :dummy_num_image_tokens] = (
        0  # image/text attend to itself
    )
    causal_mask[:, proprio_start:proprio_end, :dummy_num_image_tokens] = (
        0  # proprio attend to image/text
    )
    causal_mask[:, action_start:, :dummy_num_image_tokens] = (
        0  # action attend to image/text
    )
    causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
        0  # proprio attend to itself
    )
    causal_mask[:, action_start:, proprio_start:] = (
        0  # action attend to itself and proprio
    )

    # Add the head dimension
    # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
    causal_mask = causal_mask.unsqueeze(1)

    # dummy denoising - naive action inference
    print("Initial action embeds", action_embeds)
    num_step = 3
    for _step in range(num_step):
        print("running dummy denoising step", _step)
        action_embeds = model(
            attention_mask=causal_mask,
            position_ids_all=position_ids_all,
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            kv_caches=kv_caches,
            time_cond=time_cond,
            cache_mode="no_append",
        )["action"]
        print("Updated action embeds", action_embeds)
