# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from .internvl_chat.modeling_intern_vit import InternVisionModel, InternVisionEncoderLayer, InternVisionEncoder

class FiLMedVisionTransformerBlock(nn.Module):
    """
    Wrapper for ViT blocks that adds components to implement FiLM language conditioning.

    Modulates visual feature embeddings via
        x = (1 + gamma) * x + beta,
    where x is visual feature and gamma and beta are learned projections of the average language embedding.
    gamma and beta have D dimensions each, where D is the number of hidden dimensions in the ViT's features.

    NOTE #1 (Moo Jin):
    In convolutional neural architectures, the "feature" in FiLM is an entire feature map, i.e., each channel in a
    convolutional layer (so gamma and beta have C dimensions, where C is the number of channels). Therefore, FiLM's
    scaling and shifting is applied across all spatial locations for conv nets -- i.e., it is spatially agnostic.

    For vision transformer architectures, you may consider individual patch embeddings as individual "features" at first
    instinct, but this would make FiLM scaling and shifting spatially local. In order to make the modulation spatially
    global like in convolutional architectures, we should apply the scaling and shifting to each dimension of each patch
    embedding. I.e., gamma and beta should have D dimensions, where D is the number of dimensions in a visual embedding.

    NOTE #2 (Moo Jin):
    x = (1 + gamma) * x + beta is used in the original FiLM paper as opposed to x = gamma * x + beta (see section 7.2 in
    https://arxiv.org/pdf/1709.07871.pdf). Since gamma and beta are close to zero upon initialization, this leads to an
    identity transformation at the beginning of training, which minimizes perturbation to the pretrained representation.
    """

    def __init__(
        self,
        block: InternVisionEncoderLayer,
        vision_dim: int,
        llm_dim: int,
    ):
        """
        Initializes FiLM ViT block wrapper.

        Args:
            block (timm.models.vision_transformer.Block): Vision transformer block.
            vision_dim (int): Number of hidden dimensions in visual embeddings.
            llm_dim (int): Number of hidden dimensions in language embeddings.
        """
        super().__init__()
        self.block = block
        # Initialize gamma and beta projectors
        self.scale = nn.Linear(llm_dim, vision_dim)
        self.shift = nn.Linear(llm_dim, vision_dim)

    def forward(self, hidden_states, average_language_embedding):
        """
        Overrides the vision transformer block forward pass to use FiLM.

        Args:
            hidden_states (torch.Tensor): Visual input embeddings, (batch_size, vision_seq_len, vision_dim).
            average_language_embedding (torch.Tensor): Average language embedding for task, (batch_size, llm_dim).
        """
        # Project average language embedding to visual embedding space to get gamma and beta
        gamma = self.scale(average_language_embedding)  # (batch_size, vision_dim)
        beta = self.shift(average_language_embedding)  # (batch_size, vision_dim)

        # Pass visual inputs through attention portion of original block
        hidden_states = hidden_states + self.block.drop_path1(self.block.attn(self.block.norm1(hidden_states).to(hidden_states.dtype)) * self.block.ls1)

        # Modulate intermediate visual representations via FiLM
        hidden_states = hidden_states * (1 + gamma.view(gamma.shape[0], 1, gamma.shape[1])) + beta.view(beta.shape[0], 1, beta.shape[1])

        # Pass visual inputs through attention portion of original block
        hidden_states = hidden_states + self.block.drop_path2(self.block.mlp(self.block.norm2(hidden_states).to(hidden_states.dtype)) * self.block.ls2)

        return hidden_states

class FiLMedInternVisionEncoder(InternVisionEncoder):

    def forward(
            self,
            inputs_embeds,
            language_embeddings: torch.Tensor,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                # Use non-reentrant checkpointing to avoid DDP "marked ready twice" errors
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states,
                    language_embeddings,
                    use_reentrant=False,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    language_embeddings,
                )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )

class FiLMedInternVisionBackbone(nn.Module):
    """
    Wrapper for InternVL's vision backbone that implements feature-wise linear modulation (FiLM).

    Wraps the Vision Transformers in the vision backbone to enable language conditioning through FiLM.
    """

    def __init__(
        self,
        vision_backbone: InternVisionModel,
        llm_dim: int = 896,
    ) -> None:
        """
        Initializes FiLM wrapper.

        Args:
            vision_backbone (InternVisionModel): Base vision backbone.
            llm_dim (int): Dimension of language model embeddings.
        """
        super().__init__()
        self.vision_backbone = vision_backbone
        self.llm_dim = llm_dim

        # Wrap vision transformers
        self._wrap_vit(self.vision_backbone.encoder)

    def _wrap_vit(self, vit: InternVisionEncoder) -> None:
        """
        Creates wrapper around an individual vision transformer to allow for infusion of language inputs.

        Args:
            vit (InternVisionEncoder): Original vision transformer.
        """
        # Wrap vision transformer blocks
        block_wrappers = []
        for block in vit.layers:
            block_wrappers.append(
                FiLMedVisionTransformerBlock(block=block, vision_dim=vit.config.hidden_size, llm_dim=self.llm_dim)
            )
        vit.layers = nn.Sequential(*block_wrappers)

        # Wrap vision transformer with new class that overrides functions used for forward pass
        vit.__class__ = FiLMedInternVisionEncoder

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            language_embeddings: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        encoder_outputs = self.vision_backbone.encoder(
            inputs_embeds=hidden_states,
            language_embeddings=language_embeddings,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )