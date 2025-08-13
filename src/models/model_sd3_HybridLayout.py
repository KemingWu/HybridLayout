# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import inspect
import numbers
import types
import numpy as np
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.attention import FeedForward, _chunked_feed_forward, JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0, JointAttnProcessor2_0
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm, SD35AdaLayerNormZeroX
import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.attention_sd3_HybridLayout import JointAttnProcessor2_0_w_mask_XFormer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
    

def hack_mm_block_w_mask(block: JointTransformerBlock):

    block.attn.set_processor(JointAttnProcessor2_0_w_mask_XFormer())

    def new_forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        attention_mask = None,
    ):
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, attention_mask=attention_mask
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states

    block.forward = types.MethodType(new_forward, block) # monkey patch

def hack_model_attn_w_mask(model: SD3Transformer2DModel, attn_type="interact"):

    for block in model.transformer_blocks:
        hack_mm_block_w_mask(block)

    def crop_each_layer(hidden_states, list_layer_box):
        """
            hidden_states: [1, f, h, w, inner_dim]
            list_layer_box: List, length=f, each element is a Tuple of 4 elements (x1, y1, x2, y2)
        """
        token_list = []
        for layer_idx in range(hidden_states.shape[1]):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
                layer_token = hidden_states[:, layer_idx, y1:y2, x1:x2, :]
                bs, h, w, c = layer_token.shape
                layer_token = layer_token.reshape(bs, -1, c)
                token_list.append(layer_token)
        result = torch.cat(token_list, dim=1)

        img_segs = []
        curr = 0
        for token in token_list:
            token_l = token.shape[1]
            img_segs.append([curr, curr+token_l])
            curr += token_l

        seq_len = result.shape[1]
        
        if seq_len%8 != 0:
            pad_len = (seq_len//8 + 1) * 8 - seq_len
            result = torch.nn.functional.pad(result, (0, 0, 0, pad_len))
            img_segs[-1][-1] += pad_len

        return result, img_segs

    def calc_attn_mask_interact(img_segs, txt_segs, device, is_training):

        img_l = img_segs[-1][-1]
        
        txt_l = max(seg[-1] for seg in txt_segs)

        attn_mask_img2img = torch.ones(img_l, img_l)
        attn_mask_img2txt = torch.zeros(img_l, txt_l)
        attn_mask_txt2img = torch.zeros(txt_l, img_l)
        attn_mask_txt2txt = torch.ones(txt_l, txt_l)

        # txt_segs.insert(1, [0, 0]) for art w/o bg caption
        # print(f"img_segs len: {len(img_segs)}, txt_segs len: {len(txt_segs)}.\n")

        for img_seg, txt_seg in zip(img_segs, txt_segs):
            # if txt_seg == [0, 0]: print(f"semi-attention mask doing.\n")

            if txt_seg == [0, 0]:
                attn_mask_img2txt[img_seg[0]:img_seg[1], txt_segs[0][0]:txt_segs[0][1]] = 0
                attn_mask_txt2img[txt_segs[0][0]:txt_segs[0][1], img_seg[0]:img_seg[1]] = 0

            attn_mask_img2txt[img_seg[0]:img_seg[1], txt_seg[0]:txt_seg[1]] = 1
            attn_mask_txt2img[txt_seg[0]:txt_seg[1], img_seg[0]:img_seg[1]] = 1
        
        # NOTE: this is for FLUX
        # attn_mask_txt = torch.cat([attn_mask_txt2txt, attn_mask_txt2img], dim=1)
        # attn_mask_img = torch.cat([attn_mask_img2txt, attn_mask_img2img], dim=1)
        # attn_mask = torch.cat([attn_mask_txt, attn_mask_img], dim=0)

        attn_mask_txt = torch.cat([attn_mask_txt2img, attn_mask_txt2txt], dim=1)
        attn_mask_img = torch.cat([attn_mask_img2img, attn_mask_img2txt], dim=1)
        attn_mask = torch.cat([attn_mask_img, attn_mask_txt], dim=0)

        attn_mask = (1 - attn_mask) * -10000.0
        # attn_mask = attn_mask.to(dtype=torch.bfloat16, device=device) ### modify

        # print(f"is_training: {is_training}.\n")
        # NOTE: IMPORTANT for SD3 / SD3.5 training !!!!!
        if is_training:
            attn_mask = attn_mask.to(dtype=torch.float16, device=device)
        else:
            attn_mask = attn_mask.to(dtype=torch.bfloat16, device=device)  

        return attn_mask

    def calc_attn_mask_isolate(img_segs, txt_segs, device):

        img_l = img_segs[-1][-1]
        txt_l = txt_segs[-1][-1]

        attn_mask_img2img = torch.zeros(img_l, img_l)
        attn_mask_img2txt = torch.zeros(img_l, txt_l)
        attn_mask_txt2img = torch.zeros(txt_l, img_l)
        attn_mask_txt2txt = torch.ones(txt_l, txt_l)

        for img_seg in img_segs:
            attn_mask_img2img[img_seg[0]:img_seg[1], img_seg[0]:img_seg[1]] = 1

        txt_segs.insert(1, [0, 0])
        for img_seg, txt_seg in zip(img_segs, txt_segs):
            attn_mask_img2txt[img_seg[0]:img_seg[1], txt_seg[0]:txt_seg[1]] = 1
            attn_mask_txt2img[txt_seg[0]:txt_seg[1], img_seg[0]:img_seg[1]] = 1
        
        attn_mask_txt = torch.cat([attn_mask_txt2txt, attn_mask_txt2img], dim=1)
        attn_mask_img = torch.cat([attn_mask_img2txt, attn_mask_img2img], dim=1)
        attn_mask = torch.cat([attn_mask_txt, attn_mask_img], dim=0)

        attn_mask = (1 - attn_mask) * -10000.0
        attn_mask = attn_mask.to(dtype=torch.bfloat16, device=device)

        return attn_mask

    def new_forward(
        self,
        hidden_states: torch.FloatTensor,
        list_layer_box: List[Tuple] = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        txt_segs: list = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:  # False
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:  # False
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:  # False
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # print(f"hidden_states dtype: {hidden_states.dtype}, encoder_hidden_states dtype: {encoder_hidden_states.dtype}")

        bs, n_frames, channel_latent, height, width = hidden_states.shape  # [bs, f, c_latent, h, w]
        # assert bs == 1, "Only batch size 1 is supported for now."

        hidden_states = hidden_states.view(bs * n_frames, channel_latent, height, width)  # [bs * f, c_latent, h, w]
        hidden_states = self.pos_embed(hidden_states)                                     # [bs * f, h * w, inner_dim]
        hidden_states = hidden_states.view(bs, n_frames, height//2, width//2, self.inner_dim)  # [bs, f, h, w, inner_dim]
        full_hidden_states = hidden_states
        layer_pe = self.layer_pe.view(1, self.max_layer_num, 1, 1, self.inner_dim)  # [1, 5, 1, 1, 1536]
        hidden_states = hidden_states + layer_pe[:, :n_frames]    # [bs, f, h, w, inner_dim] + [1, f, 1, 1, inner_dim] -->  [bs, f, h, w, inner_dim]

        hidden_states, img_segs = crop_each_layer(hidden_states, list_layer_box)  # [bs, h1xw1 + h2xw2 + ... + hlxwl, c_latent]

        temb = self.time_text_embed(timestep, pooled_projections)             # [2], [2, 2048] --> [2, 1536]
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)  # [2, 333, 4096] --> [2, 333, 1536]; nn.Linear(4096, 1536, bias=True)

        real_l = hidden_states.shape[1] + encoder_hidden_states.shape[1]
        if attn_type == "interact":
            calc_attn_mask = calc_attn_mask_interact
        elif attn_type == "isolate":    
            calc_attn_mask = calc_attn_mask_isolate
        attention_mask = calc_attn_mask(img_segs, txt_segs, hidden_states.device, is_training=self.training)
        attention_mask = attention_mask[None, None, :real_l, :real_l].tile(1, self.config.num_attention_heads, 1, 1)
        attention_mask = attention_mask.expand(bs, -1, -1, -1)
        
        # hidden_states = hidden_states.to(dtype=torch.bfloat16)
        # attention_mask = attention_mask.to(dtype=torch.bfloat16)
        # encoder_hidden_states = encoder_hidden_states.to(dtype=torch.bfloat16)
        # print(f"hidden_states dtype: {hidden_states.dtype}, encoder_hidden_states dtype: {encoder_hidden_states.dtype}, attention_mask dtype: {attention_mask.dtype}")

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, attention_mask=attention_mask
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.fill_in_processed_tokens(hidden_states, full_hidden_states, list_layer_box)  # [bs, 236, inner_dim], [bs, f, h, w, inner_dim]
        hidden_states = hidden_states.reshape(bs, -1, self.inner_dim)  # [bs, f * full_len, inner_dim]

        hidden_states = self.norm_out(hidden_states, temb)  # [2, 4096, 1536], [2, 1536] --> [2, 4096, 1536] (AdaLayerNormContinuous, LN then AdaLN)
        hidden_states = self.proj_out(hidden_states)        # nn.Linear(1536, 64, bias=True); [2, 4096, 1536] --> [2, 4096, 64]  # [2, 5*4096, 64]


        # unpatchify
        patch_size = self.config.patch_size  # 2
        height = height // patch_size        # 128 // 2 = 64
        width = width // patch_size          # 128 // 2 = 64

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], n_frames, height, width, patch_size, patch_size, self.out_channels)
        )  # [2, 5, 64, 64, 2, 2, 16]
        hidden_states = torch.einsum("nfhwpqc->nfchpwq", hidden_states)  # [2, 5, 16, 64, 2, 64, 2]
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], n_frames, self.out_channels, height * patch_size, width * patch_size)
        )  # [2, 5, 16, 128, 128]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    model.forward = types.MethodType(new_forward, model) # monkey patch

@maybe_allow_in_graph
class JointTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
    ):
        super().__init__()

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=1e-6,
        )

        if use_dual_attention:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                processor=processor,
                qk_norm=qk_norm,
                eps=1e-6,
            )
        else:
            self.attn2 = None

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
    ):
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states

## diffusers/models/transformers/transformer_sd3.py
class HybridLayoutSD3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        out_channels (`int`, defaults to 16): Number of output channels.

    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        max_layer_num: int = 10,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        default_out_channels = in_channels                                                      # 16
        self.out_channels = out_channels if out_channels is not None else default_out_channels  # 16
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim       # 24 * 64 = 1536

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.max_layer_num = max_layer_num
        self.layer_pe = nn.Parameter(torch.empty(1, self.max_layer_num, 1, 1, self.inner_dim))
        nn.init.trunc_normal_(self.layer_pe, mean=0.0, std=0.02, a=-2.0, b=2.0)
        # nn.init.zeros_(self.layer_pe)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)
            
    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedJointAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def crop_each_layer(self, hidden_states, list_layer_box):
        """
            hidden_states: [1, f, h, w, c_latent]
            list_layer_box: List, length=f, each element is a Tuple of 4 elements (x1, y1, x2, y2)
        """
        token_list = []
        for layer_idx in range(hidden_states.shape[1]):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
                layer_token = hidden_states[:, layer_idx, y1:y2, x1:x2, :]
                bs, h, w, c = layer_token.shape
                layer_token = layer_token.reshape(bs, -1, c)
                token_list.append(layer_token)
        result = torch.cat(token_list, dim=1)
        return result

    def fill_in_processed_tokens(self, hidden_states, full_hidden_states, list_layer_box):
        """
            hidden_states: [1, h1xw1 + h2xw2 + ... + hlxwl , inner_dim]
            full_hidden_states: [1, f, h, w, inner_dim]
            list_layer_box: List, length=f, each element is a Tuple of 4 elements (x1, y1, x2, y2)
        """
        used_token_len = 0
        bs = hidden_states.shape[0]
        for layer_idx in range(full_hidden_states.shape[1]):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
                full_hidden_states[:, layer_idx, y1:y2, x1:x2, :] = hidden_states[:, used_token_len: used_token_len + (y2-y1) * (x2-x1), :].reshape(bs, y2-y1, x2-x1, -1)
                used_token_len = used_token_len + (y2-y1) * (x2-x1)
        return full_hidden_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        list_layer_box: List[Tuple] = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        txt_segs: list = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:  # False
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:  # False
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:  # False
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        bs, n_frames, channel_latent, height, width = hidden_states.shape  # [bs, f, c_latent, h, w]
        # assert bs == 1, "Only batch size 1 is supported for now."

        hidden_states = hidden_states.view(bs * n_frames, channel_latent, height, width)  # [bs * f, c_latent, h, w]
        hidden_states = self.pos_embed(hidden_states)                                     # [bs * f, h * w, inner_dim]
        hidden_states = hidden_states.view(bs, n_frames, height//2, width//2, self.inner_dim)  # [bs, f, h, w, inner_dim]
        full_hidden_states = hidden_states
        layer_pe = self.layer_pe.view(1, self.max_layer_num, 1, 1, self.inner_dim)  # [1, 5, 1, 1, 1536]
        hidden_states = hidden_states + layer_pe[:, :n_frames]    # [bs, f, h, w, inner_dim] + [1, f, 1, 1, inner_dim] -->  [bs, f, h, w, inner_dim]

        hidden_states = self.crop_each_layer(hidden_states, list_layer_box)  # [bs, h1xw1 + h2xw2 + ... + hlxwl, c_latent]

        temb = self.time_text_embed(timestep, pooled_projections)             # [2], [2, 2048] --> [2, 1536]
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)  # [2, 333, 4096] --> [2, 333, 1536]; nn.Linear(4096, 1536, bias=True)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.fill_in_processed_tokens(hidden_states, full_hidden_states, list_layer_box)  # [bs, 236, inner_dim], [bs, f, h, w, inner_dim]
        hidden_states = hidden_states.reshape(bs, -1, self.inner_dim)  # [bs, f * full_len, inner_dim]

        hidden_states = self.norm_out(hidden_states, temb)  # [2, 4096, 1536], [2, 1536] --> [2, 4096, 1536] (AdaLayerNormContinuous, LN then AdaLN)
        hidden_states = self.proj_out(hidden_states)        # nn.Linear(1536, 64, bias=True); [2, 4096, 1536] --> [2, 4096, 64]  # [2, 5*4096, 64]


        # unpatchify
        patch_size = self.config.patch_size  # 2
        height = height // patch_size        # 128 // 2 = 64
        width = width // patch_size          # 128 // 2 = 64

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], n_frames, height, width, patch_size, patch_size, self.out_channels)
        )  # [2, 5, 64, 64, 2, 2, 16]
        hidden_states = torch.einsum("nfhwpqc->nfchpwq", hidden_states)  # [2, 5, 16, 64, 2, 64, 2]
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], n_frames, self.out_channels, height * patch_size, width * patch_size)
        )  # [2, 5, 16, 128, 128]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)