import torch
import torch.nn as nn
import types
from typing import Any, Dict, List, Optional, Union, Tuple

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock

from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from src.models.attention_flux_HybridLayout import FluxAttnProcessor2_0_w_mask_XFormer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def hack_mm_block_w_mask(block: FluxTransformerBlock):

    block.attn.set_processor(FluxAttnProcessor2_0_w_mask_XFormer())

    def new_forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        attention_mask = None,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    block.forward = types.MethodType(new_forward, block) # monkey patch

def hack_single_block_w_mask(block: FluxSingleTransformerBlock):

    block.attn.set_processor(FluxAttnProcessor2_0_w_mask_XFormer())

    def new_forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        attention_mask = None,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    block.forward = types.MethodType(new_forward, block) # monkey patch


def hack_model_attn_w_mask(model: FluxTransformer2DModel, attn_type="interact"):

    for block in model.transformer_blocks:
        hack_mm_block_w_mask(block)

    for block in model.single_transformer_blocks:
        hack_single_block_w_mask(block)

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

    def calc_attn_mask_interact(img_segs, txt_segs, device):

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
            attn_mask_img2txt[img_seg[0]:img_seg[1], txt_seg[0]:txt_seg[1]] = 1
            attn_mask_txt2img[txt_seg[0]:txt_seg[1], img_seg[0]:img_seg[1]] = 1
        
        attn_mask_txt = torch.cat([attn_mask_txt2txt, attn_mask_txt2img], dim=1)
        attn_mask_img = torch.cat([attn_mask_img2txt, attn_mask_img2img], dim=1)
        attn_mask = torch.cat([attn_mask_txt, attn_mask_img], dim=0)

        attn_mask = (1 - attn_mask) * -10000.0
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
        hidden_states: torch.Tensor,
        list_layer_box: List[Tuple] = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        txt_segs: list = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        bs, n_frames, channel_latent, height, width = hidden_states.shape  # [bs, f, c_latent, h, w]

        hidden_states = hidden_states.view(bs, n_frames, channel_latent, height // 2, 2, width // 2, 2)  # [bs, f, c_latent, h/2, 2, w/2, 2]
        hidden_states = hidden_states.permute(0, 1, 3, 5, 2, 4, 6) # [bs, f, h/2, w/2, c_latent, 2, 2]
        hidden_states = hidden_states.reshape(bs, n_frames, height // 2, width // 2, channel_latent * 4) # [bs, f, h/2, w/2, c_latent*4]
        hidden_states = self.x_embedder(hidden_states) # [bs, f, h/2, w/2, inner_dim]

        full_hidden_states = torch.zeros_like(hidden_states) # [bs, f, h/2, w/2, inner_dim]
        layer_pe = self.layer_pe.view(1, self.max_layer_num, 1, 1, self.inner_dim)  # [1, mf, 1, 1, inner_dim]
        hidden_states = hidden_states + layer_pe[:, :n_frames]    # [bs, f, h/2, w/2, inner_dim] + [1, f, 1, 1, inner_dim] -->  [bs, f, h/2, w/2, inner_dim]
        hidden_states, img_segs = crop_each_layer(hidden_states, list_layer_box)  # [bs, L, inner_dim]

        real_l = hidden_states.shape[1] + encoder_hidden_states.shape[1]
        if attn_type == "interact":
            calc_attn_mask = calc_attn_mask_interact
        elif attn_type == "isolate":    
            calc_attn_mask = calc_attn_mask_isolate
        attention_mask = calc_attn_mask(img_segs, txt_segs, hidden_states.device)
        attention_mask = attention_mask[None, None, :real_l, :real_l].tile(1, self.config.num_attention_heads, 1, 1)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

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
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        
        hidden_states = self.fill_in_processed_tokens(hidden_states, full_hidden_states, list_layer_box)  # [bs, f, h, w, inner_dim]
        hidden_states = hidden_states.view(bs, -1, self.inner_dim)  # [bs, f * full_len, inner_dim]

        hidden_states = self.norm_out(hidden_states, temb) # [bs, f * full_len, inner_dim]
        hidden_states = self.proj_out(hidden_states) # [bs, f * full_len, c_latent*4]

        # unpatchify
        hidden_states = hidden_states.view(bs, n_frames, height//2, width//2, channel_latent, 2, 2) # [bs, f, h/2, w/2, c_latent, 2, 2]
        hidden_states = hidden_states.permute(0, 1, 4, 2, 5, 3, 6)
        output = hidden_states.reshape(bs, n_frames, channel_latent, height, width)  # [bs, f, c_latent, h, w]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    model.forward = types.MethodType(new_forward, model) # monkey patch


class HybridLayoutFluxTransformer2DModel(FluxTransformer2DModel):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        max_layer_num: int = 10,
    ):
        super(FluxTransformer2DModel, self).__init__()
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.max_layer_num = max_layer_num
        self.layer_pe = nn.Parameter(torch.empty(1, self.max_layer_num, 1, 1, self.inner_dim))
        nn.init.trunc_normal_(self.layer_pe, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def crop_each_layer(self, hidden_states, list_layer_box):
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
        hidden_states: torch.Tensor,
        list_layer_box: List[Tuple] = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        txt_segs: list = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

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
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        bs, n_frames, channel_latent, height, width = hidden_states.shape  # [bs, f, c_latent, h, w]

        hidden_states = hidden_states.view(bs, n_frames, channel_latent, height // 2, 2, width // 2, 2)  # [bs, f, c_latent, h/2, 2, w/2, 2]
        hidden_states = hidden_states.permute(0, 1, 3, 5, 2, 4, 6) # [bs, f, h/2, w/2, c_latent, 2, 2]
        hidden_states = hidden_states.reshape(bs, n_frames, height // 2, width // 2, channel_latent * 4) # [bs, f, h/2, w/2, c_latent*4]
        hidden_states = self.x_embedder(hidden_states) # [bs, f, h/2, w/2, inner_dim]

        full_hidden_states = torch.zeros_like(hidden_states) # [bs, f, h/2, w/2, inner_dim]
        layer_pe = self.layer_pe.view(1, self.max_layer_num, 1, 1, self.inner_dim)  # [1, mf, 1, 1, inner_dim]
        hidden_states = hidden_states + layer_pe[:, :n_frames]    # [bs, f, h/2, w/2, inner_dim] + [1, f, 1, 1, inner_dim] -->  [bs, f, h/2, w/2, inner_dim]
        hidden_states = self.crop_each_layer(hidden_states, list_layer_box)  # [bs, L, inner_dim]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

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
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        
        hidden_states = self.fill_in_processed_tokens(hidden_states, full_hidden_states, list_layer_box)  # [bs, f, h, w, inner_dim]
        hidden_states = hidden_states.view(bs, -1, self.inner_dim)  # [bs, f * full_len, inner_dim]

        hidden_states = self.norm_out(hidden_states, temb) # [bs, f * full_len, inner_dim]
        hidden_states = self.proj_out(hidden_states) # [bs, f * full_len, c_latent*4]

        # unpatchify
        hidden_states = hidden_states.view(bs, n_frames, height//2, width//2, channel_latent, 2, 2) # [bs, f, h/2, w/2, c_latent, 2, 2]
        hidden_states = hidden_states.permute(0, 1, 4, 2, 5, 3, 6)
        output = hidden_states.reshape(bs, n_frames, channel_latent, height, width)  # [bs, f, c_latent, h, w]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)