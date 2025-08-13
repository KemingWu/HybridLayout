import os
import shutil
import json

import re
import math
import torch
import torch.nn as nn
import itertools
from typing import Any, Callable, Dict, List, Optional, Union

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline, retrieve_timesteps, StableDiffusion3PipelineOutput
from diffusers.loaders.lora_pipeline import SD3LoraLoaderMixin
from diffusers.utils import scale_lora_layers, USE_PEFT_BACKEND, unscale_lora_layers
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin

from PIL import Image, ImageDraw, ImageFont

from diffusers.utils import (
    USE_PEFT_BACKEND,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
    deprecate,
    get_adapter_name,
    get_peft_kwargs,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    logging,
    scale_lora_layers,
)
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.loaders.lora_conversion_utils import (
    _convert_kohya_flux_lora_to_diffusers,
    _convert_non_diffusers_lora_to_diffusers,
    _convert_xlabs_flux_lora_to_diffusers,
    _maybe_map_sgm_blocks_to_diffusers,
)


_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True


if is_transformers_available():
    from diffusers.models.lora import text_encoder_attn_modules, text_encoder_mlp_modules

logger = logging.get_logger(__name__)

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"
TRANSFORMER_NAME = "transformer"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

def greedy_merge(ids_segments_reg, tokenizer, max_seg_len, pos_bias=0):
    seg_postions = []
    ids_subprompts = []
    curr_ids_subprompt = []
    while len(ids_segments_reg) > 0 or len(curr_ids_subprompt) > 0:
        if len(ids_segments_reg) == 0 or len(curr_ids_subprompt) + len(ids_segments_reg[0]) > max_seg_len:
            curr_ids_subprompt = [tokenizer.bos_token_id] + curr_ids_subprompt + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * max_seg_len
            curr_ids_subprompt = curr_ids_subprompt[:(max_seg_len+2)] # 77
            ids_subprompts.append(curr_ids_subprompt)
            curr_ids_subprompt = []
        else:
            ids_segment_reg = ids_segments_reg.pop(0)
            start_pos = (len(ids_subprompts) + pos_bias) * (max_seg_len+2) + len(curr_ids_subprompt) + 1 # add 1 for the bos token
            end_pos = start_pos + len(ids_segment_reg)
            curr_ids_subprompt += ids_segment_reg
            seg_postions.append([start_pos, end_pos])

    return ids_subprompts, seg_postions

def clip_partition_prompt(
    prompts, 
    tokenizer, 
    max_seg_len=75, 
    tokenize_max_length=512,
    num_subprompts_limit=8,
):

    ### gather subprompts
    gather_ids_subprompts = []
    max_num_subprompts = 0
    for prompt in prompts:
        segments = re.split(r"(?<=[,.])", prompt) # split prompt by (,.)
        segments = [seg.strip(' ') for seg in segments if len(seg.strip(' ')) > 0]
        if len(segments) == 0:
            ids_subprompts = [[tokenizer.bos_token_id] + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * max_seg_len]
        else:
            ids_segments = tokenizer(
                segments, 
                max_length=tokenize_max_length, 
                truncation=False, 
                add_special_tokens=False,
            ).input_ids # tokenize the segments without special token

            ids_segments_reg = []
            for ids_segment in ids_segments:
                while len(ids_segment) > max_seg_len:
                    ids_segments_reg.append(ids_segment[:max_seg_len])
                    ids_segment = ids_segment[max_seg_len:]
                ids_segments_reg.append(ids_segment)

            # greedy merge
            ids_subprompts, _ = greedy_merge(ids_segments_reg, tokenizer, max_seg_len)

        ids_subprompts = ids_subprompts[:num_subprompts_limit] # avoid OOM error (max 616 tokens)
        gather_ids_subprompts.append(ids_subprompts)
        max_num_subprompts = max(max_num_subprompts, len(ids_subprompts))
    ### padding
    pad_mask = []
    for ids_subprompts in gather_ids_subprompts:
        num_subprompts = len(ids_subprompts)
        pad_len = max_num_subprompts - num_subprompts
        paddings = [([tokenizer.bos_token_id] + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * max_seg_len) for _ in range(pad_len)]
        ids_subprompts += paddings
        pad_mask.append([1]*num_subprompts+[0]*pad_len)

    gather_ids_subprompts = torch.tensor(gather_ids_subprompts) # (b, p, max_seg_len+2)
    pad_mask = torch.tensor(pad_mask) # (b, p)

    return gather_ids_subprompts, pad_mask

def _get_clip_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt,
    num_images_per_prompt: int = 1,
    tokenize_max_length=512,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    
    text_input_ids, pad_mask = clip_partition_prompt(
        prompt,
        tokenizer,
        tokenize_max_length=tokenize_max_length,
    )
    b, p, l = text_input_ids.shape
    text_input_ids = text_input_ids.reshape(b*p, l).to(device=device)

    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]

    pooled_prompt_embeds = pooled_prompt_embeds.reshape(b, p, -1)[:, 0, :] # (b, d)
    prompt_embeds = prompt_embeds.reshape(b, p, l, -1) # (b, p, l, d)
    prompt_embeds = prompt_embeds * pad_mask[..., None, None].to(prompt_embeds)
    prompt_embeds = prompt_embeds.reshape(b, p*l, -1) # (b, p*l, d)
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    seg_positions = [[0, prompt_embeds.shape[1]]]
    return prompt_embeds, pooled_prompt_embeds, seg_positions

def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    max_total_sequence_length: int = 2048,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = 1

    curr = 0
    emb_list = []
    text_segs = []
    for p in prompt:
        if p == "":
            text_segs.append([0, 0])
            continue
        text_inputs = tokenizer(
            p,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        emb_list.append(prompt_embeds)
        text_segs.append([curr, curr+prompt_embeds.shape[1]])

        curr += prompt_embeds.shape[1]
        if curr > max_total_sequence_length:
            break

    if len(prompt) == 1 and prompt[0] == "":
        text_segs = []
        text_inputs = tokenizer(
            prompt[0],
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        emb_list.append(prompt_embeds)
        text_segs.append([curr, curr+prompt_embeds.shape[1]])
        curr += prompt_embeds.shape[1]

    prompt_embeds = torch.cat(emb_list, dim=1)

    _, seq_len, _ = prompt_embeds.shape

    if seq_len % 8 != 0:
        pad_len = (seq_len//8 + 1) * 8 - seq_len
        prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad_len))

        txt_l = max(seg[-1] for seg in text_segs)
        for seg in text_segs:
            if seg[-1] == txt_l:
                seg[-1] += pad_len
                break

        seq_len += pad_len

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, text_segs

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: list,
    t5_prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    joint_attention_dim: int = 4096,
    max_sequence_length: int = 512,
    max_total_sequence_length: int = 2048,
    disable_clip: bool = False,
    disable_t5: bool = False,
    device=None,
):
    assert not (disable_clip and disable_t5)
    prompt = [prompt] if isinstance(prompt, str) else prompt
    t5_prompt = prompt if t5_prompt is None else t5_prompt
    t5_prompt = [t5_prompt] if isinstance(t5_prompt, str) else t5_prompt

    batch_size = 1

    text_encoder, tokenizer = text_encoders[0], tokenizers[0]
    # We only use the pooled prompt output from the CLIPTextModel
    device = text_encoder.device
    dtype = text_encoder.dtype

    clip_prompt_embeds_1, pooled_prompt_embeds_1, clip_seg_positions_1 = _get_clip_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt[0],
        num_images_per_prompt=num_images_per_prompt,
        tokenize_max_length=max_sequence_length,
        device=device if device is not None else text_encoder.device,
    )

    text_encoder, tokenizer = text_encoders[1], tokenizers[0] # use exactly the same tokenizer to guarantee one-to-one correspondence
    ori_pad_token_id = tokenizer.pad_token_id
    tokenizer.pad_token_id = tokenizers[1].pad_token_id

    clip_prompt_embeds_2, pooled_prompt_embeds_2, clip_seg_positions_2 = _get_clip_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt[0],
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoder.device,
    )
    
    tokenizer.pad_token_id = ori_pad_token_id
    
    clip_prompt_embeds = torch.cat([clip_prompt_embeds_1, clip_prompt_embeds_2], dim=-1)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)

    prompt_embeds = torch.nn.functional.pad(   
        clip_prompt_embeds, (0, joint_attention_dim - clip_prompt_embeds.shape[-1])
    )

    clip_seg_positions_1 = [[0, prompt_embeds.shape[1]]]

    if disable_clip:
        prompt_embeds = None
    
    if text_encoders[-1] is not None and not disable_t5:
        text_encoder, tokenizer = text_encoders[-1], tokenizers[-1]
        t5_prompt_embed, t5_text_segs = _get_t5_prompt_embeds(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            max_total_sequence_length=max_total_sequence_length,
            device=device if device is not None else text_encoder.device,
        )
        if prompt_embeds is None:
            prompt_embeds = t5_prompt_embed
            text_segs = t5_text_segs
        else:
            # print(f"prompt_embeds shape is {prompt_embeds.shape}")
            # print(f"t5_prompt_embed shape is {t5_prompt_embed.shape}")
            offset = prompt_embeds.shape[1]
            text_segs = [[0, t5_text_segs[0][1] + offset]] + [
                [0, 0] if pos == [0, 0] else [pos[0] + offset, pos[1] + offset] for pos in t5_text_segs[1:]
            ]
            prompt_embeds = torch.cat([prompt_embeds, t5_prompt_embed], dim=-2)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    # zero-padding for xformers attention
    if seq_len % 8 != 0:
        pad_seq_len = math.ceil(seq_len / 8) * 8 - seq_len
        prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad_seq_len))

        txt_l = max(seg[-1] for seg in text_segs)
        for seg in text_segs:
            if seg[-1] == txt_l:
                seg[-1] += pad_seq_len
                break
    
    # prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)
    # pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch.bfloat16)
    # print(f"prompt_embeds dtype: {prompt_embeds.dtype}, pooled_prompt_embeds dtype: {pooled_prompt_embeds.dtype}")
    return prompt_embeds, pooled_prompt_embeds, text_segs

class HybridLayoutStableDiffusion3Pipeline(StableDiffusion3Pipeline):

    def prepare_latents(
        self,
        batch_size,
        num_layers,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        # 计算单个 noise 的形状
        noise_shape = (
            1,  # 单个 batch
            1,  # 单个 layer
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        # 检查 generator 的一致性
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 生成单一 noise
        single_noise = randn_tensor(noise_shape, generator=generator, device=device, dtype=dtype)

        # 将 noise 广播到目标形状
        latents = single_noise.expand(batch_size, num_layers, -1, -1, -1)

        return latents


    def encode_prompt(
        self,
        prompt: list,
        t5_prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        t5_negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        disable_clip: bool = False,
        disable_t5: bool = False,
    ):

        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        # batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]    ### batch_size = 1
        batch_size = 1

        if prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds, text_segs = encode_prompt(
                text_encoders=[self.text_encoder, self.text_encoder_2, self.text_encoder_3],
                tokenizers=[self.tokenizer, self.tokenizer_2, self.tokenizer_3],
                prompt=prompt,
                t5_prompt=t5_prompt,
                num_images_per_prompt=num_images_per_prompt,
                joint_attention_dim=self.transformer.config.joint_attention_dim,
                disable_clip=disable_clip,
                disable_t5=disable_t5,
            )
            prompt_embeds = prompt_embeds[:,:1024,:]

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_segs = encode_prompt(
                text_encoders=[self.text_encoder, self.text_encoder_2, self.text_encoder_3],
                tokenizers=[self.tokenizer, self.tokenizer_2, self.tokenizer_3],
                prompt=negative_prompt,
                t5_prompt=t5_negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                joint_attention_dim=self.transformer.config.joint_attention_dim,
                disable_clip=disable_clip,
                disable_t5=disable_t5,
            )

            # pad prompt_embeds to the same length
            if prompt_embeds.shape[1] > negative_prompt_embeds.shape[1]:
                negative_prompt_embeds = torch.nn.functional.pad(
                    negative_prompt_embeds, (0, 0, 0, prompt_embeds.shape[1] - negative_prompt_embeds.shape[1])
                )
            elif prompt_embeds.shape[1] < negative_prompt_embeds.shape[1]:
                prompt_embeds = torch.nn.functional.pad(
                    prompt_embeds, (0, 0, 0, negative_prompt_embeds.shape[1] - prompt_embeds.shape[1])
                )

        # dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        # prompt_embeds = prompt_embeds.to(dtype=torch.float16)
        # negative_prompt_embeds = negative_prompt_embeds.to(dtype=torch.float16)
        # pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float16)
        # negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=torch.float16)
        # print(f"prompt_embeds dtype: {prompt_embeds.dtype}, pooled_prompt_embeds dtype: {pooled_prompt_embeds.dtype}")
        # print(f"negative_prompt_embeds dtype: {negative_prompt_embeds.dtype}, negative_pooled_prompt_embeds dtype: {negative_pooled_prompt_embeds.dtype}")
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, text_segs, negative_text_segs

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        validation_box: List[tuple] = None,
        t5_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        t5_negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        disable_clip: bool = False,
        disable_t5: bool = False,
        num_layers: int = 5,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            None,
            t5_prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=t5_negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = 1  ### default value

        device = self._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            text_segs,
            negative_text_segs
        ) = self.encode_prompt(
            prompt=prompt,
            t5_prompt=t5_prompt,
            negative_prompt=negative_prompt,
            t5_negative_prompt=t5_negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            disable_clip=disable_clip,
            disable_t5=disable_t5,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_layers,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    list_layer_box=validation_box,
                    timestep=timestep,
                    txt_segs=text_segs,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        # create a grey image / latent
        pixel_grey = torch.zeros(
            size=(latents.shape[0], latents.shape[1], 3, latents.shape[3]*8, latents.shape[4]*8),
            device=latents.device, dtype=latents.dtype)
        grey_bs, grey_num_layers, grey_c, grey_h, grey_w = pixel_grey.shape
        pixel_grey = pixel_grey.view(grey_bs * grey_num_layers, grey_c, grey_h, grey_w)
        latent_grey = self.vae.encode(pixel_grey).latent_dist.sample()
        latent_grey = (latent_grey - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latent_grey = latent_grey.reshape(grey_bs, grey_num_layers, latents.shape[2], latents.shape[3], latents.shape[4])  # torch.Size([bs, 5, 16, 64, 64])

        for layer_idx in range(latent_grey.shape[1]):
            x1, y1, x2, y2 = validation_box[layer_idx]
            x1, y1, x2, y2 = x1 // 8, y1 // 8, x2 // 8, y2 // 8
            latent_grey[:, layer_idx, :, y1:y2, x1:x2] = latents[:, layer_idx, :, y1:y2, x1:x2]
        latents = latent_grey

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            bs, num_layers, c, h, w = latents.shape
            latents = latents.reshape(bs * num_layers, c, h, w)
            image = self.vae.decode(latents, return_dict=False)[0]
            print(image.shape)
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)

    @staticmethod
    def draw_box_desc(pil_img: Image, bboxes: List[List[float]], prompt: List[str]) -> Image:
        """Utility function to draw bbox on the image"""
        color_list = ['red', 'blue', 'yellow', 'purple', 'green', 'black', 'brown', 'orange', 'white', 'gray']
        width, height = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        font_folder = "/openseg_blob/keming/workspace/multilayer_v2/exps/v04sv02-sd3.5-layoutsam-generic"
        font_path = os.path.join(font_folder, 'Rainbow-Party-2.ttf')
        font = ImageFont.truetype(font_path, 30)

        for box_id in range(len(bboxes)):
            obj_box = bboxes[box_id]
            text = prompt[box_id]
            fill = 'black'
            for color in prompt[box_id].split(' '):
                if color in color_list:
                    fill = color
            text = text.split(',')[0]
            x_min, y_min, x_max, y_max = (
                obj_box[0] * width,
                obj_box[1] * height,
                obj_box[2] * width,
                obj_box[3] * height,
            )
            draw.rectangle(
                [int(x_min), int(y_min), int(x_max), int(y_max)],
                outline=fill,
                width=4,
            )
            draw.text((int(x_min), int(y_min)), text, fill=fill, font=font)

        return pil_img