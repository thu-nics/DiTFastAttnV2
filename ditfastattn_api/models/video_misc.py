import torch
import torch.nn as nn
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
# from diffusers.pipelines.dit.pipeline_dit import DiTPipeline, randn_tensor
from diffusers import CogVideoXPipeline, WanPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward, JointTransformerBlock, BasicTransformerBlock
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from ditfastattn_api.modules.dfa_ffn import DiTFastAttnFFN
import numpy as np
import math
from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler

block_class = JointTransformerBlock
basic_block_class = BasicTransformerBlock
cogvideox_block_class = CogVideoXBlock
wan_block_class = WanTransformerBlock
attn_class = Attention
ffn_class = FeedForward

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.no_grad()
def inference_fn_plan_update_iop_per_head_cogvideox(
    pipe: CogVideoXPipeline,
    dfa_config,
    alpha, 
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 226,
):
    
    height = height or pipe.transformer.config.sample_height * pipe.vae_scale_factor_spatial
    width = width or pipe.transformer.config.sample_width * pipe.vae_scale_factor_spatial
    num_frames = num_frames or pipe.transformer.config.sample_frames

    num_videos_per_prompt = 1

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, timesteps)
    pipe._num_timesteps = len(timesteps)

   # 5. Prepare latents
    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = pipe.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * pipe.vae_scale_factor_temporal

    latent_channels = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        pipe._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, cogvideox_block_class):
            # for MMDiT
            if isinstance(module.attn1, attn_class):
                module.attn1.processor.compression_influences = {}
            #     module.attn.processor.dfa_config = dfa_config
                module.attn1.processor.forward_mode = "calib_collect_info"
                module.attn1.processor.dfa_config = dfa_config
                module.attn1.processor.alpha = alpha
                module.attn1.processor.timestep_block_mask = {}
                module.attn1.processor.prev_calib_output = None
                module.attn1.processor.cached_output = None
                module.attn1.processor.output_share_dict = {}
                module.attn1.processor.num_frames = ((num_frames - 1) // 4 + 1) // 2
                module.attn1.processor.height = height // 16
                module.attn1.processor.width = width // 16

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            print(f"-------------------step {i}--------------------")

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            for name, module in pipe.transformer.named_modules():
                module.timestep_index = i
                if hasattr(module, "stepi"):
                    module.stepi = i
                if isinstance(module, attn_class):
                    module.processor.stepi = i

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            if use_dynamic_cfg:
                pipe._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)


            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = pipe.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()


    # save plan
    # torch.save(dfa_config.plan, f"cache/plan_{height}_cogvideox_{alpha}.json")

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, cogvideox_block_class):
            # for MMDiT
            if isinstance(module.attn1, attn_class):
                module.attn1.compression_influences = {}
                module.attn1.processor.cached_output = None
                module.attn1.processor.cached_residual = None
            #     module.attn.processor.cache_residual_forced = False
                module.attn1.processor.dfa_config=None
                module.attn1.processor.forward_mode = "perhead_normal"
                module.attn1.processor.prev_calib_output = None
                dfa_config.wt[module.attn1.name] = module.attn1.processor.wt
                dfa_config.output_share_dict[module.attn1.name] = module.attn1.processor.output_share_dict
                dfa_config.reorder_mask[module.attn1.name] = module.attn1.processor.reorder_mask
    torch.cuda.empty_cache()


ASPECT_RATIO_2048_BIN = {
    "0.25": [1024.0, 4096.0],
    "0.26": [1024.0, 3968.0],
    "0.27": [1024.0, 3840.0],
    "0.28": [1024.0, 3712.0],
    "0.32": [1152.0, 3584.0],
    "0.33": [1152.0, 3456.0],
    "0.35": [1152.0, 3328.0],
    "0.4": [1280.0, 3200.0],
    "0.42": [1280.0, 3072.0],
    "0.48": [1408.0, 2944.0],
    "0.5": [1408.0, 2816.0],
    "0.52": [1408.0, 2688.0],
    "0.57": [1536.0, 2688.0],
    "0.6": [1536.0, 2560.0],
    "0.68": [1664.0, 2432.0],
    "0.72": [1664.0, 2304.0],
    "0.78": [1792.0, 2304.0],
    "0.82": [1792.0, 2176.0],
    "0.88": [1920.0, 2176.0],
    "0.94": [1920.0, 2048.0],
    "1.0": [2048.0, 2048.0],
    "1.07": [2048.0, 1920.0],
    "1.13": [2176.0, 1920.0],
    "1.21": [2176.0, 1792.0],
    "1.29": [2304.0, 1792.0],
    "1.38": [2304.0, 1664.0],
    "1.46": [2432.0, 1664.0],
    "1.67": [2560.0, 1536.0],
    "1.75": [2688.0, 1536.0],
    "2.0": [2816.0, 1408.0],
    "2.09": [2944.0, 1408.0],
    "2.4": [3072.0, 1280.0],
    "2.5": [3200.0, 1280.0],
    "2.89": [3328.0, 1152.0],
    "3.0": [3456.0, 1152.0],
    "3.11": [3584.0, 1152.0],
    "3.62": [3712.0, 1024.0],
    "3.75": [3840.0, 1024.0],
    "3.88": [3968.0, 1024.0],
    "4.0": [4096.0, 1024.0],
}


@torch.no_grad()
def inference_fn_plan_update_iop_per_head_wan(
    pipe: WanPipeline,
    dfa_config,
    alpha,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
):
    if num_frames % pipe.vae_scale_factor_temporal != 1:
        num_frames = num_frames // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False

    device = pipe._execution_device

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, wan_block_class):
            # for MMDiT
            if isinstance(module.attn1, attn_class):
                module.attn1.processor.compression_influences = {}
                module.attn1.processor.forward_mode = "calib_collect_info"
                module.attn1.processor.dfa_config = dfa_config
                module.attn1.processor.alpha = alpha
                module.attn1.processor.timestep_block_mask = {}
                module.attn1.processor.prev_calib_output = None
                module.attn1.processor.cached_output = None
                module.attn1.processor.output_share_dict = {}
                module.attn1.processor.num_frames = (num_frames - 1) // 4 + 1
                module.attn1.processor.height = height // 16
                module.attn1.processor.width = width // 16
                # dfa_config.wt[module.attn1.name] = module.attn1.processor.wt
                # dfa_config.output_share_dict[module.attn1.name] = module.attn1.processor.output_share_dict

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue

            pipe._current_timestep = t
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0])

            for name, module in pipe.transformer.named_modules():
                module.timestep_index = i
                if hasattr(module, "stepi"):
                    module.stepi = i
                if isinstance(module, attn_class):
                    module.processor.stepi = i

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            if pipe.do_classifier_free_guidance:
                noise_uncond = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    pipe._current_timestep = None

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, wan_block_class):
            # for MMDiT
            if isinstance(module.attn1, attn_class):
                module.attn1.compression_influences = {}
                module.attn1.processor.cached_output = None
                module.attn1.processor.cached_residual = None
            #     module.attn.processor.cache_residual_forced = False
                module.attn1.processor.dfa_config=None
                module.attn1.processor.forward_mode = "perhead_normal"
                module.attn1.processor.prev_calib_output = None
                dfa_config.wt[module.attn1.name] = module.attn1.processor.wt
                dfa_config.output_share_dict[module.attn1.name] = module.attn1.processor.output_share_dict
                dfa_config.reorder_mask[module.attn1.name] = module.attn1.processor.reorder_mask
    torch.cuda.empty_cache()