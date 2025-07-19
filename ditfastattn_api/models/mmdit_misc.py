import torch
import torch.nn as nn
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
# from diffusers.pipelines.dit.pipeline_dit import DiTPipeline, randn_tensor
from diffusers import StableDiffusion3Pipeline, FluxPipeline, CogVideoXPipeline, PixArtSigmaPipeline, WanPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward, JointTransformerBlock, BasicTransformerBlock
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from ditfastattn_api.modules.dfa_ffn import DiTFastAttnFFN
import numpy as np
import math
from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler

from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_256_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)

# from transformers import (
#     BaseImageProcessor,
#     CLIPTextModelWithProjection,
#     CLIPTokenizer,
#     PreTrainedModel,
#     T5EncoderModel,
#     T5TokenizerFast,
# )

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
# from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
# from diffusers.models.autoencoders import AutoencoderKL
# from diffusers.models.transformers import SD3Transformer2DModel
# from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
# from diffusers.utils import (
#     USE_PEFT_BACKEND,
#     is_torch_xla_available,
#     logging,
#     replace_example_docstring,
#     scale_lora_layers,
#     unscale_lora_layers,
# )
# from diffusers.utils.torch_utils import randn_tensor
# from diffusers.pipelines.pipeline_utils import DiffusionPipeline
# from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

block_class = JointTransformerBlock
basic_block_class = BasicTransformerBlock
flux_block_class = FluxTransformerBlock
flux_single_block_class = FluxSingleTransformerBlock
cogvideox_block_class = CogVideoXBlock
wan_block_class = WanTransformerBlock
attn_class = Attention
ffn_class = FeedForward

def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter(
        (str(o.device), str(o.dtype), tuple(o.shape))
        for o in gc.get_objects()
        if torch.is_tensor(o)
    )
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
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

def inference_fn_with_backward_plan_update_binary(
    pipe: StableDiffusion3Pipeline,
    dfa_config,
    alpha, 
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[torch.Tensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
    skip_layer_guidance_scale: float = 2.8,
    skip_layer_guidance_stop: float = 0.2,
    skip_layer_guidance_start: float = 0.01,
    mu: Optional[float] = None,
):
    

    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    # pipe.check_inputs(
    #     prompt,
    #     prompt_2,
    #     prompt_3,
    #     height,
    #     width,
    #     negative_prompt=negative_prompt,
    #     negative_prompt_2=negative_prompt_2,
    #     negative_prompt_3=negative_prompt_3,
    #     prompt_embeds=prompt_embeds,
    #     negative_prompt_embeds=negative_prompt_embeds,
    #     pooled_prompt_embeds=pooled_prompt_embeds,
    #     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
    #     max_sequence_length=max_sequence_length,
    # )

    # pipe._guidance_scale = guidance_scale
    # pipe._skip_layer_guidance_scale = skip_layer_guidance_scale
    # pipe._clip_skip = clip_skip
    # pipe._joint_attention_kwargs = joint_attention_kwargs
    # pipe._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    lora_scale = (
        pipe.joint_attention_kwargs.get("scale", None) if pipe.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=pipe.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    if pipe.do_classifier_free_guidance:
        if skip_guidance_layers is not None:
            original_prompt_embeds = prompt_embeds
            original_pooled_prompt_embeds = pooled_prompt_embeds
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, height, width = latents.shape
        image_seq_len = (height // pipe.transformer.config.patch_size) * (
            width // pipe.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    if (ip_adapter_image is not None and pipe.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            pipe.do_classifier_free_guidance,
        )

        if pipe.joint_attention_kwargs is None:
            pipe._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        else:
            pipe._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

    # define hook
    # forward hook: cache input, kwarg and output to caculate influence in backward step
    def collect_in_out_forward_hook(m, input, kwargs, output):
        if isinstance(m, DiTFastAttnFFN):
            m.cache_current_output = output.detach()
            m.cache_input = input
            m.cache_kwargs = kwargs
        elif isinstance(m, attn_class):
            m.processor.cached_current_output = output[0].detach()
            m.processor.cached_input = input
            m.processor.cached_kwargs = kwargs

    # backward hook: calculate influence using grad_output
    def collect_influence_backward_hook(m, grad_input, grad_output):
        if m.timestep_index != 0:
            if not hasattr(m,"compression_influences"):
                m.compression_influences = {}
            # breakpoint()
            candidates = dfa_config.get_available_candidates(m.name)
            if isinstance(m, DiTFastAttnFFN):
                output = m.cache_current_output
                input = m.cache_input
                kwargs = m.cache_kwargs
            elif isinstance(m, attn_class):
                output = m.processor.cached_current_output
                input = m.processor.cached_input
                kwargs = m.processor.cached_kwargs
            for candidate in candidates:
                dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
                with torch.no_grad():
                    new_output = m.forward(*input, **kwargs)
                if isinstance(m, DiTFastAttnFFN):
                    influence = ((new_output.float() - output.float()) * grad_output[0].detach()).sum().abs().cpu().numpy()
                elif isinstance(m, attn_class):
                    influence = ((new_output[0].float() - output.float()) * grad_output[0].detach()).sum().abs().cpu().numpy()
                # breakpoint()
                if candidate not in m.compression_influences:
                    m.compression_influences[candidate] = 0
                m.compression_influences[candidate] += influence
                # print(f"m name {m.name}, candidate {candidate}, influence: {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")

    # 7. Denoising loop
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # set hook, set need_cache_output to false
            all_hooks = []
            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for MMDiT
                    if isinstance(module.attn, attn_class):
                        module.attn.processor.compression_influences = {}
                        hook = module.attn.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                        all_hooks.append(hook)
                        hook = module.attn.register_full_backward_hook(collect_influence_backward_hook)
                        all_hooks.append(hook)
                        module.attn.processor.need_cache_output = False
                        module.attn.processor.cache_residual_forced = False
                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.compression_influences = {}
                        hook = module.ff.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                        all_hooks.append(hook)
                        hook = module.ff.register_full_backward_hook(collect_influence_backward_hook)
                        all_hooks.append(hook)
                        module.ff.need_cache_output = False

            if pipe.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            for name, module in pipe.transformer.named_modules():
                module.timestep_index = i
                if hasattr(module, "stepi"):
                    module.stepi = i
                if isinstance(module, attn_class):
                    module.processor.stepi = i

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input.detach(),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds.detach(),
                pooled_projections=pooled_prompt_embeds.detach(),
                joint_attention_kwargs=pipe.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
                should_skip_layers = (
                    True
                    if i > num_inference_steps * skip_layer_guidance_start
                    and i < num_inference_steps * skip_layer_guidance_stop
                    else False
                )
                if skip_guidance_layers is not None and should_skip_layers:
                    timestep = t.expand(latents.shape[0])
                    latent_model_input = latents
                    noise_pred_skip_layers = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep.detach(),
                        encoder_hidden_states=original_prompt_embeds.detach(),
                        pooled_projections=original_pooled_prompt_embeds.detach(),
                        joint_attention_kwargs=pipe.joint_attention_kwargs,
                        return_dict=False,
                        skip_layers=skip_guidance_layers,
                    )[0]
                    noise_pred = (
                        noise_pred + (noise_pred_text - noise_pred_skip_layers) * pipe._skip_layer_guidance_scale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            # latents_dtype = latents.dtype
            # latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # if latents.dtype != latents_dtype:
            #     if torch.backends.mps.is_available():
            #         # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            #         latents = latents.to(latents_dtype)

            # compute previous image: x_t -> x_t-1
            # _latents = pipe.scheduler.step(noise_pred, t.detach(), latents.detach(), return_dict=False)[0]
            # # compute fisher info
            # _latents.mean().backward()
            # # _latent_model_input.mean().backward()
            # pipe.transformer.zero_grad()


            # with torch.set_grad_enabled(True):
            # breakpoint()
            _latents = pipe.scheduler.step(noise_pred, t.detach(), latents.detach(), return_dict=False)[0]
            # compute fisher info and retain graph for later use
            # breakpoint()
            _latents.mean().backward()
            # keep the step unchanged
            pipe.scheduler._step_index -= 1
            pipe.transformer.zero_grad()
            pipe.text_encoder.zero_grad()
            pipe.text_encoder_2.zero_grad()
            pipe.text_encoder_3.zero_grad()

            # remove hooks
            for hook in all_hooks:
                hook.remove()

            # remove all cached input and kwargs
            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for MMDiT
                    if isinstance(module.attn, attn_class):
                        if hasattr(module.attn.processor, "cached_input"):
                            del module.attn.processor.cached_input
                        if hasattr(module.attn.processor, "cached_kwargs"):
                            del module.attn.processor.cached_kwargs
                        if hasattr(module.attn.processor, "cached_current_output"):
                            del module.attn.processor.cached_current_output

                    if isinstance(module.ff, DiTFastAttnFFN):
                        if hasattr(module.ff, "cache_input"):
                            del module.ff.cache_input
                        if hasattr(module.ff, "cache_kwargs"):
                            del module.ff.cache_kwargs
                        if hasattr(module.ff, "cache_current_output"):
                            del module.ff.cache_current_output
            
            # set dfa config after each timestep
            if i != 0:
                # print(dfa_config)
                influences = {}
                # iterate through modules and get compression influence
                for name, module in pipe.transformer.named_modules():
                    module.name = name
                    if isinstance(module, block_class):
                        # for DiT
                        if isinstance(module.attn, attn_class):
                            if module.attn.name not in influences:
                                influences[module.attn.name] = {}
                            for candidate, cached_influence in module.attn.compression_influences.items():
                                if candidate not in influences[module.attn.name]:
                                    influences[module.attn.name][candidate] = 0
                                influences[module.attn.name][candidate] += cached_influence
                            
                        if isinstance(module.ff, DiTFastAttnFFN):
                            if module.ff.name not in influences:
                                influences[module.ff.name] = {}
                            for candidate, cached_influence in module.ff.compression_influences.items():
                                if candidate not in influences[module.ff.name]:
                                    influences[module.ff.name][candidate] = 0
                                influences[module.ff.name][candidate] += cached_influence

                # sort the influence
                sorted_layer_compression_influences = []
                for layer_name, layer_influences in influences.items():
                    for candidate, influence in layer_influences.items():
                        sorted_layer_compression_influences.append((layer_name, candidate, influence))
                sorted_layer_compression_influences.sort(key=lambda x: x[2])
                # print(sorted_layer_compression_influences)
                # conduct binary search
                l_idx = 0
                r_idx = len(sorted_layer_compression_influences)
                m_idx = (r_idx - l_idx) // 2
                tol = 1e-3
                l = 1
                # breakpoint()
                while r_idx - l_idx >= 2:
                    # choose compression method that achieve highest compression ratio
                    # print(dfa_config)
                    # reset method of all layers to raw
                    with torch.no_grad():
                        dfa_config.reset_step_method(i, None)
                        for j in range(m_idx):
                            dfa_config.set_layer_step_method(sorted_layer_compression_influences[j][0], i, sorted_layer_compression_influences[j][1])
                        # dfa_config.display_step_method(timestep_index)
                        # breakpoint()
                        # expand the latents if we are doing classifier free guidance
                        # latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
                        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                        # timestep = t.expand(latent_model_input.shape[0])

                        temp_noise_pred = pipe.transformer(
                            hidden_states=latent_model_input.detach(),
                            timestep=timestep.detach(),
                            encoder_hidden_states=prompt_embeds.detach(),
                            pooled_projections=pooled_prompt_embeds.detach(),
                            joint_attention_kwargs=pipe.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        # perform guidance
                        if pipe.do_classifier_free_guidance:
                            temp_noise_pred_uncond, temp_noise_pred_text = temp_noise_pred.chunk(2)
                            temp_noise_pred = temp_noise_pred_uncond + pipe.guidance_scale * (temp_noise_pred_text - temp_noise_pred_uncond)
                            should_skip_layers = (
                                True
                                if i > num_inference_steps * skip_layer_guidance_start
                                and i < num_inference_steps * skip_layer_guidance_stop
                                else False
                            )
                            if skip_guidance_layers is not None and should_skip_layers:
                                timestep = t.expand(latents.shape[0])
                                latent_model_input = latents
                                temp_noise_pred_skip_layers = pipe.transformer(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=original_prompt_embeds.detach(),
                                    pooled_projections=original_pooled_prompt_embeds.detach(),
                                    joint_attention_kwargs=pipe.joint_attention_kwargs,
                                    return_dict=False,
                                    skip_layers=skip_guidance_layers,
                                )[0]
                                temp_noise_pred = (
                                    temp_noise_pred + (temp_noise_pred_text - temp_noise_pred_skip_layers) * pipe._skip_layer_guidance_scale
                                )

                        # compute the previous noisy sample x_t -> x_t-1
                        # latents_dtype = latents.dtype
                        temp_latents = pipe.scheduler.step(temp_noise_pred, t.detach(), latents.detach(), return_dict=False)[0]
                        pipe.scheduler._step_index -= 1

                    # print(torch.max(noise_pred, temp_noise_pred) + 1e-6)
                    # print(noise_pred.sum())
                    # print(temp_noise_pred.sum())         
                    diff = (_latents - temp_latents) / (torch.max(_latents, temp_latents) + 1e-6)
                    # print(f"diff {diff}")
                    l = diff.abs().clip(0,10).mean()
                    # add up influence
                    influence = 0
                    for name, module in pipe.transformer.named_modules():
                        module.name = name
                        if isinstance(module, block_class):
                            # for DiT
                            if isinstance(module.attn, attn_class):
                                current_method = dfa_config.layers[module.attn.name]["kwargs"]['steps_method'][i]
                                if current_method != "raw":
                                    influence = influence + influences[module.attn.name][current_method]
                            if isinstance(module.ff, DiTFastAttnFFN):
                                current_method = dfa_config.layers[module.ff.name]["kwargs"]['steps_method'][i]
                                if current_method != "raw":
                                    influence = influence + influences[module.ff.name][current_method]
                    print(f"l idx {l_idx}, m idx {m_idx}, r idx {r_idx}, l {l:,.3f}, influence ratio {(influence / latent_model_input.mean().pow(2))}, alpha {alpha:,.3f}")
                    if l <= alpha:
                        l_idx = m_idx
                    elif l > alpha:
                        r_idx = m_idx
                    m_idx = (r_idx + l_idx) // 2

                    if abs(l - alpha) < tol:
                        break

                if l_idx == 0:
                    dfa_config.reset_step_method(i, None)

            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn, attn_class):
                        module.attn.processor.need_cache_output = True
                        module.attn.processor.cached_residual = None
                        if module.attn.processor.steps_method[i] == "raw":
                            module.attn.processor.cache_residual_forced = True
                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.need_cache_output = True
            print(dfa_config)

            # expand the latents if we are doing classifier free guidance
            # latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            # timestep = t.expand(latent_model_input.shape[0])


            noise_pred = pipe.transformer(
                hidden_states=latent_model_input.detach(),
                timestep=timestep.detach(),
                encoder_hidden_states=prompt_embeds.detach(),
                pooled_projections=pooled_prompt_embeds.detach(),
                joint_attention_kwargs=pipe.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
                should_skip_layers = (
                    True
                    if i > num_inference_steps * skip_layer_guidance_start
                    and i < num_inference_steps * skip_layer_guidance_stop
                    else False
                )
                if skip_guidance_layers is not None and should_skip_layers:
                    timestep = t.expand(latents.shape[0])
                    latent_model_input = latents
                    noise_pred_skip_layers = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=original_prompt_embeds.detach(),
                        pooled_projections=original_pooled_prompt_embeds.detach(),
                        joint_attention_kwargs=pipe.joint_attention_kwargs,
                        return_dict=False,
                        skip_layers=skip_guidance_layers,
                    )[0]
                    noise_pred = (
                        noise_pred + (noise_pred_text - noise_pred_skip_layers) * pipe._skip_layer_guidance_scale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            # latents_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, t.detach(), latents.detach(), return_dict=False)[0]
            # breakpoint()

            torch.cuda.empty_cache()
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, block_class):
            # for MMDiT
            if isinstance(module.attn, attn_class):
                module.attn.compression_influences = {}
                module.attn.processor.cached_output = None
                module.attn.processor.cached_residual = None
                module.attn.processor.cache_residual_forced = False
            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.compression_influences = {}
                module.ff.cache_output = None
    torch.cuda.empty_cache()


@torch.no_grad()
def inference_fn_with_backward_plan_update_binary_new_method_per_head(
    pipe: StableDiffusion3Pipeline,
    dfa_config,
    alpha, 
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[torch.Tensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
    skip_layer_guidance_scale: float = 2.8,
    skip_layer_guidance_stop: float = 0.2,
    skip_layer_guidance_start: float = 0.01,
    mu: Optional[float] = None,
):
    

    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    lora_scale = (
        pipe.joint_attention_kwargs.get("scale", None) if pipe.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=pipe.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    if pipe.do_classifier_free_guidance:
        if skip_guidance_layers is not None:
            original_prompt_embeds = prompt_embeds
            original_pooled_prompt_embeds = pooled_prompt_embeds
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, height, width = latents.shape
        image_seq_len = (height // pipe.transformer.config.patch_size) * (
            width // pipe.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    if (ip_adapter_image is not None and pipe.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            pipe.do_classifier_free_guidance,
        )

        if pipe.joint_attention_kwargs is None:
            pipe._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        else:
            pipe._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, block_class):
            # for MMDiT
            if isinstance(module.attn, attn_class):
                module.attn.processor.compression_influences = {}
            #     module.attn.processor.dfa_config = dfa_config
                module.attn.processor.forward_mode = "calib_collect_info"
                module.attn.processor.dfa_config = dfa_config
                module.attn.processor.alpha = alpha
                module.attn.processor.timestep_block_mask = {}
                module.attn.processor.prev_calib_output = None
                module.attn.processor.cached_output = None
                module.attn.processor.output_share_dict = {}
            if hasattr(module, "attn2"):
                if isinstance(module.attn2, attn_class):
                    module.attn2.processor.compression_influences = {}
            #     module.attn.processor.dfa_config = dfa_config
                    module.attn2.processor.forward_mode = "calib_collect_info"
                    module.attn2.processor.dfa_config = dfa_config
                    module.attn2.processor.alpha = alpha
                    module.attn2.processor.timestep_block_mask = {}
                    module.attn2.processor.prev_calib_output = None
                    module.attn2.processor.cached_output = None
                    module.attn2.processor.output_share_dict = {}
            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.compression_influences = {}
                module.ff.mode = "get_calib_info"
                module.ff.cache_output = None
                module.ff.dfa_config = dfa_config
                module.ff.alpha = alpha
            if isinstance(module.ff_context, DiTFastAttnFFN):
                module.ff_context.compression_influences = {}
                module.ff_context.mode = "get_calib_info"
                module.ff_context.cache_output = None
                module.ff_context.dfa_config = dfa_config
                module.ff_context.alpha = alpha


    # 7. Denoising loop
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            print(f"-------------------step {i}--------------------")

            # for name, module in pipe.transformer.named_modules():
            #     module.name = name
            #     if isinstance(module, block_class):
            #         # for MMDiT
            #         if isinstance(module.attn, attn_class):
            #             module.attn.processor.compression_influences = {}
            #         #     module.attn.processor.dfa_config = dfa_config
            #             module.attn.processor.forward_mode = "calib_collect_info"
            #             module.attn.processor.dfa_config = dfa_config
            #             module.attn.processor.alpha = alpha
            
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            for name, module in pipe.transformer.named_modules():
                module.timestep_index = i
                if hasattr(module, "stepi"):
                    module.stepi = i
                if isinstance(module, attn_class):
                    module.processor.stepi = i
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input.detach(),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds.detach(),
                pooled_projections=pooled_prompt_embeds.detach(),
                joint_attention_kwargs=pipe.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # breakpoint()
            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
                should_skip_layers = (
                    True
                    if i > num_inference_steps * skip_layer_guidance_start
                    and i < num_inference_steps * skip_layer_guidance_stop
                    else False
                )
                if skip_guidance_layers is not None and should_skip_layers:
                    timestep = t.expand(latents.shape[0])
                    latent_model_input = latents
                    noise_pred_skip_layers = pipe.transformer(
                        hidden_states=latent_model_input.detach(),
                        timestep=timestep.detach(),
                        encoder_hidden_states=original_prompt_embeds.detach(),
                        pooled_projections=original_pooled_prompt_embeds.detach(),
                        joint_attention_kwargs=pipe.joint_attention_kwargs,
                        return_dict=False,
                        skip_layers=skip_guidance_layers,
                    )[0]
                    noise_pred = (
                        noise_pred + (noise_pred_text - noise_pred_skip_layers) * pipe._skip_layer_guidance_scale
                    )

            latents = pipe.scheduler.step(noise_pred, t.detach(), latents.detach(), return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    # save plan
    # torch.save(dfa_config.plan, f"cache/sd3_{height}_plan_{alpha}.json")

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, block_class):
            # for MMDiT
            if isinstance(module.attn, attn_class):
                module.attn.compression_influences = {}
                module.attn.processor.cached_output = None
                module.attn.processor.cached_residual = None
                # module.attn.processor.block_mask = {}
            #     module.attn.processor.cache_residual_forced = False
                module.attn.processor.dfa_config=None
                module.attn.processor.forward_mode = "perhead_normal"
                module.attn.processor.prev_calib_output = None
                dfa_config.wt[module.attn.name] = module.attn.processor.wt
                dfa_config.output_share_dict[module.attn.name] = module.attn.processor.output_share_dict
            if hasattr(module, "attn2"):
                if isinstance(module.attn2, attn_class):
                    module.attn2.compression_influences = {}
                    module.attn2.processor.cached_output = None
                    module.attn2.processor.cached_residual = None
                #     module.attn2.processor.block_mask = {}
                #     module.attn.processor.cache_residual_forced = False
                    module.attn2.processor.dfa_config=None
                    module.attn2.processor.forward_mode = "perhead_normal"
                    module.attn2.processor.prev_calib_output = None
            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.compression_influences = {}
                module.ff.mode = "normal"
                module.ff.cache_output = None
            if isinstance(module.ff_context, DiTFastAttnFFN):
                module.ff_context.compression_influences = {}
                module.ff_context.mode = "normal"
                module.ff_context.cache_output = None
    torch.cuda.empty_cache()


@torch.no_grad()
def inference_fn_plan_update_iop_per_head_flux(
    pipe: FluxPipeline,
    dfa_config,
    alpha, 
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    true_cfg_scale: float = 1.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_ip_adapter_image: Optional[PipelineImageInput] = None,
    negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 512,
):
    
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    lora_scale = (
        pipe.joint_attention_kwargs.get("scale", None) if pipe.joint_attention_kwargs is not None else None
    )
    do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            _,
        ) = pipe.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    # 4. Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
        negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
    ):
        negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
    elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
        negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
    ):
        ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

    if pipe.joint_attention_kwargs is None:
        pipe._joint_attention_kwargs = {}

    image_embeds = None
    negative_image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )
    if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
        negative_image_embeds = pipe.prepare_ip_adapter_image_embeds(
            negative_ip_adapter_image,
            negative_ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )


    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, flux_block_class) or isinstance(module, flux_single_block_class):
            # for MMDiT
            if isinstance(module.attn, attn_class):
                module.attn.processor.compression_influences = {}
            #     module.attn.processor.dfa_config = dfa_config
                module.attn.processor.forward_mode = "calib_collect_info"
                module.attn.processor.dfa_config = dfa_config
                module.attn.processor.alpha = alpha
                module.attn.processor.timestep_block_mask = {}
                module.attn.processor.prev_calib_output = None
                module.attn.processor.cached_output = None
                module.attn.processor.output_share_dict = {}
                # dfa_config.wt[module.attn.name] = module.attn.processor.wt
                # dfa_config.output_share_dict[module.attn.name] = module.attn.processor.output_share_dict
            if hasattr(module, "ff"):
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.compression_influences = {}
                    module.ff.mode = "get_calib_info"
                    module.ff.cache_output = None
                    module.ff.dfa_config = dfa_config
                    module.ff.alpha = alpha
            if hasattr(module, "ff_context"):
                if isinstance(module.ff_context, DiTFastAttnFFN):
                    module.ff_context.compression_influences = {}
                    module.ff_context.mode = "get_calib_info"
                    module.ff_context.cache_output = None
                    module.ff_context.dfa_config = dfa_config
                    module.ff_context.alpha = alpha


    # 7. Denoising loop
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            print(f"-------------------step {i}--------------------")

            if image_embeds is not None:
                pipe._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
            
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            for name, module in pipe.transformer.named_modules():
                module.timestep_index = i
                if hasattr(module, "stepi"):
                    module.stepi = i
                if isinstance(module, attn_class):
                    module.processor.stepi = i

            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=pipe.joint_attention_kwargs,
                return_dict=False,
            )[0]

            if do_true_cfg:
                if negative_image_embeds is not None:
                    pipe._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                neg_noise_pred = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=negative_pooled_prompt_embeds,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=pipe.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    # save plan
    # torch.save(dfa_config.plan, f"cache/plan_{height}_flux_{alpha}.json")

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, flux_block_class) or isinstance(module, flux_single_block_class):
            # for MMDiT
            if isinstance(module.attn, attn_class):
                module.attn.compression_influences = {}
                module.attn.processor.cached_output = None
                module.attn.processor.cached_residual = None
            #     module.attn.processor.cache_residual_forced = False
                module.attn.processor.dfa_config=None
                module.attn.processor.forward_mode = "perhead_normal"
                module.attn.processor.prev_calib_output = None
                dfa_config.wt[module.attn.name] = module.attn.processor.wt
                dfa_config.output_share_dict[module.attn.name] = module.attn.processor.output_share_dict
            if hasattr(module, "ff"):
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.compression_influences = {}
                    module.ff.mode = "normal"
                    module.ff.cache_output = None
            if hasattr(module, "ff_context"):
                if isinstance(module.ff_context, DiTFastAttnFFN):
                    module.ff_context.compression_influences = {}
                    module.ff_context.mode = "normal"
                    module.ff_context.cache_output = None
    torch.cuda.empty_cache()

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
            timestep = t.expand(latents.shape[0])

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
            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.compression_influences = {}
                module.ff.mode = "normal"
                module.ff.cache_output = None
            if isinstance(module.ff_context, DiTFastAttnFFN):
                module.ff_context.compression_influences = {}
                module.ff_context.mode = "normal"
                module.ff_context.cache_output = None
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
def inference_fn_plan_update_iop_per_head_pixart(
    pipe: PixArtSigmaPipeline,
    dfa_config,
    alpha, 
    prompt: Union[str, List[str]] = None,
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    timesteps: List[int] = None,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    height: Optional[int] = None,
    width: Optional[int] = None,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    clean_caption: bool = True,
    use_resolution_binning: bool = True,
    max_sequence_length: int = 300,
):
    
    height = height or pipe.transformer.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.transformer.config.sample_size * pipe.vae_scale_factor
    if use_resolution_binning:
        if pipe.transformer.config.sample_size == 256:
            aspect_ratio_bin = ASPECT_RATIO_2048_BIN
        elif pipe.transformer.config.sample_size == 128:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        elif pipe.transformer.config.sample_size == 64:
            aspect_ratio_bin = ASPECT_RATIO_512_BIN
        elif pipe.transformer.config.sample_size == 32:
            aspect_ratio_bin = ASPECT_RATIO_256_BIN
        else:
            raise ValueError("Invalid sample size")
        orig_height, orig_width = height, width
        height, width = pipe.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    do_classifier_free_guidance = guidance_scale > 1.0

    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt,
        do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        clean_caption=clean_caption,
        max_sequence_length=max_sequence_length,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)


    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latents.
    latent_channels = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        latent_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Prepare micro-conditions.
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, wan_block_class):
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

    # 7. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, basic_block_class):
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
            if hasattr(module, "ff"):
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.compression_influences = {}
                    module.ff.mode = "get_calib_info"
                    module.ff.cache_output = None
                    module.ff.dfa_config = dfa_config
                    module.ff.alpha = alpha
            if hasattr(module, "ff_context"):
                if isinstance(module.ff_context, DiTFastAttnFFN):
                    module.ff_context.compression_influences = {}
                    module.ff_context.mode = "get_calib_info"
                    module.ff_context.cache_output = None
                    module.ff_context.dfa_config = dfa_config
                    module.ff_context.alpha = alpha


    # 7. Denoising loop
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            print(f"-------------------step {i}--------------------")

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(current_timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            for name, module in pipe.transformer.named_modules():
                module.timestep_index = i
                if hasattr(module, "stepi"):
                    module.stepi = i
                if isinstance(module, attn_class):
                    module.processor.stepi = i

            # predict noise model_output
            noise_pred = pipe.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # learned sigma
            if pipe.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

            # compute previous image: x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]


            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    # save plan
    # torch.save(dfa_config.plan, f"cache/plan_{height}_flux_{alpha}.json")

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, basic_block_class):
            # for MMDiT
            if isinstance(module.attn1, attn_class):
                module.attn1.compression_influences = {}
                module.attn1.processor.cached_output = None
                module.attn1.processor.cached_residual = None
            #     module.attn.processor.cache_residual_forced = False
                module.attn1.processor.dfa_config=None
                module.attn1.processor.forward_mode = "perhead_normal"
                module.attn1.processor.prev_calib_output = None
            if hasattr(module, "ff"):
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.compression_influences = {}
                    module.ff.mode = "normal"
                    module.ff.cache_output = None
            if hasattr(module, "ff_context"):
                if isinstance(module.ff_context, DiTFastAttnFFN):
                    module.ff_context.compression_influences = {}
                    module.ff_context.mode = "normal"
                    module.ff_context.cache_output = None
    torch.cuda.empty_cache()


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
    torch.cuda.empty_cache()