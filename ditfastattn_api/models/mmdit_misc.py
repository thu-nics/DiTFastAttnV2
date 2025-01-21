import torch
import torch.nn as nn
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
# from diffusers.pipelines.dit.pipeline_dit import DiTPipeline, randn_tensor
from diffusers import DiTPipeline, AutoPipelineForText2Image, StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.attention import FeedForward, JointTransformerBlock
from ditfastattn_api.modules.dfa_ffn import DiTFastAttnFFN
from iop_test import solve_ip

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
        
def inference_fn_with_backward_plan_update_binary_two_phase(
    pipe: StableDiffusion3Pipeline,
    dfa_config,
    alpha1, 
    alpha2,
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

    # define hook
    # forward hook: cache input, kwarg and output to caculate influence in backward step
    def collect_in_out_forward_hook(m, input, kwargs, output):
        # output[0].register_hook()
        if isinstance(m, DiTFastAttnFFN):
            m.cache_current_output = output.detach()
            m.cache_input = input
            m.cache_kwargs = kwargs
        elif isinstance(m, attn_class):
            m.processor.cached_current_output = output[0].detach()
            m.processor.cached_input = input
            m.processor.cached_kwargs = kwargs
            

    def collect_influence_backward_hook(m, grad_input, grad_output):
        if m.timestep_index != 0 and ((isinstance(m, DiTFastAttnFFN) and m.steps_method[m.timestep_index] != 'output_share' ) or (isinstance(m, attn_class) and m.processor.steps_method[m.timestep_index] != 'output_share')):
            if not hasattr(m,"compression_influences"):
                m.compression_influences = {}
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
                    new_output = m.forward(*input, **kwargs)[0].detach()
                if isinstance(m, DiTFastAttnFFN):
                    influence = ((new_output.float() - output.float()) * grad_output[0].detach()).sum().abs()
                elif isinstance(m, attn_class):
                    # breakpoint()
                    influence = ((new_output.float() - output.float()) * grad_output[0].detach()).sum().abs()
                # breakpoint()
                if candidate not in m.compression_influences:
                    m.compression_influences[candidate] = 0
                m.compression_influences[candidate] += influence
                # print(f"m name {m.name}, candidate {candidate}, influence: {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")

    # 7. Denoising loop
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):

            # Phase 1: a compression plan with output share only
            print("Phase 1")
            dfa_config.set_available_candidates_global(['output_share'])

            # set hook, set need_cache_output to false
            all_hooks = []
            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for MMDiT
                    if isinstance(module.attn, attn_class):
                        module.attn.compression_influences = {}
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
                # conduct binary search
                l_idx = 0
                r_idx = len(sorted_layer_compression_influences)
                m_idx = (r_idx - l_idx) // 2
                tol = 1e-3
                l = 1
                method_dict = dfa_config.get_step_method(i)
                # breakpoint()
                while r_idx - l_idx >= 2:
                    # choose compression method that achieve highest compression ratio
                    # print(dfa_config)
                    # reset method of all layers to raw
                    with torch.no_grad():
                        dfa_config.reset_step_method(i, None)
                        for j in range(m_idx):
                            dfa_config.set_layer_step_method(sorted_layer_compression_influences[j][0], i, sorted_layer_compression_influences[j][1])

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
                    print(f"l idx {l_idx}, m idx {m_idx}, r idx {r_idx}, l {l:,.3f}, influence ratio {(influence / latent_model_input.mean().pow(2))}, alpha {alpha1:,.3f}")
                    if l <= alpha1:
                        l_idx = m_idx
                    elif l > alpha1:
                        r_idx = m_idx
                    m_idx = (r_idx + l_idx) // 2

                    if abs(l - alpha1) < tol:
                        break

                if l_idx == 0:
                    dfa_config.reset_step_method(i, method_dict)

            
            # print(dfa_config)

            # Phase 2, set window attention
            print("Phase 2")
            dfa_config.set_available_candidates_global(['1residual_window_attn_128'])

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
                    # if isinstance(module.ff, DiTFastAttnFFN):
                    #     module.ff.compression_influences = {}
                    #     hook = module.ff.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    #     all_hooks.append(hook)
                    #     hook = module.ff.register_full_backward_hook(collect_influence_backward_hook)
                    #     all_hooks.append(hook)
                    #     module.ff.need_cache_output = False

            if pipe.interrupt:
                continue

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

                    # if isinstance(module.ff, DiTFastAttnFFN):
                    #     if hasattr(module.ff, "cache_input"):
                    #         del module.ff.cache_input
                    #     if hasattr(module.ff, "cache_kwargs"):
                    #         del module.ff.cache_kwargs
                    #     if hasattr(module.ff, "cache_current_output"):
                    #         del module.ff.cache_current_output

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
                            
                        # if isinstance(module.ff, DiTFastAttnFFN):
                        #     if module.ff.name not in influences:
                        #         influences[module.ff.name] = {}
                        #     for candidate, cached_influence in module.ff.compression_influences.items():
                        #         if candidate not in influences[module.ff.name]:
                        #             influences[module.ff.name][candidate] = 0
                        #         influences[module.ff.name][candidate] += cached_influence

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
                method_dict = dfa_config.get_step_method(i)
                while r_idx - l_idx >= 2:
                    # choose compression method that achieve highest compression ratio
                    # print(dfa_config)
                    # reset method of all layers to raw
                    with torch.no_grad():
                        dfa_config.reset_step_method(i, None)
                        for j in range(m_idx):
                            dfa_config.set_layer_step_method(sorted_layer_compression_influences[j][0], i, sorted_layer_compression_influences[j][1])

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
                            # if isinstance(module.ff, DiTFastAttnFFN):
                            #     current_method = dfa_config.layers[module.ff.name]["kwargs"]['steps_method'][i]
                            #     if current_method != "raw":
                            #         influence = influence + influences[module.ff.name][current_method]
                    print(f"l idx {l_idx}, m idx {m_idx}, r idx {r_idx}, l {l:,.3f}, influence ratio {(influence / latent_model_input.mean().pow(2))}, alpha {alpha2:,.3f}")
                    if l <= alpha2:
                        l_idx = m_idx
                    elif l > alpha2:
                        r_idx = m_idx
                    m_idx = (r_idx + l_idx) // 2

                    if abs(l - alpha2) < tol:
                        break

                if l_idx == 0:
                    dfa_config.reset_step_method(i, method_dict)

            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for MMDiT
                    if isinstance(module.attn, attn_class):
                        module.attn.processor.need_cache_output = True
                        if module.attn.processor.steps_method[i] == "raw":
                            module.attn.processor.cache_residual_forced = True
                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.need_cache_output = True
            print(dfa_config)
            

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


            torch.cuda.empty_cache()
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()


    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, block_class):
            # for MMDiT
            if isinstance(module.attn, attn_class):
                module.attn.processor.compression_influences = {}
                module.attn.processor.cached_output = None
                module.attn.processor.cached_residual = None
                module.attn.processor.cache_residual_forced = False
            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.compression_influences = {}
                module.ff.cache_output = None
    torch.cuda.empty_cache()
        

def inference_fn_with_backward_plan_update_binary_by_head(
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
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
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
                    if candidate not in m.compression_influences:
                        m.compression_influences[candidate] = 0
                    m.compression_influences[candidate] += influence
                elif isinstance(m, attn_class):
                    B, S, hidden_size = grad_output.size()
                    grad = grad_output[0].detach().view(B, S, m.heads, hidden_size // m.heads)
                    
                    for h in range(m.heads):
                        influence = ((new_output[0].view(B, S, m.heads, hidden_size // m.heads)[:,:,h,:].float() \
                                      - output.view(B, S, m.heads, hidden_size // m.heads)[:,:,h,:].float()) \
                                      * grad[:,:,h,:]).sum().abs().cpu().numpy()
                        # breakpoint()
                        if h not in m.compression_influences:
                            m.compression_influences[h] = {}
                        if candidate not in m.compression_influences[h]:
                            m.compression_influences[h][candidate] = 0
                        m.compression_influences[h][candidate] += influence
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

            _latents = pipe.scheduler.step(noise_pred, t.detach(), latents.detach(), return_dict=False)[0]
            # compute fisher info and retain graph for later use
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
                        # for MMDiT
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
                    with torch.no_grad():
                        dfa_config.reset_step_method(i, None)
                        for j in range(m_idx):
                            dfa_config.set_layer_step_method(sorted_layer_compression_influences[j][0], i, sorted_layer_compression_influences[j][1])

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

def inference_fn_with_backward_plan_update_binary_new_method(
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

    # define hook
    # forward hook: cache input, kwarg and output to caculate influence in backward step
    def collect_in_out_forward_hook(m, input, kwargs, output):
        if isinstance(m, DiTFastAttnFFN):
            m.cache_current_output = output.detach()
            m.cache_input = input
            m.cache_kwargs = kwargs
        # elif isinstance(m, attn_class):
        #     m.processor.cached_current_output = output[0].detach()
        #     m.processor.cached_input = input
        #     m.processor.cached_kwargs = kwargs

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
            # elif isinstance(m, attn_class):
            #     output = m.processor.cached_current_output
            #     input = m.processor.cached_input
            #     kwargs = m.processor.cached_kwargs
            for candidate in candidates:
                dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
                with torch.no_grad():
                    new_output = m.forward(*input, **kwargs)
                if isinstance(m, DiTFastAttnFFN):
                    influence = ((new_output.float() - output.float()) * grad_output[0].detach()).sum().abs().cpu().numpy()
                # elif isinstance(m, attn_class):
                #     influence = ((new_output[0].float() - output.float()) * grad_output[0].detach()).sum().abs().cpu().numpy()
                # breakpoint()
                if candidate not in m.compression_influences:
                    m.compression_influences[candidate] = 0
                m.compression_influences[candidate] += influence
                # print(f"m name {m.name}, candidate {candidate}, influence: {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")

    # load latency_dict
    latency_dict = torch.load("cache/latency_dict.json")
    ground_truth = torch.load("ground_truth.pth")

    # 7. Denoising loop
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # set hook, set need_cache_output to false
            all_hooks = []
            print(f"-------------------step {i}--------------------")

            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for MMDiT
                    if isinstance(module.attn, attn_class):
                        module.attn.processor.compression_influences = {}
                    #     module.attn.processor.dfa_config = dfa_config
                        module.attn.processor.forward_mode = "calib_get_grad"
                        module.attn.processor.dfa_config = dfa_config
                    #     module.attn.processor.need_cache_output = False
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

            _latents = pipe.scheduler.step(noise_pred, t.detach(), latents.detach(), return_dict=False)[0]
            # compute fisher info and retain graph for later use
            _latents.mean().backward()
            #loss = torch.mean(((_latents - ground_truth[i].cuda()) ** 2).reshape(_latents.shape[0], -1))
            # loss.backward()

            # keep the step unchanged
            pipe.scheduler._step_index -= 1
            pipe.transformer.zero_grad()
            pipe.text_encoder.zero_grad()
            pipe.text_encoder_2.zero_grad()
            pipe.text_encoder_3.zero_grad()
            # torch.cuda.empty_cache()

            # set dfa config after each timestep
            if i != 0:
                # print(dfa_config)
                influences = {}
                # iterate through modules and get compression influence
                for name, module in pipe.transformer.named_modules():
                    module.name = name
                    if isinstance(module, block_class):
                        # for MMDiT
                        if isinstance(module.attn, attn_class):
                            if module.attn.name not in influences:
                                influences[module.attn.name] = {}
                            for candidate, cached_influence in module.attn.processor.compression_influences.items():
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
                        # if candidate == "output_share":
                        #     influence = influence * 0.5
                        sorted_layer_compression_influences.append((layer_name, candidate, influence))
                sorted_layer_compression_influences.sort(key=lambda x: x[2])
                # print(sorted_layer_compression_influences)
                torch.save(sorted_layer_compression_influences, "cache/influence.json")

                method_list = solve_ip(sorted_layer_compression_influences, latency_dict, alpha)
                for layer_name, method in method_list:
                    dfa_config.set_layer_step_method(layer_name, i, method)

            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn, attn_class):
                        # module.attn.processor.need_cache_output = True
                        # module.attn.processor.cached_residual = None
                        module.attn.processor.forward_mode = "calib_post_inference"
                        module.attn.processor.dfa_config = None
                        # if module.attn.processor.steps_method[i] == "raw":
                        #     module.attn.processor.cache_residual_forced = True
                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.need_cache_output = True
            print(dfa_config)

            with torch.no_grad():
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
                            hidden_states=latent_model_input.detach(),
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
            #     module.attn.processor.cache_residual_forced = False
                module.attn.processor.dfa_config=None
                module.attn.processor.forward_mode = "normal"
            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.compression_influences = {}
                module.ff.cache_output = None
    torch.cuda.empty_cache()

def inference_fn_with_backward_plan_update_binary_iop(
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

    # define hook
    # forward hook: cache input, kwarg and output to caculate influence in backward step
    def collect_in_out_forward_hook(m, input, kwargs, output):
        if isinstance(m, DiTFastAttnFFN):
            m.cache_current_output = output.detach()
            m.cache_input = input
            m.cache_kwargs = kwargs
        # elif isinstance(m, attn_class):
        #     m.processor.cached_current_output = output[0].detach()
        #     m.processor.cached_input = input
        #     m.processor.cached_kwargs = kwargs

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
            # elif isinstance(m, attn_class):
            #     output = m.processor.cached_current_output
            #     input = m.processor.cached_input
            #     kwargs = m.processor.cached_kwargs
            for candidate in candidates:
                dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
                with torch.no_grad():
                    new_output = m.forward(*input, **kwargs)
                if isinstance(m, DiTFastAttnFFN):
                    influence = ((new_output.float() - output.float()) * grad_output[0].detach()).sum().abs().cpu().numpy()
                # elif isinstance(m, attn_class):
                #     influence = ((new_output[0].float() - output.float()) * grad_output[0].detach()).sum().abs().cpu().numpy()
                # breakpoint()
                if candidate not in m.compression_influences:
                    m.compression_influences[candidate] = 0
                m.compression_influences[candidate] += influence
                # print(f"m name {m.name}, candidate {candidate}, influence: {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")

    # load latency_dict
    latency_dict = torch.load("cache/latency_dict.json")

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
                        module.attn.compression_influences = {}
                        module.attn.processor.dfa_config = dfa_config
                        module.attn.processor.calib_mode = "get_grad"
                    #     hook = module.attn.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    #     all_hooks.append(hook)
                    #     hook = module.attn.register_full_backward_hook(collect_influence_backward_hook)
                    #     all_hooks.append(hook)
                        module.attn.processor.need_cache_output = False
                    #     module.attn.processor.cache_residual_forced = False
                    # if isinstance(module.ff, DiTFastAttnFFN):
                    #     module.ff.compression_influences = {}
                    #     hook = module.ff.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    #     all_hooks.append(hook)
                    #     hook = module.ff.register_full_backward_hook(collect_influence_backward_hook)
                    #     all_hooks.append(hook)
                    #     module.ff.need_cache_output = False

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
                        module.attn.processor.dfa_config = None
                        module.attn.processor.calib_mode = "off"
                    #     if hasattr(module.attn.processor, "cached_input"):
                    #         del module.attn.processor.cached_input
                    #     if hasattr(module.attn.processor, "cached_kwargs"):
                    #         del module.attn.processor.cached_kwargs
                    #     if hasattr(module.attn.processor, "cached_current_output"):
                    #         del module.attn.processor.cached_current_output

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
                        # for MMDiT
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
                torch.save(sorted_layer_compression_influences, "cache/influence.json")
                solve_ip(sorted_layer_compression_influences, latency_dict, alpha)
                # # conduct binary search
                # l_idx = 0
                # r_idx = len(sorted_layer_compression_influences)
                # m_idx = (r_idx - l_idx) // 2
                # tol = 1e-3
                # l = 1
                # # breakpoint()
                # while r_idx - l_idx >= 2:
                #     # choose compression method that achieve highest compression ratio
                #     # print(dfa_config)
                #     # reset method of all layers to raw
                #     with torch.no_grad():
                #         dfa_config.reset_step_method(i, None)
                #         for j in range(m_idx):
                #             dfa_config.set_layer_step_method(sorted_layer_compression_influences[j][0], i, sorted_layer_compression_influences[j][1])

                #         temp_noise_pred = pipe.transformer(
                #             hidden_states=latent_model_input.detach(),
                #             timestep=timestep.detach(),
                #             encoder_hidden_states=prompt_embeds.detach(),
                #             pooled_projections=pooled_prompt_embeds.detach(),
                #             joint_attention_kwargs=pipe.joint_attention_kwargs,
                #             return_dict=False,
                #         )[0]
                #         # perform guidance
                #         if pipe.do_classifier_free_guidance:
                #             temp_noise_pred_uncond, temp_noise_pred_text = temp_noise_pred.chunk(2)
                #             temp_noise_pred = temp_noise_pred_uncond + pipe.guidance_scale * (temp_noise_pred_text - temp_noise_pred_uncond)
                #             should_skip_layers = (
                #                 True
                #                 if i > num_inference_steps * skip_layer_guidance_start
                #                 and i < num_inference_steps * skip_layer_guidance_stop
                #                 else False
                #             )
                #             if skip_guidance_layers is not None and should_skip_layers:
                #                 timestep = t.expand(latents.shape[0])
                #                 latent_model_input = latents
                #                 temp_noise_pred_skip_layers = pipe.transformer(
                #                     hidden_states=latent_model_input,
                #                     timestep=timestep,
                #                     encoder_hidden_states=original_prompt_embeds.detach(),
                #                     pooled_projections=original_pooled_prompt_embeds.detach(),
                #                     joint_attention_kwargs=pipe.joint_attention_kwargs,
                #                     return_dict=False,
                #                     skip_layers=skip_guidance_layers,
                #                 )[0]
                #                 temp_noise_pred = (
                #                     temp_noise_pred + (temp_noise_pred_text - temp_noise_pred_skip_layers) * pipe._skip_layer_guidance_scale
                #                 )

                #         # compute the previous noisy sample x_t -> x_t-1
                #         # latents_dtype = latents.dtype
                #         temp_latents = pipe.scheduler.step(temp_noise_pred, t.detach(), latents.detach(), return_dict=False)[0]
                #         pipe.scheduler._step_index -= 1
      
                #     diff = (_latents - temp_latents) / (torch.max(_latents, temp_latents) + 1e-6)
                #     # print(f"diff {diff}")
                #     l = diff.abs().clip(0,10).mean()
                #     # add up influence
                #     influence = 0
                #     for name, module in pipe.transformer.named_modules():
                #         module.name = name
                #         if isinstance(module, block_class):
                #             # for MMDiT
                #             if isinstance(module.attn, attn_class):
                #                 current_method = dfa_config.layers[module.attn.name]["kwargs"]['steps_method'][i]
                #                 if current_method != "raw":
                #                     influence = influence + influences[module.attn.name][current_method]
                #             if isinstance(module.ff, DiTFastAttnFFN):
                #                 current_method = dfa_config.layers[module.ff.name]["kwargs"]['steps_method'][i]
                #                 if current_method != "raw":
                #                     influence = influence + influences[module.ff.name][current_method]
                #     print(f"l idx {l_idx}, m idx {m_idx}, r idx {r_idx}, l {l:,.3f}, influence ratio {(influence / latent_model_input.mean().pow(2))}, alpha {alpha:,.3f}")
                #     if l <= alpha:
                #         l_idx = m_idx
                #     elif l > alpha:
                #         r_idx = m_idx
                #     m_idx = (r_idx + l_idx) // 2

                #     if abs(l - alpha) < tol:
                #         break

                # if l_idx == 0:
                #     dfa_config.reset_step_method(i, None)

            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn, attn_class):
                        module.attn.processor.need_cache_output = True
                        module.attn.processor.cached_residual = None
                        module.attn.processor.calib_mode = "cache_hidden_states"
                        if module.attn.processor.steps_method[i] == "raw":
                            module.attn.processor.cache_residual_forced = True
                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.need_cache_output = True
            print(dfa_config)


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
            #     module.attn.compression_influences = {}
            #     module.attn.processor.cached_output = None
            #     module.attn.processor.cached_residual = None
                module.attn.processor.cache_residual_forced = False
                module.attn.processor.dfa_config=None
                module.attn.processor.calib_mode = "off"
            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.compression_influences = {}
                module.ff.cache_output = None
    torch.cuda.empty_cache()

def inference_save_ground_truth(
    pipe: StableDiffusion3Pipeline,
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
    
    num_inference_steps = num_inference_steps * 2
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

    # create a list to store ground truth
    ground_truth_list = []

    # 7. Denoising loop
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            
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
            if i % 2 == 1:
                ground_truth_list.append(latents.detach().cpu())
            # ground_truth_list.append(latents.detach().cpu())
            torch.cuda.empty_cache()

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
    torch.save(ground_truth_list, f"ground_truth.pth")
    
    
def inference_prompt_prepare_latent(
    pipe: StableDiffusion3Pipeline,
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

    # lora_scale = (
    #     pipe.joint_attention_kwargs.get("scale", None) if pipe.joint_attention_kwargs is not None else None
    # )
    lora_scale=None
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

    return prompt_embeds,pooled_prompt_embeds,timesteps,latents