from ditfastattn_api.api import (
    transform_model_dfa,
    dfa_test_latency,
    dfa_test_layer_latency,
    register_refresh_stepi_hook,
    unregister_refresh_stepi_hook,
)
from ditfastattn_api.api import MethodSpeedup
import ditfastattn_api.models.mmdit_misc as mmdit_misc
import argparse

from diffusers import DiTPipeline
import torch
import random
import json
import numpy as np
from torch.nn.attention.flex_attention import flex_attention
from torchvision.transforms import functional as F
import os

from ditfastattn_api.fisher_info_planning import update_layer_influence_two_phase

from PIL import Image

import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 10000  # Increase the cache size limit

METHOD_CANDIDATES_DICT = {'full': ["output_share", 
                                   "cfg_share", 
                                   "cfg_share_without_residual_window_attn_8", 
                                   "without_residual_window_attn_8", 
                                   "cfg_share_without_residual_window_attn_16",
                                   "without_residual_window_attn_16",
                                   "cfg_share_without_residual_window_attn_32", 
                                   "without_residual_window_attn_32"],
                          'wa_ast': ["output_share", 
                                     "without_residual_window_attn_8", 
                                     "without_residual_window_attn_16",
                                     "without_residual_window_attn_32"],
                          'wa': ["without_residual_window_attn_8", 
                                 "without_residual_window_attn_16",
                                 "without_residual_window_attn_32"],
                          'wa_single': ["without_residual_window_attn_8"]}


def main():
    parser = argparse.ArgumentParser()
    # stabilityai/stable-diffusion-3.5-medium, stabilityai/stable-diffusion-3-medium-diffusers
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-512")
    parser.add_argument("--eval_n_images", type=int, default=4)
    parser.add_argument("--eval_batchsize", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="data/dit_512_imagenet_5k")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--n_calib", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--model_name", type=str, default="sd3_5")
    parser.add_argument("--method_set", type=str, default="full")

    args = parser.parse_args()
    model_misc = mmdit_misc

    seed = 3
    n_steps = 50
    resolution = args.resolution
    calib_x = torch.randint(0, 1000, (4,), generator=torch.Generator().manual_seed(seed)).to("cuda")
    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")


    def flex_attn_block_mask_wrapper(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)
    flex_attn_block_mask_compiled = torch.compile(flex_attn_block_mask_wrapper, dynamic=False)

    def dataloader():
        generator = torch.Generator().manual_seed(seed)
        calib_x = torch.randint(0, 1000, (args.n_calib,), generator=generator).to("cuda")
        yield {"class_labels": calib_x ,"num_inference_steps": n_steps, "generator": generator}

    # latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    candidates = METHOD_CANDIDATES_DICT[args.method_set]
    print(candidates)
    dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps, window_func=flex_attn_block_mask_compiled, candidates=candidates)
    ms = MethodSpeedup()
    ms.load_candidates(dfa_config.get_available_candidates(dfa_config.layer_names[0]))
    latency_dict = ms.generate_headwise_latency('estimate', pipe, n_steps, dfa_config, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    print(latency_dict)
    torch.save(latency_dict, f"cache/{args.model_name}_{args.resolution}_latency_dict.json")

    dfa_config.latency = latency_dict

    # for step in range(n_steps):
    #     dfa_config.reset_step_method(step)

    
    register_refresh_stepi_hook(pipe.transformer, n_steps=n_steps)
    ori_latency = dfa_test_latency(pipe, prompt=caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    unregister_refresh_stepi_hook(pipe.transformer)
    attn_latency, ffn_latency = dfa_test_layer_latency(pipe, n_steps, prompt=caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    with open('latency.txt', 'a') as f:
        f.write(f"model: {args.model_name}, original average attention latency: {attn_latency}, original average ffn latency: {ffn_latency}, original latency: {ori_latency}\n")

    # add
    # unregister_refresh_stepi_hook(pipe.transformer)
    print("-------start calibration--------")
    update_layer_influence_two_phase(pipe, dataloader, dfa_config, model_misc, alpha = args.threshold)
    unregister_refresh_stepi_hook(pipe.transformer)
    print("-------end calibration--------")

    torch.save(dfa_config.plan, f"cache/{args.model_name}_{args.resolution}_plan_{args.threshold}.json")
    torch.save(dfa_config.ffn_plan, f"cache/{args.model_name}_{args.resolution}_plan_{args.threshold}_ffn.json")
    register_refresh_stepi_hook(pipe.transformer, n_steps=n_steps)
    latency = dfa_test_latency(pipe, prompt=caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    unregister_refresh_stepi_hook(pipe.transformer)
    attn_latency, ffn_latency = dfa_test_layer_latency(pipe, n_steps, prompt=caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    with open('latency.txt', 'a') as f:
        f.write(f"model: {args.model_name}, average attention latency: {attn_latency}, average ffn latency: {ffn_latency}, average latency: {latency}\n")


    # latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    # generate a image
    register_refresh_stepi_hook(pipe.transformer, n_steps)
    generator = torch.Generator().manual_seed(seed)

    save_path = f"{args.save_path}_{args.threshold}_{args.eval_n_images}_calib{args.n_calib}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for index in range(0, args.eval_n_images, args.eval_batchsize):
        slice = mscoco_anno["annotations"][index : index + args.eval_batchsize]
        filename_list = [str(d["id"]).zfill(12) for d in slice]
        print(f"Processing {index}th image")
        caption_list = [d["caption"] for d in slice]
        output = pipe(
            caption_list,
            num_inference_steps=n_steps, 
            generator=generator,
            output_type="np"
        )
        fake_images = output.images
        count = 0
        for j, image in enumerate(fake_images):
            # image = image.astype(np.uint8)
            image = F.to_pil_image((image * 255).astype(np.uint8))
            image.save(f"{save_path}/{filename_list[count]}.jpg")
            count += 1
    
if __name__ == "__main__":
    main()