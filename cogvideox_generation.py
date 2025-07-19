import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


from ditfastattn_api.api import (
    transform_model_dfa,
    dfa_test_latency,
    register_refresh_stepi_hook,
    unregister_refresh_stepi_hook,
)
from ditfastattn_api.api import MethodSpeedup
import ditfastattn_api.models.video_misc as video_misc
import argparse

import torch
import random
import json
import numpy as np
# from torch.nn.attention.flex_attention import flex_attention
from torchvision.transforms import functional as F
import os

from ditfastattn_api.fisher_info_planning import set_cogvideox_compression_plan
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock

# import torch._dynamo as dynamo
# dynamo.config.cache_size_limit = 10000  # Increase the cache size limit

METHOD_CANDIDATES_DICT = {'default': ["output_share", 
                                   "arrow_attn_8",
                                   "arrow_attn_16", 
                                   "reorder_arrow_attn_8",
                                   "reorder_arrow_attn_16"],
                          'wa': ["arrow_attn_8", 
                                 "arrow_attn_16",
                                 "reorder_arrow_attn_8",
                                 "reorder_arrow_attn_16"],
                          'wa_single': ["arrow_attn_8",
                                        "reorder_arrow_attn_8"]}

def get_prompt_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Remove any trailing newline characters from each line
            lines = [line.strip() for line in lines]
            return lines
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="THUDM/CogVideoX1.5-5B")
    parser.add_argument("--eval_n_images", type=int, default=2)
    parser.add_argument("--eval_batchsize", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="data/cogvideox_vbench")
    parser.add_argument("--prompt_path", type=str, default="vbench_aug_prompt/VBench2_full_text_aug.txt")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--n_calib", type=int, default=6)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=61)
    parser.add_argument("--model_name", type=str, default="flash_cogvideox")
    parser.add_argument("--method_set", type=str, default="default")
    parser.add_argument("--cached_kernel_config", type=str, default=None)
    parser.add_argument("--cached_output_share_dict", type=str, default=None)
    parser.add_argument("--cached_reorder_mask", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--inference_num_step", type=int, default=50)
    parser.add_argument("--config_offload", action='store_true', default=False, help="Whether to enable config CPU offload in calibration")

    args = parser.parse_args()
    model_misc = video_misc

    seed = args.seed
    n_steps = args.inference_num_step
    calib_x = torch.randint(0, 800, (args.n_calib,), generator=torch.Generator().manual_seed(seed)).to("cuda")
    pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    # pipe.enable_sequential_cpu_offload()
    pipe = pipe.to("cuda")

    
    prompt_list = get_prompt_from_file(args.prompt_path)
    caption_list = [prompt_list[i] for i in calib_x]

    video = pipe(
                caption_list,
                num_videos_per_prompt=1,
                num_inference_steps=n_steps, 
                num_frames=args.num_frames,
                height=args.height, 
                width=args.width, 
                guidance_scale=6,
                generator=torch.Generator().manual_seed(seed)
            )


    def dataloader():
        generator = torch.Generator().manual_seed(seed)
        res = []
        calib_x = torch.randint(0, 800, (args.n_calib,), generator=generator).to("cuda")
        for i in calib_x:
            res.append(prompt_list[i])
        yield {"prompt":res, "num_inference_steps": n_steps, "generator": generator, "height": args.height, "width": args.width, "num_frames": args.num_frames, "guidance_scale":6}

    # latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=args.height, width=args.width)
    candidates = METHOD_CANDIDATES_DICT[args.method_set]
    print(candidates)
    dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps, window_func=None, candidates=candidates)
    ms = MethodSpeedup(vtok_len = int((args.height // 16) * (args.width // 16) * ((args.num_frames - 1) // 4 + 1)), ttok_len = 226)
    ms.load_candidates(dfa_config.get_available_candidates(dfa_config.layer_names[0]))
    latency_dict = ms.generate_headwise_latency('estimate', pipe, n_steps, dfa_config, caption_list, num_inference_steps=n_steps, height=args.height, width=args.width, num_frames=41)
    print(latency_dict)
    torch.save(latency_dict, f"cache/{args.model_name}_{args.height}_{args.width}_latency_dict.json")

    dfa_config.latency = latency_dict
    unregister_refresh_stepi_hook(pipe.transformer)

    # for step in range(n_steps):
    #     dfa_config.reset_step_method(step)
    # latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)

    # add

    if args.cached_kernel_config is not None and args.cached_output_share_dict is not None and args.cached_reorder_mask is not None:
        print("Use cached compression configurations")
        compression_config = torch.load(args.cached_kernel_config)
        output_share_dict = torch.load(args.cached_output_share_dict)
        reorder_mask = torch.load(args.cached_reorder_mask)
        for attn_name in compression_config.keys():
            for step in compression_config[attn_name].keys():
                compression_config[attn_name][step] = compression_config[attn_name][step].to('cuda')
        # for attn_name in output_share_dict.keys():
        #     for step in output_share_dict[attn_name].keys():
        #         output_share_dict[attn_name][step] = output_share_dict[attn_name][step].to('cuda')
        
        dfa_config.wt = compression_config
        dfa_config.output_share_dict = output_share_dict
        dfa_config.reorder_mask = reorder_mask

        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, CogVideoXBlock):
                # for MMDiT
                if isinstance(module.attn, Attention):
                    module.attn1.compression_influences = {}
                    module.attn1.processor.cached_output = None
                    module.attn1.processor.cached_residual = None
                    module.attn1.processor.dfa_config=None
                    module.attn1.processor.forward_mode = "perhead_normal"
                    module.attn1.processor.prev_calib_output = None
                    module.attn1.processor.wt = dfa_config.wt[module.attn.name]
                    module.attn1.processor.output_share_dict = dfa_config.output_share_dict[module.attn.name]
                    module.attn1.processor.reorder_mask = dfa_config.reorder_mask[module.attn.name]
    else:
        print("-------start calibration--------")
        set_cogvideox_compression_plan(pipe, dataloader, dfa_config, model_misc, alpha = args.threshold)
        unregister_refresh_stepi_hook(pipe.transformer)
        print("-------end calibration--------")
        if bool(dfa_config.plan):
            if args.method_set != "default":
                torch.save(dfa_config.plan, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_plan_{args.threshold}_{args.method_set}.json")
            else:
                torch.save(dfa_config.plan, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_plan_{args.threshold}.json")
        if bool(dfa_config.ffn_plan):
            if args.method_set != "default":
                torch.save(dfa_config.ffn_plan, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_plan_{args.threshold}_{args.method_set}_ffn.json")
            else:
                torch.save(dfa_config.ffn_plan, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_plan_{args.threshold}_ffn.json")
        if bool(dfa_config.wt):
            if args.method_set != "default":
                torch.save(dfa_config.wt, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_head_option_{args.threshold}_{args.method_set}.json")
            else:
                torch.save(dfa_config.wt, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_head_option_{args.threshold}.json")
        if bool(dfa_config.output_share_dict):
            if args.method_set != "default":
                torch.save(dfa_config.output_share_dict, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_outputshare_option_{args.threshold}_{args.method_set}.json")
            else:
                torch.save(dfa_config.output_share_dict, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_outputshare_option_{args.threshold}.json")
        if bool(dfa_config.reorder_mask):
            if args.method_set != "default":
                torch.save(dfa_config.reorder_mask, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_reorder_mask_{args.threshold}_{args.method_set}.json")
            else:
                torch.save(dfa_config.reorder_mask, f"cache/{args.model_name}_{args.height}_{args.width}_{args.num_frames}_reorder_mask_{args.threshold}.json")

    register_refresh_stepi_hook(pipe.transformer, n_steps)
    # latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    # generate a image
    generator = torch.Generator().manual_seed(seed)

    save_path = f"{args.save_path}_{args.threshold}_{args.eval_n_images}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for index in range(0, args.eval_n_images, args.eval_batchsize):
        caption_list = prompt_list[index: index + args.eval_batchsize]
        filename_list = [str(index + i).zfill(12) for i, d in enumerate(caption_list)]
        filename = index
        print(f"Processing {index}th video")
        if os.path.isfile(f"{save_path}/{filename}.jpg"):
            continue
        else:
            output = pipe(
                caption_list,
                num_videos_per_prompt=1,
                num_inference_steps=n_steps, 
                num_frames=args.num_frames,
                height=args.height, 
                width=args.width, 
                guidance_scale=6,
                generator=generator
            )
            generated_videos = output.frames
            count = 0
            for j, video in enumerate(generated_videos):
                export_to_video(video, f"{save_path}/{filename_list[count]}.mp4", fps=8)
                count += 1
    
if __name__ == "__main__":
    main()