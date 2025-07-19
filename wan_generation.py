from ditfastattn_api.api import (
    transform_model_dfa,
    dfa_test_latency,
    dfa_test_layer_latency,
    register_refresh_stepi_hook,
    unregister_refresh_stepi_hook,
)
from ditfastattn_api.api import MethodSpeedup
import ditfastattn_api.models.video_misc as video_misc

import argparse
import torch
import random
import numpy as np
# from torch.nn.attention.flex_attention import flex_attention
from torchvision.transforms import functional as F
import os
from ditfastattn_api.fisher_info_planning import set_wan_compression_plan
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from diffusers.models.attention_processor import Attention
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from diffusers.utils import export_to_video

METHOD_CANDIDATES_DICT = {'default': ["output_share", 
                                     "arrow_attn_8", 
                                     "arrow_attn_16",
                                     "reorder_arrow_attn_8",
                                     "reorder_arrow_attn_16"],
                          'aa_multiple': ["arrow_attn_8", 
                                          "arrow_attn_16",
                                          "reorder_arrow_attn_8",
                                          "reorder_arrow_attn_16"],
                          'aa_single': ["arrow_attn_8",
                                        "reorder_arrow_attn_8"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--eval_n_images", type=int, default=8)
    parser.add_argument("--eval_batchsize", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="data/wan_t2v_1_3B_480")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--n_calib", type=int, default=8)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--model_name", type=str, default="flash_wan")
    parser.add_argument("--method_set", type=str, default="default")
    parser.add_argument("--cached_kernel_config", type=str, default=None)
    parser.add_argument("--cached_output_share_dict", type=str, default=None)
    parser.add_argument("--cached_reorder_mask", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--inference_num_step", type=int, default=50)

    args = parser.parse_args()
    model_misc = video_misc

    seed = args.seed
    n_steps = args.inference_num_step
    calib_x = torch.randint(0, 1000, (4,), generator=torch.Generator().manual_seed(seed))

    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    with open(f"vbench_aug_prompt/Wanx_full_text_aug.txt", 'r') as f:
        lines = np.array([line.rstrip('\n') for line in f if line.strip()])
        caption_list = lines[calib_x].tolist()

    def dataloader():
        generator = torch.Generator().manual_seed(seed)
        calib_x = torch.randint(0, 1000, (args.n_calib,), generator=generator)
        res = lines[calib_x].tolist()
        yield {"prompt": res, "num_inference_steps": n_steps, "generator": generator, "height": args.height, "width": args.width, "num_frames":args.frames, "guidance_scale": 5.0}
    candidates = METHOD_CANDIDATES_DICT[args.method_set]
    print(candidates)
    dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps, candidates=candidates)

    ms = MethodSpeedup(vtok_len = int((args.height // 16) * (args.width // 16) * ((args.frames - 1) // 4 + 1)), ttok_len = 0)
    ms.load_candidates(dfa_config.get_available_candidates(dfa_config.layer_names[0]))
    latency_dict = ms.generate_headwise_latency('estimate', pipe, n_steps, dfa_config, caption_list, num_inference_steps=n_steps, height=args.height, width=args.width, num_frames=args.frames, guidance_scale=5.0)
    print(latency_dict)
    torch.save(latency_dict, f"cache/{args.model_name}_{args.height}_{args.width}_{args.frames}_latency_dict.json")
    dfa_config.latency = latency_dict

    
    register_refresh_stepi_hook(pipe.transformer, n_steps=n_steps)
    ori_latency = dfa_test_latency(pipe, repeat=1, prompt=caption_list, num_inference_steps=n_steps, height=args.height, width=args.width, num_frames=args.frames, guidance_scale=5.0)
    unregister_refresh_stepi_hook(pipe.transformer)
    attn_latency, ffn_latency = dfa_test_layer_latency(pipe, n_steps, prompt=caption_list, num_inference_steps=n_steps, height=args.height, width=args.width)
    with open('latency.txt', 'a') as f:
        f.write(f"model: {args.model_name}, original average attention latency: {attn_latency}, original average ffn latency: {ffn_latency}, original latency: {ori_latency}\n")

    # use cached compression configures

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
            if isinstance(module, WanTransformerBlock):
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
        set_wan_compression_plan(pipe, dataloader, dfa_config, model_misc, alpha = args.threshold)
        unregister_refresh_stepi_hook(pipe.transformer)
        print("-------end calibration--------")
        if bool(dfa_config.plan):
            torch.save(dfa_config.plan, f"cache/{args.model_name}_{args.height}_{args.width}_{args.frames}_plan_{args.threshold}.json")
        if bool(dfa_config.ffn_plan):
            torch.save(dfa_config.ffn_plan, f"cache/{args.model_name}_{args.height}_{args.width}_{args.frames}_plan_{args.threshold}_ffn.json")
        if bool(dfa_config.wt):
            torch.save(dfa_config.wt, f"cache/{args.model_name}_{args.height}_{args.width}_{args.frames}_head_option_{args.threshold}.json")
        if bool(dfa_config.output_share_dict):
            torch.save(dfa_config.output_share_dict, f"cache/{args.model_name}_{args.height}_{args.width}_{args.frames}_outputshare_option_{args.threshold}.json")
        if bool(dfa_config.reorder_mask):
            torch.save(dfa_config.reorder_mask, f"cache/{args.model_name}_{args.height}_{args.width}_{args.frames}_reorder_mask_{args.threshold}.json")
    register_refresh_stepi_hook(pipe.transformer, n_steps=n_steps)
    latency = dfa_test_latency(pipe, prompt=caption_list, num_inference_steps=n_steps, height=args.height, width=args.width, num_frames=args.frames)
    unregister_refresh_stepi_hook(pipe.transformer)
    attn_latency, ffn_latency = dfa_test_layer_latency(pipe, n_steps, prompt=caption_list, num_inference_steps=n_steps, height=args.height, width=args.width, num_frames=args.frames, guidance_scale=5.0)
    with open('latency.txt', 'a') as f:
        f.write(f"model: {args.model_name}, average attention latency: {attn_latency}, average ffn latency: {ffn_latency}, average latency: {latency}\n")

    # generate a image
    register_refresh_stepi_hook(pipe.transformer, n_steps)
    generator = torch.Generator().manual_seed(seed)

    save_path = f"{args.save_path}_{args.threshold}_{args.eval_n_images}_calib{args.n_calib}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for index in range(0, args.eval_n_images, args.eval_batchsize):
        slice = lines[index : index + args.eval_batchsize].tolist()
        filename_list = [str(index + i).zfill(12) for i, d in enumerate(slice)]
        print(f"Processing {index}th image")

        if os.path.isfile(f"{save_path}/{filename_list[0]}.jpg"):
            continue
        else:
            caption_list = slice
            output = pipe(
                caption_list,
                num_inference_steps=n_steps, 
                height=args.height, 
                width=args.width, 
                num_frames=args.frames,
                generator=generator,
                guidance_scale=5.0,
                output_type="np"
            )
            fake_videos = output.frames
            count = 0
            for j, video in enumerate(fake_videos):
                export_to_video(video, f"{save_path}/{filename_list[count]}.mp4", fps=15)
                count += 1
    
if __name__ == "__main__":
    main()