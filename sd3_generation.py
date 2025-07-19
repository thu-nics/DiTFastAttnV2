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
from diffusers import StableDiffusion3Pipeline
import torch
import random
import json
import numpy as np
# from torch.nn.attention.flex_attention import flex_attention
from torchvision.transforms import functional as F
import os
from ditfastattn_api.fisher_info_planning import update_layer_influence_two_phase
from PIL import Image

from diffusers.models.attention_processor import Attention
from diffusers.models.attention import JointTransformerBlock

METHOD_CANDIDATES_DICT = {'add_cfg': ["output_share", 
                                     "cfg_share", 
                                     "cfg_share_arrow_attn_8", 
                                     "arrow_attn_8", 
                                     "cfg_share_arrow_attn_16",
                                     "arrow_attn_16",
                                     "cfg_share_arrow_attn_32", 
                                     "arrow_attn_32"],
                          'default': ["output_share", 
                                     "arrow_attn_8", 
                                     "arrow_attn_16",
                                     "arrow_attn_32"],
                          'aa_multiple': ["arrow_attn_8", 
                                          "arrow_attn_16",
                                          "arrow_attn_32"],
                          'aa_single': ["arrow_attn_8"]}


def main():
    parser = argparse.ArgumentParser()
    # stabilityai/stable-diffusion-3.5-medium, stabilityai/stable-diffusion-3-medium-diffusers
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--eval_n_images", type=int, default=4)
    parser.add_argument("--eval_batchsize", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="data/flash_sd3_1024_coco_5k")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--n_calib", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--model_name", type=str, default="flash_sd3")
    parser.add_argument("--method_set", type=str, default="default")
    parser.add_argument("--cached_kernel_config", type=str, default=None)
    parser.add_argument("--cached_output_share_dict", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--inference_num_step", type=int, default=50)

    args = parser.parse_args()
    model_misc = mmdit_misc

    seed = args.seed
    n_steps = args.inference_num_step
    resolution = args.resolution
    calib_x = torch.randint(0, 1000, (8,), generator=torch.Generator().manual_seed(seed)).to("cuda")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    with open(f"/mnt/public/hanling/mscoco/annotations/captions_val2014.json") as f:
        mscoco_anno = json.load(f)

    caption_list = []
    for i in calib_x:
        caption_list.append(mscoco_anno["annotations"][i]["caption"])

    def dataloader():
        generator = torch.Generator().manual_seed(seed)
        res = []
        calib_x = torch.randint(0, 1000, (args.n_calib,), generator=generator).to("cuda")
        for i in calib_x:
            res.append(mscoco_anno["annotations"][i]["caption"])
        yield {"prompt": res,"num_inference_steps": n_steps, "generator": generator, "height": args.resolution, "width": args.resolution}
    candidates = METHOD_CANDIDATES_DICT[args.method_set]
    print(candidates)
    dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps, candidates=candidates)
    ms = MethodSpeedup()
    ms.load_candidates(dfa_config.get_available_candidates(dfa_config.layer_names[0]))
    latency_dict = ms.generate_headwise_latency('estimate', pipe, n_steps, dfa_config, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    print(latency_dict)
    torch.save(latency_dict, f"cache/{args.model_name}_{args.resolution}_latency_dict.json")
    dfa_config.latency = latency_dict
    
    register_refresh_stepi_hook(pipe.transformer, n_steps=n_steps)
    ori_latency = dfa_test_latency(pipe, prompt=caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    unregister_refresh_stepi_hook(pipe.transformer)
    attn_latency, ffn_latency = dfa_test_layer_latency(pipe, n_steps, prompt=caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    with open('latency.txt', 'a') as f:
        f.write(f"model: {args.model_name}, original average attention latency: {attn_latency}, original average ffn latency: {ffn_latency}, original latency: {ori_latency}\n")

    # use cached compression configures

    if args.cached_kernel_config is not None and args.cached_output_share_dict is not None:
        print("Use cached compression configurations")
        compression_config = torch.load(args.cached_kernel_config)
        output_share_dict = torch.load(args.cached_output_share_dict)
        for attn_name in compression_config.keys():
            for step in compression_config[attn_name].keys():
                compression_config[attn_name][step] = compression_config[attn_name][step].to('cuda')
        
        dfa_config.wt = compression_config
        dfa_config.output_share_dict = output_share_dict

        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, JointTransformerBlock):
                # for MMDiT
                if isinstance(module.attn, Attention):
                    module.attn.compression_influences = {}
                    module.attn.processor.cached_output = None
                    module.attn.processor.cached_residual = None
                    module.attn.processor.dfa_config=None
                    module.attn.processor.forward_mode = "perhead_normal"
                    module.attn.processor.prev_calib_output = None
                    module.attn.processor.wt = dfa_config.wt[module.attn.name]
                    module.attn.processor.output_share_dict = dfa_config.output_share_dict[module.attn.name]
        
    else:
        print("-------start calibration--------")
        update_layer_influence_two_phase(pipe, dataloader, dfa_config, model_misc, alpha = args.threshold)
        unregister_refresh_stepi_hook(pipe.transformer)
        print("-------end calibration--------")
        if bool(dfa_config.plan):
            torch.save(dfa_config.plan, f"cache/{args.model_name}_{args.resolution}_plan_{args.threshold}.json")
        if bool(dfa_config.ffn_plan):
            torch.save(dfa_config.ffn_plan, f"cache/{args.model_name}_{args.resolution}_plan_{args.threshold}_ffn.json")
        if bool(dfa_config.wt):
            torch.save(dfa_config.wt, f"cache/{args.model_name}_{args.resolution}_head_option_{args.threshold}.json")
        if bool(dfa_config.output_share_dict):
            torch.save(dfa_config.output_share_dict, f"cache/{args.model_name}_{args.resolution}_outputshare_option_{args.threshold}.json")
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

        if os.path.isfile(f"{save_path}/{filename_list[0]}.jpg"):
            continue
        else:
            caption_list = [d["caption"] for d in slice]
            output = pipe(
                caption_list,
                num_inference_steps=n_steps, 
                height=resolution, 
                width=resolution, 
                generator=generator,
                output_type="np"
            )
            fake_images = output.images
            count = 0
            for j, image in enumerate(fake_images):
                image = F.to_pil_image((image * 255).astype(np.uint8))
                image.save(f"{save_path}/{filename_list[count]}.jpg")
                count += 1
    
if __name__ == "__main__":
    main()