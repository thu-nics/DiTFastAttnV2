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

from diffusers import FluxPipeline
import torch
import random
import json
import numpy as np
from torch.nn.attention.flex_attention import flex_attention
from torchvision.transforms import functional as F
import os

from ditfastattn_api.fisher_info_planning import set_flux_compression_plan
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock

from PIL import Image
import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 10000  # Increase the cache size limit

METHOD_CANDIDATES_DICT = {'default': ["output_share", 
                                     "without_residual_window_attn_8", 
                                     "without_residual_window_attn_16",
                                     "without_residual_window_attn_32"],
                          'wa': ["without_residual_window_attn_8", 
                                 "without_residual_window_attn_16",
                                 "without_residual_window_attn_32"],
                          'wa_single': ["without_residual_window_attn_8"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--eval_n_images", type=int, default=4)
    parser.add_argument("--eval_batchsize", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="data/flux_1024_coco_5k")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--n_calib", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--model_name", type=str, default="flash_flux")
    parser.add_argument("--method_set", type=str, default="default")
    parser.add_argument("--cached_kernel_config", type=str, default=None)
    parser.add_argument("--cached_output_share_dict", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--mem_efficient_calibration', default=False, action=argparse.BooleanOptionalAction)


    args = parser.parse_args()
    model_misc = mmdit_misc

    seed = args.seed
    n_steps = 50
    resolution = args.resolution
    calib_x = torch.randint(0, 1000, (1,), generator=torch.Generator().manual_seed(seed)).to("cuda")
    pipe = pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    with open(f"/mnt/public/hanling/mscoco/annotations/captions_val2014.json") as f:
        mscoco_anno = json.load(f)

    caption_list = []
    for i in calib_x:
        caption_list.append(mscoco_anno["annotations"][i]["caption"])


    def flex_attn_block_mask_wrapper(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)
    flex_attn_block_mask_compiled = torch.compile(flex_attn_block_mask_wrapper, dynamic=False)

    def dataloader():
        generator = torch.Generator().manual_seed(seed)
        res = []
        calib_x = torch.randint(0, 1000, (args.n_calib,), generator=generator).to("cuda")
        for i in calib_x:
            res.append(mscoco_anno["annotations"][i]["caption"])
        yield {"prompt":res, "num_inference_steps": n_steps, "generator": generator, "height": args.resolution, "width": args.resolution}

    latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    candidates = METHOD_CANDIDATES_DICT[args.method_set]
    dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps, window_func=flex_attn_block_mask_compiled, candidates=candidates)
    ms = MethodSpeedup()
    ms.vtok_len = (args.resolution // 16) ** 2
    ms.ttok_len = 512
    ms.load_candidates(dfa_config.get_available_candidates(dfa_config.layer_names[0]))
    latency_dict = ms.generate_headwise_latency('estimate', pipe, n_steps, dfa_config, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    print(latency_dict)
    torch.save(latency_dict, f"cache/{args.model_name}_{args.resolution}_latency_dict.json")

    dfa_config.latency = latency_dict
    dfa_config.mem_efficient = args.mem_efficient_calibration

    for step in range(n_steps):
        dfa_config.reset_step_method(step)
    # latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)

    register_refresh_stepi_hook(pipe.transformer, n_steps=n_steps)
    ori_latency = dfa_test_latency(pipe, prompt=caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    unregister_refresh_stepi_hook(pipe.transformer)
    attn_latency, ffn_latency = dfa_test_layer_latency(pipe, n_steps, prompt=caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    with open('latency.txt', 'a') as f:
        f.write(f"model: {args.model_name}, original average attention latency: {attn_latency}, original average ffn latency: {ffn_latency}, original latency: {ori_latency}\n")

    if args.cached_kernel_config is not None and args.cached_output_share_dict is not None:
        print("Use cached compression configurations")
        compression_config = torch.load(args.cached_kernel_config)
        output_share_dict = torch.load(args.cached_output_share_dict)
        for attn_name in compression_config.keys():
            for step in compression_config[attn_name].keys():
                compression_config[attn_name][step] = compression_config[attn_name][step].to('cuda')
        # for attn_name in output_share_dict.keys():
        #     for step in output_share_dict[attn_name].keys():
        #         output_share_dict[attn_name][step] = output_share_dict[attn_name][step].to('cuda')
        
        dfa_config.wt = compression_config
        dfa_config.output_share_dict = output_share_dict

        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module,FluxTransformerBlock) or isinstance(module, FluxSingleTransformerBlock):
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
        # add
        # unregister_refresh_stepi_hook(pipe.transformer)
        print("-------start calibration--------")
        set_flux_compression_plan(pipe, dataloader, dfa_config, model_misc, alpha = args.threshold)
        unregister_refresh_stepi_hook(pipe.transformer)
        # register_refresh_stepi_hook(pipe.transformer, n_steps)
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

    save_path = f"{args.save_path}_{args.threshold}_{args.eval_n_images}"
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
                # image = image.astype(np.uint8)
                image = F.to_pil_image((image * 255).astype(np.uint8))
                image.save(f"{save_path}/{filename_list[count]}.jpg")
                count += 1
    
if __name__ == "__main__":
    main()