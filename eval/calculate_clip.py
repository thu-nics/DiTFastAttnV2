import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).mul(255).byte().unsqueeze(0)
    return img_tensor.to(device)

def calculate_clip_score(folder):
    
    with open("/mnt/public/hanling/mscoco/annotations/captions_val2014.json") as f:
        mscoco_anno = json.load(f)
    hvps_score = 0
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    for i in range(500):
        path = f"{folder}/{str(mscoco_anno['annotations'][i]['id']).zfill(12)}.jpg"
        img_torch = load_image(path)
        caption = mscoco_anno["annotations"][i]['caption'] 
        # print(f"pic {i} | prompt {prompts[i]} | score: {result}")
        clip.update(img_torch, caption)

    clip_result = clip.compute()

    print("finish calculation process")

    with open('clip_score_output.txt', 'a') as f:
        f.write(f"settings: {folder.split("/")[-1]}, clip score: {clip_result}\n")
    return clip_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="data/sd3_1024_coco_5k_0.0_5000")

    args = parser.parse_args()

    clip_result = calculate_clip_score(os.path.abspath(args.img_path))
    print(f"clip score: {clip_result}")

if __name__ == "__main__":
    main()
