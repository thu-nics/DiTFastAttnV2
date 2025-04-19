import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).mul(255).byte().unsqueeze(0)
    return img_tensor.to(device)

def calculate_inception_score(folder):
    inception = InceptionScore().to(device)
    files = sorted(os.listdir(folder))

    if len(files) != 5000:
        raise ValueError(f"wrong number of images, number of images: {len(files)}")
    for file in tqdm(files, total=len(files)):
        path = os.path.join(folder, file)
        img_torch = load_image(path)
        # breakpoint()
        inception.update(img_torch)
    inception_mean, inception_std = inception.compute()

    print("finish calculation process")

    with open('inception_score_output.txt', 'a') as f:
        f.write(f"settings: {folder.split("/")[-1]}, inception_mean: {inception_mean}, inception_std: {inception_std}\n")
    return inception_mean, inception_std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="data/sd3_1024_coco_5k_0.0_5000")

    args = parser.parse_args()

    # 计算平均 SSIM 和 LPIPS
    inception_mean, inception_std = calculate_inception_score(os.path.abspath(args.img_path))
    print(f"inception_mean: {inception_mean}, inception_std: {inception_std}")
if __name__ == "__main__":
    main()
