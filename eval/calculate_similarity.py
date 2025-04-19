import os
import lpips
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
import argparse
import logging
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = lpips.LPIPS(net='alex').to(device)

def load_image(image_path):
    """加载图像并将其转换为 PyTorch 张量"""
    img = Image.open(image_path).convert('RGB')  # covert to RGB
    transform = transforms.ToTensor()  # convert to tensor
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension
    return img_tensor.to(device)

def calculate_ssim(image1, image2):
    # image1 = img_as_float(image1)
    # image2 = img_as_float(image2)
    # breakpoint()
    return ssim(image1, image2, multichannel=True, channel_axis=-1, data_range=255)

def calculate_lpips(image1, image2):
    with torch.no_grad(): 
        return loss_fn(image1, image2).item()

def calculate_average_metrics(folder1, folder2):
    """calculate average SSIM & LPIPS"""
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))
    
    ssim_values = []
    lpips_values = []
    
    print("start calculation process ...")
    count = 0
    for file1, file2 in tqdm(zip(files1, files2), total=len(files1)):
        if file1 != file2:
            breakpoint()
        path1 = os.path.join(folder1, file1)
        path2 = os.path.join(folder2, file2)
        
        # load image
        img1_skimage = io.imread(path1)
        img2_skimage = io.imread(path2)
        
        img1_torch = load_image(path1)
        img2_torch = load_image(path2)

        # breakpoint()
        
        ssim_value = calculate_ssim(img1_skimage, img2_skimage)
        # if ssim_value > 0.88:
        #     print(file1)
        ssim_values.append(ssim_value)
        
        lpips_value = calculate_lpips(img1_torch, img2_torch)
        lpips_values.append(lpips_value)
        count+=1
        if count >= 5000:
            break

    
    # 计算平均 SSIM 和 LPIPS
    average_ssim = np.mean(ssim_values)
    average_lpips = np.mean(lpips_values)

    print("finish calculation process")


    # 将结果写入日志
    with open('similarity_output.txt', 'a') as f:
        f.write(f"settings: {folder2.split("/")[-1]}, average SSIM: {average_ssim}, average LPIPS: {average_lpips}\n")

    return average_ssim, average_lpips

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_folder", type=str, default="data/sd3_1024_coco_5k_0.0_5000")
    parser.add_argument("--target_folder", type=str, default="data/sd3_1024_coco_5k_0.01_5000_calib16")

    args = parser.parse_args()

    average_ssim, average_lpips = calculate_average_metrics(os.path.abspath(args.ref_folder), os.path.abspath(args.target_folder))
    print(f"Average SSIM: {average_ssim}")
    print(f"Average LPIPS: {average_lpips}")

if __name__ == "__main__":
    main()