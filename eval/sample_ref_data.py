import os
import random
import argparse
from shutil import copyfile

def random_sample_images(source_folder, destination_folder, sample_size=5000):
    # Get a list of all image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    # Check if there are enough images to sample
    if len(image_files) < sample_size:
        raise ValueError(f"Not enough images in the folder. Found {len(image_files)} images, but need {sample_size}.")
    
    # Randomly sample the images without replacement
    sampled_images = random.sample(image_files, sample_size)
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Copy the sampled images to the destination folder
    for image in sampled_images:
        src_path = os.path.join(source_folder, image)
        dest_path = os.path.join(destination_folder, image)
        copyfile(src_path, dest_path)
    
    print(f"Successfully sampled and copied {sample_size} images to {destination_folder}.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_folder", type=str, help="Path to the source folder containing images.")
    parser.add_argument("--destination_folder", type=str, help="Path to the destination folder to save sampled images.")
    parser.add_argument("--sample_size", type=int, default=5000, help="Number of images to sample (default: 5000).")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with the provided arguments
    random_sample_images(args.source_folder, args.destination_folder, args.sample_size)

if __name__ == "__main__":
    main()
