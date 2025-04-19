CUDA_VISIBLE_DEVICES=4 python flux_generation_flash.py --resolution 2048 --save_path data/flash_flux_2048_coco_5k --threshold 0.2 --eval_n_images 4 --n_calib 8 &
CUDA_VISIBLE_DEVICES=5 python flux_generation_flash.py --resolution 2048 --save_path data/flash_flux_2048_coco_5k --threshold 0.4 --eval_n_images 4 --n_calib 8 &
CUDA_VISIBLE_DEVICES=6 python flux_generation_flash.py --resolution 2048 --save_path data/flash_flux_2048_coco_5k --threshold 0.6 --eval_n_images 4 --n_calib 8 &
CUDA_VISIBLE_DEVICES=7 python flux_generation_flash.py --resolution 2048 --save_path data/flash_flux_2048_coco_5k --threshold 0.8 --eval_n_images 4 --n_calib 8 &
# CUDA_VISIBLE_DEVICES=7 python flux_generation.py --resolution 2048 --save_path data/flux_2048_coco_5k --threshold 1.0 --eval_n_images 4 --n_calib 8 &