# CUDA_VISIBLE_DEVICES=7 python run_dit.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0 --eval_n_images 1000 --raw_eval

CUDA_VISIBLE_DEVICES=0 python run_dit.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.05 --eval_n_images 1000 &
CUDA_VISIBLE_DEVICES=1 python run_dit.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.1 --eval_n_images 1000 &
CUDA_VISIBLE_DEVICES=2 python run_dit.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.15 --eval_n_images 1000