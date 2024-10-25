CUDA_VISIBLE_DEVICES=0 python run_dit.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.05 --eval_n_images 100
CUDA_VISIBLE_DEVICES=1 python run_dit.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.1 --eval_n_images 100
CUDA_VISIBLE_DEVICES=2 python run_dit.py --n_calib 8 --n_steps 20 --window_size 128 --threshold 0.15 --eval_n_images 100