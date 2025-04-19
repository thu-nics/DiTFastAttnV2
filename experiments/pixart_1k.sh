CUDA_VISIBLE_DEVICES=0 python pixart_generation.py --threshold 0.0 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=3 python pixart_generation.py --threshold 0.1 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=4 python pixart_generation.py --threshold 0.2 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=5 python pixart_generation.py --threshold 0.3 --eval_n_images 5000 &