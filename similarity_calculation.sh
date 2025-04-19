CUDA_VISIBLE_DEVICES=4 python eval/calculate_similarity.py --ref_folder data/flux_1024_coco_5k_0.0_5000 --target_folder data/flux_1024_coco_5k_0.05_5000 &
CUDA_VISIBLE_DEVICES=5 python eval/calculate_similarity.py --ref_folder data/flux_1024_coco_5k_0.0_5000 --target_folder data/flux_1024_coco_5k_0.1_5000 &
CUDA_VISIBLE_DEVICES=6 python eval/calculate_similarity.py --ref_folder data/flux_1024_coco_5k_0.0_5000 --target_folder data/flux_1024_coco_5k_0.15_5000 &
CUDA_VISIBLE_DEVICES=7 python eval/calculate_similarity.py --ref_folder data/flux_1024_coco_5k_0.0_5000 --target_folder data/flux_1024_coco_5k_0.2_5000 &
