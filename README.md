# OFAFastDiT: Once For All Fast DiT Inference Acceleration

# TODO
- [ ] config for layer replace mapping to support different model

# Install

```
conda create -n speedupdit python=3.12
```


```
pip install torch numpy packaging matplotlib scikit-image ninja torchvision
pip install git+https://github.com/huggingface/diffusers
pip install thop pytorch_fid torchmetrics accelerate torchmetrics[image] beautifulsoup4 ftfy flash-attn transformers SentencePiece
```

# Prepare dataset

Sample real images to `data/real_images` from ImageNet to compute the IS and FID:
```
python data/sample_real_images.py <imagenet_path>
```

If you will use Pixart, place coco dataset to `data/mscoco`.

# Usage
All the experiment code can be found in folder `experiments/`.

DiT compression:
```
python run_dit.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.05 --eval_n_images 5000
```

PixArt 1k compression:
```
python run_pixart.py --n_calib 6 --n_steps 50 --window_size 512 --threshold 0.0725 --eval_n_images 5000

```
