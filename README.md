# DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers

# TODO
- [ ] release scripts and kernels that support residual sharing
- [ ] extend and support for video generation (CogVideoX, HunyuanVideo, Wan2.1, etc)

# Install

```
conda create -n speedupdit python=3.12
```


```
pip install torch numpy packaging matplotlib scikit-image ninja torchvision
pip install git+https://github.com/huggingface/diffusers
pip install thop pytorch_fid torchmetrics accelerate torchmetrics[image] beautifulsoup4 ftfy transformers SentencePiece
```

# Usage

Please see `sd3_generation.py` and `flux_generation.py` for a quick start.
```
python sd3_generation.py
```

