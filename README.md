# DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers

# TODO
- [ ] Release Fused Kernel for DiTFastAttnV2
- [ ] 

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

Please see `demo.py` for a quick start.
```
python demo.py
```

