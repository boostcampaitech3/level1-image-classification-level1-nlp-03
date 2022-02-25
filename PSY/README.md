<div align="center">

## Mask Detection

</div>

# PyTorch Lightning

## Use AMP(Automatic Mixed Precision) 16bit

- EfficientNet-b0 (batch size: 64): 3115MB -> 2347MB (Save about 24.6% GPU Mem)
- EfficientNet-b7 (batch size: 64): 11825 MB -> 7281 MB (about 38.4%)
- SwinTransformer-Large-384x384 (batch size: 16): 27567 MB -> 23279 MB (about 15.5%)

# Example train setting

### EfficientNet-Base7

```bash
SM_CHANNEL_TRAIN=/opt/ml/input/data/train/images \
SM_MODEL_DIR=/opt/ml/me/result/train \
python train.py \
--epochs 100 \
--lr 1e-4 \
--model EfficientNet \
--optimizer AdamW \
--lr_decay_step 10 \
--name EffNet-b7
```

### SwinTransformer-large-384

```bash
SM_CHANNEL_TRAIN=/opt/ml/input/data/train/images \
SM_MODEL_DIR=/opt/ml/me/result/train \
python train.py \
--epochs 30 \
--lr 2e-04 \
--batch_size 16 \
--model SwinTransformerLarge384 \
--optimizer AdamW \
--lr_decay_step 10 \
--name SwinTransformer-large-384
```

# Pretrained Models

- SwinTransformer-Large-384x384 https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth