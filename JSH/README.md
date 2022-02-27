# P-Stage Image Classification - Mask
> baseline by sihyun jung<br>
> https://github.com/alpha-src

<br>

## Getting Started

### `install requirements`

```
pip install -r requirements.txt
```

<br>

### Train

```bash
SM_CHANNEL_TRAIN=/opt/ml/input/data/train/images \
python train.py \
--epochs <Epochs> \
--lr 2e-05 \
--batch_size <Batch size> \
--valid_batch_size <Valid set Batch size> \
--model <Model> \
--optimizer <Optimizer> \
--lr_decay_step 4 \
--val_ratio <Val ratio> \
--log_interval 20 \
--early_stop <Early Stop> \
--criterion <Criterion> \
--dropout_rate <Dropout Rate> \
--name <Model Name>
```

<br>

### Inference

```bash
SM_CHANNEL_EVAL=/opt/ml/input/data/eval \
python inference.py \
--model <Model> \
--name <Model Name> \
--batch_size <Batch size>
```

<br>

## Models Available(pretrained)
* ViT_L_16_imagenet1k
* ViT_B_16_imagenet1k
* EfficientNet b7
* SwinTransformer

<br>

## Loss
* F1Loss
* FocalLoss
* LabelSmoothingLoss

## Augmentation
**[AutoAugment - DeepVoltaire](https://github.com/DeepVoltaire/AutoAugment)**


**Resize** - 224,224

