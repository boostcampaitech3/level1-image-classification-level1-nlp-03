SM_CHANNEL_TRAIN=/opt/ml/input/data/train/images \
SM_MODEL_DIR=/opt/ml/result/train \
python train.py \
--epochs 30 \
--lr 2e-05 \
--batch_size 16 \
--valid_batch_size 64 \
--model SwinTransformerLarge384 \
--optimizer AdamW \
--lr_decay_step 4 \
--val_ratio 0.1 \
--log_interval 20 \
--name swin-large-384
# SM_CHANNEL_EVAL=/opt/ml/input/data/eval \
# SM_CHANNEL_MODEL=/opt/ml/me/result/train/SwinTransformer-large-38424 \
# SM_OUTPUT_DATA_DIR=/opt/ml/me/result/eval \
# python inference.py \
# --batch_size 64 \
# --model SwinTransformerLarge384