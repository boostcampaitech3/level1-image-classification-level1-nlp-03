SM_CHANNEL_EVAL=/opt/ml/input/data_edit_nobg/eval \
SM_CHANNEL_MODEL=/opt/ml/level1-image-classification-level1-nlp-03/KSR/train_models/swin-large-384-nobg \
SM_OUTPUT_DATA_DIR=/opt/ml/level1-image-classification-level1-nlp-03/KSR/train_models/swin-large-384-nobg \
python inference.py \
--batch_size 64 \
--model SwinTransformerLarge384