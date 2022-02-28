# pstage_01_image_classification

## Getting Started

### Idea
`Image Segemtation` 기법을 사용하여 학습과 판별에 불필요한 배경을 제거한다. f1-score와 Accuracy 의 상승에 도움이 돠었다.

### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`

### References
* rembg(https://github.com/danielgatis/rembg)

