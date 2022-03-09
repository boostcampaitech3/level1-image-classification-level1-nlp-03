# P Stage - Mask Image Classification
> 카메라로 촬영한 사람 얼굴 이미지를 통해 **연령,성별,마스크 착용 여부**를 판단


![public 1st](https://img.shields.io/badge/Public%20LB-1st-yellow?style=for-the-badge&logo=appveyor) ![private1st](https://img.shields.io/badge/Private%20LB-1st-yellow?style=for-the-badge&logo=appveyor)

<br>

## Team Introduction & Score


<a href="https://maylilyo.notion.site/NLP-3-Wrap-up-report-952b51a771244dab9a52f973a6368e74"><img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png" height=80 width=80px/></a>
**Click logo**

![teamlogo](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F1c92df5d-fffa-4a96-9bef-c09cd8c42ada%2FUntitled.png?table=block&id=5e6b6865-e77a-462a-ac80-272a5dec19c6&spaceId=36a608c4-6cdc-4bd8-a27f-6b7ba453834d&width=2000&userId=81c2b489-fdab-42ca-bfb1-ad6e3e7936e4&cache=v2)

![public rank](https://user-images.githubusercontent.com/46811558/156877191-1a23e91f-865a-4ee2-b3ed-c12111879e47.JPG)![private rank](https://user-images.githubusercontent.com/46811558/156877438-fb6a6ebc-f9b4-4c2e-8893-d466b55f91b4.JPG)


<br>
<br>


### Members
김상렬|김태일|박세연|이재학|정시현|
:-:|:-:|:-:|:-:|:-:
<img src='https://user-images.githubusercontent.com/46811558/157459759-151f2053-92f8-4d00-961a-7cd6511363d2.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/157460657-1caa79d7-7cc9-465f-a3c0-1876e18a3ce6.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/157460405-16cbade1-1430-4283-acbd-44a99857ec5e.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/157460675-9ee90b62-7a39-4542-893d-00eafdb0fd95.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/157460704-6a5ac09f-fe71-4dd3-b30a-f2fa347b08d2.jpg' height=80 width=80px></img>
[Github](https://github.com/SangRyul)|[Github](https://github.com/detailTales)|[Github](https://github.com/maylilyo)|[Github](https://github.com/wogkr810)|[Github](https://github.com/alpha-src)
ksl970330@naver.com|gimty97@gmail.com|maylilyo02@gmail.com|jaehahk810@naver.com|sh2298@naver.com

<br>

### Members' Role
| 팀원 | 역할 | 
| --- | --- |
| 김태일_T3063 | 데이터를 분포를 확인하는 EDA와 불균형한 데이터의 분포를 해결할 수 있는 방법 제시 |
| 박세연_T3091 | 적합한 Model 적용과 AMP를 포함한 Baseline 배포 |
| 이재학_T3161 | f1 loss 제안, 원본 데이터에서 모든 실험 담당 및 실험관리 검토 후 놓친 부분 실험 제안 |
| 김상렬_T3032 | miss labeling 데이터에 대한 정정, background 제거 코드 작성 |
| 정시현_T3198 | EDA를 통한 miss labeling, incorrect path 확인 및 Annotation 코드 작성 |

<br>

### F1 Score Record
![resultpic](https://www.notion.so/image/https%3A%2F%2Fuser-images.githubusercontent.com%2F46811558%2F156929263-ee1ef05a-dd41-4fc1-b3c1-856709091179.JPG?table=block&id=c797260a-788d-4707-a2c5-ea0fb4b24840&spaceId=36a608c4-6cdc-4bd8-a27f-6b7ba453834d&width=2000&userId=81c2b489-fdab-42ca-bfb1-ad6e3e7936e4&cache=v2)


<br>

## Project Introduction

| 분류 | 내용 |
| --- | --- |
| 프로젝트 주제 | 카메라로 촬영한 사람 얼굴 이미지를 통해 연령,성별,마스크 착용 여부를 판단 |
| 프로젝트 개요 | 부스트캠프 Level1-U stage 강의를 통해 배운 내용을 바탕으로, 모델을 설계하고 학습하며 추론을 통해 나온 결과를 바탕으로 순위 산정하는 방식 |
| 활용 장비 및 재료 | • GPU : Tesla V100<br>• 개발환경 : Jupyter Lab , VS code<br>• 협업 tool<br><blockquote>◦Notion : 회의록 정리 , Experiment Page 만들어서 실험 정리 및 공유</blockquote><blockquote>◦Github : 코드 협업</blockquote><blockquote>◦Slack, 카카오톡 : 활발한 의견 공유</blockquote> |
| 프로젝트 구조및 사용 데이터셋의 구조도(연관도)  | • metric : Macro F1 Score<br>• Data :<blockquote>◦ 20~70대의 아시아인 남녀(4500명)<br>◦ 한 사람당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]<br>◦ 이미지 크기 : (384,512)<br>◦ train : 전체 데이터셋의 60%<br>◦ eval : public 20% , private 20%</blockquote> |
| 기대 효과 | 사진 이미지 만으로 사람이 마스크를 올바르게 잘 썼는지 자동으로 가려낼 수 있는 시스템을 구현하고, 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것이다. |
|


<br> 

## Architecture

| 분류 | 내용 |
| --- | --- |
| 아키텍처 | **SwinTransformer-Large(ImageNet-22K pre-trained models)** + **Pytorch Lightning Lite** |
| LB점수`(1/48등)` | • public : 0.8022 , acc : 83.7143<br>• private : 0.7959 , acc : 83.0000 |
| training time augmentation | CustomAugmentation<br>• CenterCrop : 320,256<br>• Resize : 384,384  /  BILinear Interpolation<br>• ColorJitter : (0.1, 0.1, 0.1, 0.1)<br>• ToTensor()<br>• Normalize<br><blockquote>◦ mean : (0.20629331, 0.17723827, 0.16767759)<br>◦ std : (0.34968819, 0.30846628, 0.29714536)</blockqoute> |
| TTA(Test Time Augmentation) | • ColorJitter를 제외하면 train data와 상동<br>• Normalize<br><blockquote>◦ mean : (0.20683326, 0.16344436, 0.15733106)<br>◦ std : (0.35039557, 0.28900916, 0.27917677)<blockquote> |
| 데이터 | • 배경 제거, mislabeling 수정, 샘플링 적용<br>• ageband : (30,60) → (29,57) |
| 검증 전략 |  SOTA 모델의  output.csv 파일을 비교하는 코드 구현하여 비교 |
| 앙상블 방법 | 상위 8개 score를 달성한 모델로 Hard Voting 앙상블 적용 |
| 모델 평가 및 개선  | Vit보다 파라미터 수가 적기에 오버피팅을 방지할 수 있었고, Lightning까지 적용한 후에는 학습시에 많은 시간을 단축 할 수 있었다(1epoch당 20분 → 8분). 또, Lightning를 통해 효율적으로 GPU를 이용할 수 있기에, image size를 384,384로 논문에 맞게 구현한 결과, 좋은 성능을 낼 수 있었다. |


## Getting Started

### `install requirements`

```
pip install -r requirements.txt
```

<br>

### Train

`sh train.sh`

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
--criterion <Criterion> \
--name <Model Name>
```

<br>

### Inference

`sh inference.sh`

```bash
SM_CHANNEL_EVAL=/opt/ml/input/data/eval \
python inference.py \
--model <Model> \
--name <Model Name> \
--batch_size <Batch size>
```
