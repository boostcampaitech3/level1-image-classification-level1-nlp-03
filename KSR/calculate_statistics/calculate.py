import os
import random
from collections import defaultdict
from enum import Enum
import numpy as np
import torch
from PIL import Image

img_path = []
train_input_nobg_path = '/opt/ml/input/data_edit_nobg/eval/images'
for image in os.listdir(train_input_nobg_path):
    if image.startswith("."):
        continue
    img_path.append(os.path.join(train_input_nobg_path,image))

print("test 길이 : ", len(img_path))
mean = 0
std = 0

print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
sums = []
squared = []
#원래는 끝에가 3000
for image_path in img_path[:]:
    image = np.array(Image.open(image_path)).astype(np.int32)
    sums.append(image.mean(axis=(0, 1)))
    squared.append((image ** 2).mean(axis=(0, 1)))
mean = np.mean(sums, axis=0) / 255
std = (np.mean(squared, axis=0) - mean ** 2) ** 0.5 / 255
print("평균값", mean)
print("표준편차값", std)