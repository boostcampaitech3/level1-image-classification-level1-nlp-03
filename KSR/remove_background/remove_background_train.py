from rembg.bg import remove
import numpy as np
import io
from PIL import Image
from PIL import ImageFile
import os
from pathlib import Path

# input_path = './incorrect_mask.jpg'
# output_path = './incorrect_mask_bgremoved.jpg'

# Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# f = np.fromfile(input_path)
# result = remove(f)
# img = Image.open(io.BytesIO(result)).convert("RGB")
# img.save(output_path)

print("train start")
train_input_path = '/opt/ml/input/data_edit/train/images'
train_input_nobg_path = '/opt/ml/input/data_edit_nobg/train/images'
path = Path(train_input_nobg_path)
path.mkdir(parents=True, exist_ok=True)

profiles = os.listdir(train_input_path)
for profile in profiles:
    if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue
    img_folder = os.path.join(train_input_path, profile)

    #background폴더 생성
    os.mkdir(os.path.join(train_input_nobg_path, profile))

    #폴더안에 들어있는 이미지
    for image in os.listdir(img_folder):
        if image.startswith("."):
            continue
        img_path = os.path.join(train_input_path, profile, image)

        #remote background
        f = np.fromfile(img_path)
        result = remove(f)
        img = Image.open(io.BytesIO(result)).convert("RGB")
        
        #make output
        output_img_path = os.path.join(train_input_nobg_path, profile, image)
        img.save(output_img_path)
        

print("train end")
