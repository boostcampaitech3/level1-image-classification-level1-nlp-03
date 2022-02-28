from rembg.bg import remove
import numpy as np
import io
from PIL import Image
from PIL import ImageFile
import torch

# Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
input_path = './incorrect_mask.jpg'
output_path = './incorret_mask_background.jpg'
ImageFile.LOAD_TRUNCATED_IMAGES = True

f = np.fromfile(input_path)
result = remove(f, only_mask=True, session=None)
torch.cuda.synchronize()
img = Image.open(io.BytesIO(result)).convert("RGB")
img.save(output_path)