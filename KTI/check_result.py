import random
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

def label_to_info(label_index):
    return "{}, {}, {}".format({0:"wear",1:"incorrect",2:"normal"}[label_index//6],
                                       {0:"0~29",1:"30~59",2:"60~"}[label_index%6%3],
                                       {0:"male",1:"female"}[label_index%6//3])

def show_predictions(csv_path , base_img_path, png_path, sample_size = 16):
    assert sample_size%4 == 0, "샘플 수는 4의 배수로 주세요"
    info_csv = pd.read_csv(csv_path)
    plt.figure(figsize=(12,3*sample_size//4))
    info_csv = info_csv.sample(n = sample_size)
    paths = info_csv["ImageID"]
    labels = info_csv["ans"]
    if sample_size < 16:
        return
    for n, (path, label) in enumerate(zip(paths, labels), start=1):
        plt.subplot(sample_size//4 , 4, n)
        image = Image.open(os.path.join(base_img_path, path))
        plt.imshow(image, cmap='gray')
        plt.title(f"predicted:{label_to_info(label)}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(png_path, "result.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=16, help='number of samples to check (default: 16)')
    parser.add_argument('--csv_path', type=str, default="/opt/ml/baseline/output/output.csv", help='output csv path(default: "/opt/ml/baseline/output/output.csv")')
    parser.add_argument('--base_img_path', type=str, default="/opt/ml/input/data/eval/images", help='eval image path(default: "/opt/ml/baseline/output/output.csv")')
    parser.add_argument('--png_path', type=str, default="/opt/ml/baseline", help='path to save result.png(default: "/opt/ml/baseline")')
    args = parser.parse_args()
    show_predictions(csv_path = args.csv_path, sample_size = args.samples, base_img_path= args.base_img_path, png_path = args.png_path)