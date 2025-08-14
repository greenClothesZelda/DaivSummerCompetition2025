#print img size on "C:\Users\wlstn\.cache\kagglehub\datasets\train\n01443537\n01443537_000.JPEG"

import torch
from PIL import Image

img_path = r"C:\Users\wlstn\.cache\kagglehub\datasets\train\n01443537\n01443537_000.JPEG"
with Image.open(img_path) as img:
    print("Image size:", img.size)  # (width, height)

