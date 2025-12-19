import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from PIL import Image
import numpy as np

# tiff格式转换成bmp格式
src_dir = Path("dataset/USC-SIPI/")
dst_dir = Path("dataset/bmp_converted/")
dst_dir.mkdir(exist_ok=True)

for tiff_path in src_dir.glob("*.tiff"):
    img = Image.open(tiff_path).convert("RGB")
    bmp_path = dst_dir / (tiff_path.stem + ".bmp")
    img.save(bmp_path, format="BMP")
    print(f"Converted: {tiff_path.name} -> {bmp_path.name}")


im1 = np.array(Image.open("dataset/USC-SIPI/Milk.tiff"))
im2 = np.array(Image.open("dataset/bmp_converted/Milk.bmp"))
print(np.allclose(im1, im2))  # True 表示像素一致
