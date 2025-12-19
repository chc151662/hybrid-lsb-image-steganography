import numpy as np
from PIL import Image

def calculate_mse(original: np.ndarray, stego: np.ndarray) -> float:
    """
    计算两张图像之间的均方误差（MSE）
    """
    if original.shape != stego.shape:
        raise ValueError("图像尺寸不一致！")
    diff = original.astype("float") - stego.astype("float")
    mse = np.mean(diff ** 2)
    return mse

def calculate_psnr(original_path: str, stego_path: str) -> float:
    """
    计算两张图像之间的 PSNR 值（单位：dB）
    """
    original = np.array(Image.open(original_path))
    stego = np.array(Image.open(stego_path))
    mse = calculate_mse(original, stego)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr
