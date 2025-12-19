# core/ncc.py
from __future__ import annotations
import numpy as np
from typing import Literal
from PIL import Image

__all__ = ["calculate_ncc", "calculate_ncc_from_paths"]

Mode = Literal["y", "mean", "r", "g", "b"]

def _rgb_to_gray_bt601(img_rgb: np.ndarray) -> np.ndarray:
    """
    BT.601 灰度: Y = 0.299 R + 0.587 G + 0.114 B
    输入: HxWx3, 任意数值型；输出: HxW, float64
    """
    img = img_rgb.astype(np.float64, copy=False)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def _ncc_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算一维向量的 ZNCC（去均值归一化互相关）。
    a, b: 1D float64
    返回: 标量 ∈ [-1, 1]
    """
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)

    a_mean = a.mean()
    b_mean = b.mean()
    a_z = a - a_mean
    b_z = b - b_mean

    num = np.dot(a_z, b_z)
    denom = np.sqrt(np.dot(a_z, a_z) * np.dot(b_z, b_z))

    if denom == 0.0:
        # 若两者全为常量：完全相同 → 1.0；否则无法定义，给出 0.0
        return 1.0 if np.allclose(a, b) else 0.0
    return float(num / denom)

def calculate_ncc(
    original: np.ndarray,
    stego: np.ndarray,
    mode: Mode = "y",
) -> float:
    """
    计算两张图像之间的 NCC（去均值的归一化互相关，ZNCC）。
    - original, stego: HxW 或 HxWx3 的 numpy 数组，dtype 任意（内部转 float64）
    - mode:
        "y"    : RGB→BT.601 灰度后计算（推荐，口径与 PSNR/SSIM 一致）
        "mean" : 对 R/G/B 三通道分别计算 NCC 再取平均
        "r"/"g"/"b" : 只在对应通道计算 NCC
    返回：标量 NCC，取值范围 [-1, 1]，越接近 1 越相似。
    """
    if original.shape != stego.shape:
        raise ValueError("NCC: 图像尺寸不一致。")

    if original.ndim == 2:  # 灰度
        a = original.astype(np.float64, copy=False).ravel()
        b = stego.astype(np.float64, copy=False).ravel()
        return _ncc_1d(a, b)

    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("NCC: 仅支持 HxW 或 HxWx3 的图像。")

    o = original
    s = stego
    mode = mode.lower()

    if mode == "y":
        oy = _rgb_to_gray_bt601(o).ravel()
        sy = _rgb_to_gray_bt601(s).ravel()
        return _ncc_1d(oy, sy)

    if mode == "mean":
        vals = []
        for c in range(3):
            vals.append(_ncc_1d(o[..., c].ravel(), s[..., c].ravel()))
        return float(np.mean(vals))

    if mode in ("r", "g", "b"):
        idx = {"r": 0, "g": 1, "b": 2}[mode]
        return _ncc_1d(o[..., idx].ravel(), s[..., idx].ravel())

    raise ValueError("NCC: 不支持的 mode。可选 'y'、'mean'、'r'、'g'、'b'。")

def calculate_ncc_from_paths(
    original_path: str,
    stego_path: str,
    mode: Mode = "y",
) -> float:
    """
    从文件路径直接计算 NCC（默认在 BT.601 灰度上）。
    """
    o = np.array(Image.open(original_path).convert("RGB"))
    s = np.array(Image.open(stego_path).convert("RGB"))
    return calculate_ncc(o, s, mode=mode)
