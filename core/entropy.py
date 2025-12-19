# core/entropy.py
from __future__ import annotations
import numpy as np
from typing import Literal, Tuple, Dict, Optional
from PIL import Image

__all__ = [
    "calculate_entropy",
    "calculate_entropy_from_paths",
    "compare_entropy",
    "compare_entropy_from_paths",
]

Mode = Literal["y", "mean", "r", "g", "b"]

def _rgb_to_gray_bt601(img_rgb: np.ndarray) -> np.ndarray:
    """
    BT.601 灰度: Y = 0.299 R + 0.587 G + 0.114 B
    输入: HxWx3, 任意 dtype；输出: HxW, float64（未量化）
    """
    img = img_rgb.astype(np.float64, copy=False)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def _entropy_from_u8(channel_u8: np.ndarray) -> float:
    """
    对单通道 uint8 图像计算 Shannon entropy（bits per pixel）。
    """
    flat = channel_u8.reshape(-1)
    hist = np.bincount(flat, minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    # 仅对 p>0 的项求和，避免 log2(0)
    m = p > 0
    H = -np.sum(p[m] * np.log2(p[m]))
    return float(H)

def calculate_entropy(img: np.ndarray, mode: Mode = "y") -> float:
    """
    计算图像的信息熵（bits per pixel）。
    - img: HxW 或 HxWx3 的 numpy 数组（任意数值型）
    - mode:
        "y"    : RGB → BT.601 灰度后量化到 uint8 再计算（推荐）
        "mean" : 对 R/G/B 三通道分别计算熵后取平均
        "r"/"g"/"b": 仅在对应通道计算
    返回：H（越接近 8 表示分布越均匀）
    """
    if img.ndim == 2:  # 已是灰度
        ch = img.astype(np.uint8, copy=False)
        return _entropy_from_u8(ch)

    if img.ndim == 3 and img.shape[2] == 3:
        mode = mode.lower()
        if mode == "y":
            y = _rgb_to_gray_bt601(img)
            y = np.clip(np.rint(y), 0, 255).astype(np.uint8)
            return _entropy_from_u8(y)
        if mode == "mean":
            vals = []
            for c in range(3):
                vals.append(_entropy_from_u8(img[..., c].astype(np.uint8, copy=False)))
            return float(np.mean(vals))
        if mode in ("r", "g", "b"):
            idx = {"r": 0, "g": 1, "b": 2}[mode]
            return _entropy_from_u8(img[..., idx].astype(np.uint8, copy=False))

    raise ValueError("calculate_entropy: 仅支持 HxW 或 HxWx3 图像。")

def calculate_entropy_from_paths(path: str, mode: Mode = "y") -> float:
    """
    从文件路径计算信息熵（默认在 BT.601 灰度上）。
    """
    img = np.array(Image.open(path).convert("RGB"))
    return calculate_entropy(img, mode=mode)

def compare_entropy(orig: np.ndarray, stego: np.ndarray, mode: Mode = "y") -> Dict[str, float]:
    """
    对比原图与嵌入图的熵，返回 {H_orig, H_stego, delta, rel_delta}。
    - delta = H_stego - H_orig
    - rel_delta = delta / max(H_orig, 1e-12)
    """
    if orig.shape != stego.shape:
        raise ValueError("compare_entropy: 图像尺寸不一致。")
    H_o = calculate_entropy(orig, mode=mode)
    H_s = calculate_entropy(stego, mode=mode)
    delta = H_s - H_o
    rel = delta / max(H_o, 1e-12)
    return {"H_orig": H_o, "H_stego": H_s, "delta": delta, "rel_delta": rel}

def compare_entropy_from_paths(orig_path: str, stego_path: str, mode: Mode = "y") -> Dict[str, float]:
    """
    从路径对比原图与嵌入图的熵（默认在 BT.601 灰度上）。
    """
    o = np.array(Image.open(orig_path).convert("RGB"))
    s = np.array(Image.open(stego_path).convert("RGB"))
    return compare_entropy(o, s, mode=mode)
