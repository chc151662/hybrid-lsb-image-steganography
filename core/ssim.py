# core/ssim.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from PIL import Image

__all__ = ["calculate_ssim", "calculate_ssim_map", "calculate_ssim_from_paths"]

# -------------------------
# 内部工具：高斯核 & 分离卷积
# -------------------------

def _gaussian_kernel_1d(win_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    if win_size % 2 == 0 or win_size < 1:
        raise ValueError("win_size 必须为正奇数。")
    r = win_size // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    return g

def _sep_conv2d(img: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """
    分离卷积（reflect padding），支持 2D(H,W) 或 3D(H,W,C)。
    逐通道独立卷积；输出为 float64。
    """
    r = len(k1d) // 2

    if img.ndim == 2:
        # 水平
        a = np.pad(img, ((0, 0), (r, r)), mode="reflect").astype(np.float64, copy=False)
        out_h = np.zeros_like(img, dtype=np.float64)
        for dx in range(-r, r + 1):
            out_h += k1d[dx + r] * a[:, (r + dx):(r + dx + img.shape[1])]

        # 垂直
        a = np.pad(out_h, ((r, r), (0, 0)), mode="reflect")
        out_v = np.zeros_like(out_h, dtype=np.float64)
        for dy in range(-r, r + 1):
            out_v += k1d[dy + r] * a[(r + dy):(r + dy + img.shape[0]), :]
        return out_v

    if img.ndim == 3 and img.shape[2] == 3:
        H, W, C = img.shape
        res = np.zeros((H, W, C), dtype=np.float64)
        for c in range(C):
            ch = img[..., c]
            # 水平
            a = np.pad(ch, ((0, 0), (r, r)), mode="reflect").astype(np.float64, copy=False)
            out_h = np.zeros_like(ch, dtype=np.float64)
            for dx in range(-r, r + 1):
                out_h += k1d[dx + r] * a[:, (r + dx):(r + dx + W)]
            # 垂直
            a = np.pad(out_h, ((r, r), (0, 0)), mode="reflect")
            out_v = np.zeros_like(out_h, dtype=np.float64)
            for dy in range(-r, r + 1):
                out_v += k1d[dy + r] * a[(r + dy):(r + dy + H), :]
            res[..., c] = out_v
        return res

    raise ValueError("仅支持 2D(H,W) 或 3D(H,W,3) 图像。")

# -------------------------
# SSIM 主体
# -------------------------

def _ssim_single_channel(
    x: np.ndarray, y: np.ndarray,
    data_range: float = 255.0,
    win_size: int = 11, sigma: float = 1.5,
    K1: float = 0.01, K2: float = 0.03,
) -> Tuple[float, np.ndarray]:
    if x.shape != y.shape:
        raise ValueError("SSIM: 输入通道尺寸不一致。")

    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    k = _gaussian_kernel_1d(win_size, sigma)

    mu_x = _sep_conv2d(x, k)
    mu_y = _sep_conv2d(y, k)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _sep_conv2d(x * x, k) - mu_x2
    sigma_y2 = _sep_conv2d(y * y, k) - mu_y2
    sigma_xy = _sep_conv2d(x * y, k) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)

    score = float(ssim_map.mean())
    return score, ssim_map

def _rgb_to_gray_bt601(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float64, copy=False)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def calculate_ssim(
    original: np.ndarray,
    stego: np.ndarray,
    data_range: float = 255.0,
    win_size: int = 11,
    sigma: float = 1.5,
    mode: str = "y"  # "y" | "mean" | "r" | "g" | "b"
) -> float:
    if original.shape != stego.shape:
        raise ValueError("SSIM: 图像尺寸不一致。")

    if original.ndim == 2:  # 灰度
        score, _ = _ssim_single_channel(original, stego, data_range, win_size, sigma)
        return score

    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("SSIM: 仅支持 HxW 或 HxWx3 的图像。")

    o, s = original, stego
    m = mode.lower()
    if m == "y":
        oy = _rgb_to_gray_bt601(o)
        sy = _rgb_to_gray_bt601(s)
        score, _ = _ssim_single_channel(oy, sy, data_range, win_size, sigma)
        return score
    if m == "mean":
        vals = []
        for c in range(3):
            sc, _ = _ssim_single_channel(o[..., c], s[..., c], data_range, win_size, sigma)
            vals.append(sc)
        return float(np.mean(vals))
    if m in ("r", "g", "b"):
        idx = {"r": 0, "g": 1, "b": 2}[m]
        score, _ = _ssim_single_channel(o[..., idx], s[..., idx], data_range, win_size, sigma)
        return score
    raise ValueError("SSIM: 不支持的 mode。可选 'y'、'mean'、'r'、'g'、'b'。")

def calculate_ssim_map(
    original: np.ndarray,
    stego: np.ndarray,
    data_range: float = 255.0,
    win_size: int = 11,
    sigma: float = 1.5,
    mode: str = "y"
) -> Tuple[float, np.ndarray]:
    if original.shape != stego.shape:
        raise ValueError("SSIM: 图像尺寸不一致。")

    if original.ndim == 2:
        return _ssim_single_channel(original, stego, data_range, win_size, sigma)

    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("SSIM: 仅支持 HxW 或 HxWx3 的图像。")

    o, s = original, stego
    m = mode.lower()
    if m == "y":
        oy = _rgb_to_gray_bt601(o)
        sy = _rgb_to_gray_bt601(s)
        return _ssim_single_channel(oy, sy, data_range, win_size, sigma)
    if m == "mean":
        maps, scores = [], []
        for c in range(3):
            sc, sm = _ssim_single_channel(o[..., c], s[..., c], data_range, win_size, sigma)
            scores.append(sc); maps.append(sm)
        mean_map = np.mean(np.stack(maps, axis=0), axis=0)
        return float(np.mean(scores)), mean_map
    if m in ("r", "g", "b"):
        idx = {"r": 0, "g": 1, "b": 2}[m]
        return _ssim_single_channel(o[..., idx], s[..., idx], data_range, win_size, sigma)
    raise ValueError("SSIM: 不支持的 mode。可选 'y'、'mean'、'r'、'g'、'b'。")

def calculate_ssim_from_paths(
    original_path: str,
    stego_path: str,
    mode: str = "y",
    data_range: float = 255.0,
    win_size: int = 11,
    sigma: float = 1.5,
) -> float:
    o = np.array(Image.open(original_path).convert("RGB"))
    s = np.array(Image.open(stego_path).convert("RGB"))
    return calculate_ssim(o, s, data_range=data_range, win_size=win_size, sigma=sigma, mode=mode)
