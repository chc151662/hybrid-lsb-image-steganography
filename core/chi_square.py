# core/chi_square.py
from __future__ import annotations
import numpy as np
from typing import Dict, Literal, Tuple, Optional
from PIL import Image

__all__ = ["chi_square_pov", "chi_square_from_paths"]

Mode = Literal["y", "mean", "r", "g", "b"]

def _rgb_to_gray_bt601(img_rgb: np.ndarray) -> np.ndarray:
    """
    BT.601 灰度: Y = 0.299 R + 0.587 G + 0.114 B
    输入: HxWx3，输出: HxW (float64)
    """
    img = img_rgb.astype(np.float64, copy=False)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def _prepare_channel(img: np.ndarray, mode: Mode = "y") -> np.ndarray:
    """
    将图像转换成单通道 uint8 用于 PoV 直方图统计。
    mode:
      - "y": 先转 BT.601 灰度，再四舍五入到 [0,255] 并转 uint8（推荐）
      - "mean": 对 RGB 三通道分别做统计，最终取三者 chi2 的平均（不常用）
      - "r"/"g"/"b": 指定单通道
    """
    if img.ndim == 2:
        return img.astype(np.uint8, copy=False)

    if img.ndim == 3 and img.shape[2] == 3:
        m = mode.lower()
        if m == "y":
            y = _rgb_to_gray_bt601(img)
            # 灰度值裁剪并量化到 uint8
            y = np.clip(np.rint(y), 0, 255).astype(np.uint8)
            return y
        elif m in ("r", "g", "b"):
            idx = {"r": 0, "g": 1, "b": 2}[m]
            return img[..., idx].astype(np.uint8, copy=False)
        elif m == "mean":
            # 平均三通道的卡方：分别算再取平均
            # 这里返回 None，交由上层特殊处理
            return None
    raise ValueError("chi_square_pov: 仅支持 HxW 或 HxWx3 图像。")

def _chi_square_for_channel(channel_u8: np.ndarray) -> Tuple[float, int, np.ndarray]:
    """
    对单通道 uint8 图像计算 PoV 卡方统计量。
    返回: (chi2, df, hist256)
    """
    flat = channel_u8.reshape(-1)
    hist = np.bincount(flat, minlength=256).astype(np.float64)

    chi2 = 0.0
    df = 0
    for k in range(0, 256, 2):
        o0 = hist[k]
        o1 = hist[k + 1]
        t = o0 + o1
        if t <= 0:
            continue  # 空对，不计入自由度
        e = t / 2.0
        # 对内等频假设：E0 = E1 = t/2
        chi2 += ((o0 - e) ** 2) / (e + 1e-12) + ((o1 - e) ** 2) / (e + 1e-12)
        df += 1
    return float(chi2), int(df), hist

def chi_square_pov(
    original_or_stego: np.ndarray,
    mode: Mode = "y",
) -> Dict[str, Optional[float]]:
    """
    计算基于 PoV 的卡方统计量（Westfeld & Pfitzmann 风格）。
    - 输入: HxW 或 HxWx3 的 numpy 数组（任意 dtype）
    - mode: "y" | "mean" | "r" | "g" | "b"
    返回:
      {
        "chi2": float,            # 总统计量
        "df": int,                # 自由度（非空对数量）
        "p_value": float|None,    # 若安装了 scipy，则给出 1-CDF(chi2, df)，否则为 None
        "hist": np.ndarray(256,)  # 若 mode="mean"，为三通道平均直方图；否则为该通道直方图
      }
    解释：
      - p_value 越大(接近 1)，越支持“对内等频”的假设 → 越像有 LSB 嵌入；
      - p_value 越小(接近 0)，越不支持等频 → 越像自然图像。
    """
    img = original_or_stego
    if img.ndim == 2:
        ch = img.astype(np.uint8, copy=False)
        chi2, df, hist = _chi_square_for_channel(ch)
        p_val = _pvalue_chi2(chi2, df)
        return {"chi2": chi2, "df": df, "p_value": p_val, "hist": hist}

    if img.ndim == 3 and img.shape[2] == 3:
        if mode == "mean":
            res = []
            hists = []
            for c in range(3):
                chi2, df, hist = _chi_square_for_channel(img[..., c].astype(np.uint8, copy=False))
                res.append((chi2, df))
                hists.append(hist)
            chi2_mean = float(np.mean([r[0] for r in res]))
            df_mean   = int(np.round(np.mean([r[1] for r in res])))
            hist_mean = np.mean(np.stack(hists, axis=0), axis=0)
            p_val = _pvalue_chi2(chi2_mean, df_mean)
            return {"chi2": chi2_mean, "df": df_mean, "p_value": p_val, "hist": hist_mean}

        # 其他单通道模式
        ch = _prepare_channel(img, mode=mode)
        if ch is None:
            raise RuntimeError("内部错误：mean 模式已单独处理，理论上不会到这里。")
        chi2, df, hist = _chi_square_for_channel(ch)
        p_val = _pvalue_chi2(chi2, df)
        return {"chi2": chi2, "df": df, "p_value": p_val, "hist": hist}

    raise ValueError("chi_square_pov: 仅支持 HxW 或 HxWx3 图像。")

def _pvalue_chi2(chi2: float, df: int) -> Optional[float]:
    """
    计算卡方分布的右尾 P 值（survival function）。
    若系统安装了 scipy，则使用 scipy；否则返回 None。
    """
    if df <= 0:
        return None
    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore
        return float(chi2_dist.sf(chi2, df))
    except Exception:
        return None

def chi_square_from_paths(
    image_path: str,
    mode: Mode = "y",
) -> Dict[str, Optional[float]]:
    """
    从路径直接计算 PoV 卡方指标（默认在 BT.601 灰度上）。
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    return chi_square_pov(img, mode=mode)
