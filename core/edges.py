# core/edges.py
from __future__ import annotations
from typing import Literal, Dict, Tuple
import numpy as np

Array = np.ndarray

def _sobel_abs_sum_inner(gray: Array) -> Array:
    """
    计算 |Gx|+|Gy|，返回 shape=(H-2, W-2) 的内区梯度图（float64）。
    使用标准3x3 Sobel核；不做任何归一化。
    """
    g = gray.astype(np.float64, copy=False)

    # Gx 核
    # [-1 0 1
    #  -2 0 2
    #  -1 0 1]
    gx = (
        (-1)*g[:-2, :-2] + (0)*g[:-2, 1:-1] + (1)*g[:-2, 2:] +
        (-2)*g[1:-1, :-2] + (0)*g[1:-1, 1:-1] + (2)*g[1:-1, 2:] +
        (-1)*g[2:,  :-2] + (0)*g[2:,  1:-1] + (1)*g[2:,  2:]
    )

    # Gy (标准方向，与Equation 2.3一致)
    # [-1 -2 -1
    #   0  0  0
    #  +1 +2 +1]
    gy = (
        (-1)*g[:-2, :-2] + (-2)*g[:-2, 1:-1] + (-1)*g[:-2, 2:] +
        ( 0)*g[1:-1, :-2] + ( 0)*g[1:-1, 1:-1] + ( 0)*g[1:-1, 2:] +
        ( 1)*g[2:, :-2] + ( 2)*g[2:, 1:-1] + ( 1)*g[2:, 2:]
    )

    return np.abs(gx) + np.abs(gy)


def sobel_gradient_map(gray: Array) -> Array:
    """
    将 (H-2)x(W-2) 的内区梯度图“嵌回”到 HxW（边界补0），便于与原图同维度索引。
    输入: gray 为 HxW（uint8 / float）灰度
    返回: grad_full: HxW（float64），边缘一圈为0
    """
    if gray.ndim != 2:
        raise ValueError("sobel_gradient_map: 输入必须是灰度图(HxW)。")
    H, W = gray.shape
    grad_inner = _sobel_abs_sum_inner(gray)
    grad_full = np.zeros((H, W), dtype=np.float64)
    if H >= 3 and W >= 3:
        grad_full[1:-1, 1:-1] = grad_inner
    return grad_full


def choose_adaptive_threshold(grad_full: Array,
                              message_bits: int,
                              start: int = 400,
                              step: int = 10,
                              minimum: int = 0) -> Tuple[int, int]:
    """
    自适应阈值: g = start, start-step, ..., minimum
    选择满足 |R(g)| >= |M| 的“最大” g 作为最终阈值 G。
    若所有 g 都不满足，返回 (minimum, |R(minimum)|)。
    """
    if message_bits < 0:
        raise ValueError("message_bits 不能为负数。")

    # 兜底统计
    G = minimum
    cnt_at_G = int((grad_full >= minimum).sum())

    # 从高到低搜索
    for g in range(start, minimum - 1, -step):
        cnt = int((grad_full >= g).sum())
        if cnt >= message_bits:
            G, cnt_at_G = g, cnt
            break
    return G, cnt_at_G


def select_edge_pixels(gray: Array,
                       message_bits: int,
                       start: int = 400,
                       step: int = 10,
                       minimum: int = 0,
                       return_mode: Literal["flat", "yx"] = "flat") -> Dict[str, object]:
    """
    主入口：Sobel(|Gx|+|Gy|, 未归一化) → 自适应阈值 → 返回入选像素集合 R(G)

    返回字典:
      {
        "threshold": int,         # 选中的 G
        "count": int,             # |R(G)|
        "indices": np.ndarray,    # (N,) 扁平索引 或 (N,2) [y,x]
        "mask": np.ndarray,       # (H,W) 布尔掩码
        "grad": np.ndarray,       # (H,W) 梯度图（float64）
      }
    """
    grad_full = sobel_gradient_map(gray)
    G, cnt = choose_adaptive_threshold(grad_full, message_bits, start, step, minimum)

    mask = (grad_full >= G)
    ys, xs = np.where(mask)
    if return_mode == "flat":
        idx = ys * gray.shape[1] + xs
    else:
        idx = np.stack([ys, xs], axis=1)

    return {
        "threshold": int(G),
        "count": int(cnt),
        "indices": idx,
        "mask": mask,
        "grad": grad_full,
    }
