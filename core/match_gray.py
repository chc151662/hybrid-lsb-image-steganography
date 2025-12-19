# core/match_gray.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from PIL import Image

from core.edges import sobel_gradient_map   # 你已有
# 灰度图像直接使用 Image.convert("L") 得到 0..255 的二维数组

def raster_indices_from_mask(mask: np.ndarray, select_false: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """从二值 mask 取行扫描索引。select_false=True 表示取 ~mask 区域(~R)。"""
    target = ~mask if select_false else mask
    ys, xs = np.nonzero(target)
    return ys.astype(np.int32), xs.astype(np.int32)

def read_bit_gray(img_gray: np.ndarray, y: int, x: int, bit_plane: int = 0) -> int:
    """读灰度像素 (y,x) 在指定 bit plane 的位（0 为 LSB）。"""
    return (int(img_gray[y, x]) >> bit_plane) & 1

def planes_str(ps) -> str:
    """把已用平面集合变成 '0,1,2' 这种字符串。"""
    ps = sorted(set(ps))
    return ",".join(map(str, ps))

def prepare_grad_and_masks_gray(imgL: Image.Image, T_notR: int):
    """
    输入 PIL 灰度图 L，计算 Sobel 梯度、边缘区 R、非边缘区 ~R 以及扫描坐标 (~R)。
    返回: gray_np, grad, R_mask, notR_mask, ys, xs
    """
    gray_np = np.array(imgL, dtype=np.uint8)
    grad = sobel_gradient_map(gray_np)
    R_mask = (grad >= T_notR)
    notR_mask = ~R_mask
    ys, xs = raster_indices_from_mask(R_mask, select_false=True)
    return gray_np, grad, R_mask, notR_mask, ys, xs

def gray_planes_sequential_match(
    img_gray: np.ndarray,
    ys: np.ndarray, 
    xs: np.ndarray,
    bitstream: str,
    planes = (0,1,2,3,4,5,6,7),                # 默认只用 LSB；如需多平面可传 (0,1,2,...)
) -> Tuple[List[Tuple[int,int,int,str,int]], int]:
    """
    仅匹配（不修改像素）。逐 plane（LSB→MSB）在 ~R 区域按行扫描，匹配 bitstream。
    输出 rows 列表（y,x,bit_plane）和“使用到的最高平面编号”。
    """
    out: List[Tuple[int,int,int]] = []
    bit_idx = 0
    need = len(bitstream)
    used_planes = set()

    for plane in planes:          # 外层：位平面
        if bit_idx >= need: break
        for y, x in zip(ys, xs):  # 内层：按像素顺序
            if bit_idx >= need: break
            b = read_bit_gray(img_gray, int(y), int(x), plane)
            if b == int(bitstream[bit_idx]):
                out.append((int(y), int(x), plane))
                used_planes.add(plane)
                bit_idx += 1

    planes_used = max(used_planes) if used_planes else -1
    return out, planes_used
