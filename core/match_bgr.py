# 负责匹配阶段：BT.601 灰度 + Sobel，取 ~R(T) 按 B→G→R 顺序在 LSB 上匹配 bitstream（不改像素）
# 提供：raster_indices_from_mask(), read_bit(), bgr_lsb_sequential_match(), prepare_bt601_grad_and_masks()
# core/match_bgr.py

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

from core.preprocess import rgb_to_gray_bt601
from core.edges import sobel_gradient_map

def raster_indices_from_mask(mask: np.ndarray, select_false: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    target = ~mask if select_false else mask
    ys, xs = np.nonzero(target)
    return ys.astype(np.int32), xs.astype(np.int32)

def channel_to_idx(ch: str) -> int:
    return {"R":0, "G":1, "B":2}[ch]

def read_bit(img_rgb: np.ndarray, y: int, x: int, ch: str, bit_plane: int = 0) -> int:
    c = channel_to_idx(ch)
    return (int(img_rgb[y, x, c]) >> bit_plane) & 1

def planes_str(ps):
    ps = sorted(set(ps))
    # 连续段压缩为 "0-2,4" 的形式，简单实现也可直接 ",".join(map(str,ps))
    return ",".join(map(str, ps))

def bgr_planes_sequential_match(
    img_rgb: np.ndarray,
    ys: np.ndarray, xs: np.ndarray,
    bitstream: str,
    planes = (0, 1, 2, 3, 4, 5, 6, 7),           # 先占用哪些位平面（建议从低到高）
    channel_order = ("B", "G", "R")  # 每个 plane 内的通道顺序
) -> Tuple[List[Tuple[int,int,int,str,int]], Tuple[int, ...]]:
    
    out: List[Tuple[int,int,int,str,int]] = []
    bit_idx = 0
    need = len(bitstream)
    used_planes = set()

    for plane in planes:                 # 外层：位平面（LSB→MSB）
        if bit_idx >= need: break
        for ch in channel_order:         # 里层：通道（B→G→R）
            if bit_idx >= need: break
            for y, x in zip(ys, xs):     # 再里层：像素序
                if bit_idx >= need: break
                b = read_bit(img_rgb, int(y), int(x), ch, plane)
                if b == int(bitstream[bit_idx]):
                    out.append((int(y), int(x), b, ch, plane))
                    used_planes.add(plane)
                    bit_idx += 1
    # planes_used = tuple(sorted(used_planes))
    planes_used = max(used_planes) if used_planes else -1
    return out, planes_used

def prepare_bt601_grad_and_masks(img: Image.Image, T_notR: int):
    gray_bt = rgb_to_gray_bt601(img)
    grad = sobel_gradient_map(gray_bt)
    R_mask = (grad >= T_notR)
    notR_mask = ~R_mask
    ys, xs = raster_indices_from_mask(R_mask, select_false=True)
    return gray_bt, grad, R_mask, notR_mask, ys, xs
