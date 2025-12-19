# 负责Proposed01 嵌入/提取（R 通道重建 R(G)，上三角 XNOR / 下三角 Fibonacci，主对角线跳过，且避免和匹配阶段重叠）
# 提供：edge_coords_for_length_R(), embed_proposed01_on_R_edges(), extract_proposed01_bits()
# core/proposed01.py

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

from core.preprocess import gray_from_channel
from core.edges import select_edge_pixels
from core.xnor import embed_xnor_pixel, extract_xnor_pixel
from core.fibonacci import embed_fibonacci_pixel, extract_fibonacci

def edge_coords_for_length_R(gray_R: np.ndarray, length: int, start=400, step=10, minimum=110):
    sel = select_edge_pixels(gray=gray_R, message_bits=length, start=start, step=step, minimum=minimum, return_mode="yx")
    coords_all = sel["indices"]
    coords = coords_all[coords_all[:,0] != coords_all[:,1]]  # 去掉主对角线
    return coords, int(sel["threshold"]), int(sel["count"]), int(coords.shape[0])

def embed_proposed01_on_R_edges(
    input_image_arr: np.ndarray, bitstream: str,
    start: int = 400, step: int = 10, minimum: int = 110,
    skip_flat_set: Optional[set[int]] = None
) -> Tuple[np.ndarray, Dict[str,int]]:
    img = input_image_arr.copy()
    H, W = img.shape[:2]
    gray_R = gray_from_channel(Image.fromarray(img, mode="RGB"), channel="R")
    coords, G, count_all, effective_capacity = edge_coords_for_length_R(gray_R, len(bitstream), start, step, minimum)

    skip_overlap = 0
    if skip_flat_set:
        keep = []
        for (y, x) in coords:
            if (y*W + x) in skip_flat_set:
                skip_overlap += 1; continue
            keep.append((y, x))
        coords = np.array(keep, dtype=np.int32) if keep else np.empty((0,2), dtype=np.int32)

    if coords.shape[0] < len(bitstream):
        raise RuntimeError(f"[proposed01] 容量不足：{coords.shape[0]} < {len(bitstream)} (G={G}, min={minimum}, skip_overlap={skip_overlap})")

    embedded = 0
    for (y, x), b in zip(coords, bitstream):
        bit = int(b)
        r, g, bch = img[y, x]
        if y < x:
            r, g, bch = embed_xnor_pixel(r, g, bch, bit)
        else:
            r, g, bch = embed_fibonacci_pixel(r, g, bch, bit)
        img[y, x] = [r, g, bch]
        embedded += 1
        if embedded >= len(bitstream): break

    meta = {
        "threshold_R": G,
        "count_all_R": count_all,
        "effective_capacity_R": effective_capacity,
        "embedded_R": embedded,
        "skip_overlap": skip_overlap,
    }
    return img, meta

def extract_proposed01_bits(
    stego_img_np: np.ndarray, remain_len: int, used_flat_from_match: set,
    start=400, step=10, minimum=110, expect_threshold: Optional[int]=None
) -> Tuple[str, Dict[str,int]]:
    if remain_len <= 0:
        return "", {"threshold_R":-1, "coords_available":0, "skip_overlap":0}
    H, W = stego_img_np.shape[:2]
    gray_R = gray_from_channel(Image.fromarray(stego_img_np, mode="RGB"), channel="R")
    coords, G, _cnt, eff = edge_coords_for_length_R(gray_R, remain_len, start, step, minimum)

    filtered, skip_overlap = [], 0
    for (y, x) in coords:
        if (y*W+x) in used_flat_from_match: skip_overlap += 1; continue
        filtered.append((int(y), int(x)))
    coords = filtered
    if len(coords) < remain_len:
        raise RuntimeError(f"[extract_proposed01] 可用坐标不足：{len(coords)} < {remain_len} (G={G}, min={minimum}, skip_overlap={skip_overlap})")

    if expect_threshold is not None and G != int(expect_threshold):
        print(f"[warn] G={G} != header.proposed_threshold_R={expect_threshold} —— 继续按当前 G 提取。")

    bits: List[str] = []
    for (y, x) in coords:
        if len(bits) >= remain_len: break
        r, g, bch = stego_img_np[y, x]
        bit = extract_xnor_pixel(r, g, bch) if y < x else extract_fibonacci(g)
        bits.append(str(bit))
    return "".join(bits[:remain_len]), {"threshold_R": G, "coords_available": len(coords), "skip_overlap": skip_overlap}
