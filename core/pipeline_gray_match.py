# core/pipeline_gray_match.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
from core.positions import parse_positions_header, parse_positions_rows
from core.match_gray import read_bit_gray
import numpy as np
from PIL import Image

from core.positions import save_positions, save_positions_gray, parse_positions_rows_gray
from core.match_gray import (
    prepare_grad_and_masks_gray, gray_planes_sequential_match, planes_str
)

def gen_random_bits(n_bits: int, seed: int = 2025) -> str:
    import numpy as np
    rng = np.random.default_rng(seed)
    return ''.join(rng.choice(['0','1'], size=n_bits))

def gray_match_embed(
    image_path: Path,
    out_stego_path: Path,
    positions_path: Path,
    payload_bits: Optional[Union[int, str]] = None,  # None/int = 生成随机，str='01...' 直接用
    T_notR: int = 100,
    seed: int = 2025,
) -> Dict[str, Union[int,float,str]]:
    """
    灰度图像（L 模式）仅匹配的嵌入（零失真）。
    - 在 ~R(T_notR) 上、按指定 bit-planes 顺序匹配 bitstream；
    - 不进行 proposed/XNOR/Fibonacci；
    - 仍写出 “stego” 文件（与原图像一致）与 positions（行级通道记 'Y'）。
    返回统计：matched、remain、planes_used、bpp 等。
    """
    # 读取灰度图
    imgL = Image.open(image_path).convert("L")
    gray_np = np.array(imgL, dtype=np.uint8)
    H, W = gray_np.shape[:2]
    num_pixels = H * W

    # 生成/解析负载
    if payload_bits is None:
        # 给一个保守的默认：~R 区域的一半容量（按 1 平面估算）
        # 这里不改像素，仅匹配，所以最多也只能等于 ~R 像素数
        # 为了有随机性，用 0.5 × 像素数 的上限
        payload_len = 6553
        bitstream = gen_random_bits(payload_len, seed=seed)
    elif isinstance(payload_bits, int):
        payload_len = int(payload_bits)
        if payload_len < 0:
            raise ValueError("payload_bits(int) must be non-negative")
        bitstream = gen_random_bits(payload_len, seed=seed)
    elif isinstance(payload_bits, str):
        if any(c not in "01" for c in payload_bits):
            raise ValueError("payload_bits(str) must be a '0'/'1' bitstring")
        bitstream = payload_bits
        payload_len = len(bitstream)
    else:
        raise TypeError("payload_bits must be None, int, or str")

    # 准备 ~R(T) 与扫描坐标
    gray_bt, grad, R_mask, notR_mask, ys, xs = prepare_grad_and_masks_gray(imgL, T_notR)
    notR_count = int(notR_mask.sum())

    # 匹配阶段
    matched_rows, planes_used = gray_planes_sequential_match(
        gray_np, ys, xs, bitstream
    )
    matched_len = len(matched_rows)
    remain_len = max(0, payload_len - matched_len)

    header = {
        # "image": str(image_path),
        # "stego_image": str(out_stego_path),
        # "size(HxW)": f"{H}x{W}",
        # "positions_format": "verbose_v1",     # 保持你现有 positions 读写
        # "match_threshold_notR": str(T_notR),
        # "position_type": "non-edge",
        # "bit_planes_used": planes_str([planes_used]) if planes_used >= 0 else "",
        "total_bits": str(payload_len),
        # "matched_bits_in_match_phase": str(matched_len),
        # "total_positions_in_non-edge": str(notR_count),
        # "need_proposed01": "False",
        # "remaining_bits_to_embed_by_proposed01": "0",
        # "pipeline": "GRAY_MATCH_ONLY",
        # "selector": "Sobel(~R)",
        # "seed": str(seed),
    }
    save_positions_gray(positions_path, header, matched_rows)

    # “stego” 等于原图（零失真），依然输出，便于评测流水线统一
    imgL.save(out_stego_path)

    bpp = matched_len / (H * W)
    return {
        "matched": matched_len,
        "embedded_R": 0,
        "payload": payload_len,
        "remain": remain_len,
        "planes_used": planes_used,
        "bpp": bpp,
    }


def gray_match_extract(positions_path: Path, stego_path: Path) -> Tuple[str, Dict[str, int]]:
    """
    从 positions_path 读取 header+rows（灰度·仅匹配格式），
    按行顺序从 stego（灰度图）逐位读取，恢复 bitstream。
    返回： (bitstream, stats)
    stats: {"match_bits": N, "proposed_bits": 0}
    """
    header = parse_positions_header(positions_path)
    rows   = parse_positions_rows_gray(positions_path)

    if not stego_path.exists():
        raise FileNotFoundError(str(stego_path.resolve()))

    # stego 灰度图（match-only 情况下，等于原图；但我们仍从该文件读取，保持流程统一）
    imgL = Image.open(stego_path).convert("L")
    stego_np = np.array(imgL, dtype=np.uint8)
    H, W = stego_np.shape[:2]

    bitsA: List[str] = []
    for r in rows:
        y  = int(r["y"])
        x  = int(r["x"])
        bp = int(r["bit_plane"])
        b = read_bit_gray(stego_np, y, x, bp)
        bitsA.append(str(b))

    bitstream = "".join(bitsA)

    # # 若 header 里记录了总负载长度，则做一次安全裁剪
    # payload_total = -1
    # if "payload_total_bits" in header:
    #     try:
    #         payload_total = int(header.get("payload_total_bits", "-1"))
    #     except Exception:
    #         payload_total = -1
    # elif "total_bits" in header:
    #     # 兼容你彩色管线里使用的键名
    #     try:
    #         payload_total = int(header.get("total_bits", "-1"))
    #     except Exception:
    #         payload_total = -1

    # if payload_total > 0 and len(bitstream) > payload_total:
    #     bitstream = bitstream[:payload_total]

    stats = {
        "match_bits": len(bitstream)
    }
    return bitstream, stats
