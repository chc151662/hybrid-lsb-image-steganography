# 负责把匹配 + proposed01 串起来；对外提供两个高层 API：
# fuse_embed(...)：返回 stego_np 和 positions_header, rows
# fuse_extract(...)：返回恢复的 bitstream 与统计

# core/pipeline_fuse.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image
from core.match_bgr import read_bit

from core.match_bgr import prepare_bt601_grad_and_masks, bgr_planes_sequential_match, planes_str
from core.positions import save_positions, parse_positions_header, parse_positions_rows
from core.proposed01 import embed_proposed01_on_R_edges, extract_proposed01_bits

def gen_random_bits(n_bits: int, seed: int = 2025) -> str:
    import numpy as np
    rng = np.random.default_rng(seed)
    return ''.join(rng.choice(['0','1'], size=n_bits))

# -------- EMBED --------
def fuse_embed(
    image_path: Path,
    out_stego_path: Path,
    positions_path: Path,
    payload_bits: Optional[Union[int, str]] = None,  # 支持 None/int/str
    T_notR: int = 100,
    proposed_min: int = 110,
    startG: int = 400, stepG: int = 10,
    seed: int = 2025,
) -> Dict[str, int]:
    """
    先在 ~R(T_notR) 上 BGR-LSB 顺序匹配；若不足则在 R-edges 上执行 proposed01。
    不管是否执行 proposed01，最终 stego 都写到 out_stego_path。
    还会写 positions_path（含 need_proposed01 标志和 stego_image 字段）。
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    H, W = img_np.shape[:2]

    # 准备 ~R(T) 和扫描坐标
    gray_bt, grad, R_mask, notR_mask, ys, xs = prepare_bt601_grad_and_masks(img, T_notR)
    notR_count = int(notR_mask.sum())

    # ------- 解析/生成负载 bitstream -------
    if payload_bits is None:
        # 原“半容量”估计：非边缘数量 * 3 通道 * 0.5；并限制一个上限（每像素最多 6 bit）
        payload_len = int(min(3 * notR_count * 0.5, H * W * 6))
        bitstream = gen_random_bits(payload_len, seed=seed)
    elif isinstance(payload_bits, int):
        payload_len = int(payload_bits)
        if payload_len < 0:
            raise ValueError("payload_bits(int) must be non-negative")
        bitstream = gen_random_bits(payload_len, seed=seed)
    elif isinstance(payload_bits, str):
        # 校验是 '0'/'1'
        if any(c not in "01" for c in payload_bits):
            raise ValueError("payload_bits(str) must be a '0'/'1' bitstring")
        bitstream = payload_bits
        payload_len = len(bitstream)
    else:
        raise TypeError("payload_bits must be None, int, or str")

    # ------- 匹配阶段（~R 区域的 BGR-LSB 顺序）-------
    matched, planes_used = bgr_planes_sequential_match(img_np, ys, xs, bitstream)
    matched_len = len(matched)
    remain_len = max(0, payload_len - matched_len)
    need_proposed = (remain_len > 0)

    # 保存头部 + 行级（先写入，方便溯源）
    header = {
        "total_bits": str(payload_len),
    }
    save_positions(positions_path, header, matched, width=W, note="fuse_embed")

    # 生成最终 stego
    if not need_proposed:
        # 仅 match 阶段已满足；img_np 已被就地修改（若 bgr_lsb_sequential_match 为原地嵌入）
        bpp = matched_len / (H * W)
        Image.fromarray(img_np, mode="RGB").save(out_stego_path)
        return {"matched": matched_len, 
                "embedded_R": 0, 
                "payload": payload_bits, 
                "remain": remain_len, 
                "planes_used": planes_used,
                "bpp": bpp
                }

    # 需要 proposed01：排除与 match 重叠像素
    used_flat = {(y*W + x) for (y, x, _b, _ch, _bp) in matched}
    remaining_bits = bitstream[matched_len:]

    stego_np, meta = embed_proposed01_on_R_edges(
        img_np, 
        remaining_bits, 
        start=startG, 
        step=stepG, 
        minimum=proposed_min, 
        skip_flat_set=used_flat
    )
    Image.fromarray(stego_np, mode="RGB").save(out_stego_path)

    bpp = (remain_len + matched_len)/ (H * W)


    return {"matched": matched_len, 
            "embedded_R": meta["embedded_R"], 
            "payload": payload_bits, 
            "remain": remain_len, 
            "planes_used": planes_used,
            "bpp": bpp
            }

# -------- EXTRACT --------
def fuse_extract(positions_path: Path, stego_path: Path) -> Tuple[str, Dict[str,int]]:
    """
    从 positions_path 读取 header+rows，自动定位 stego_image，
    先按行序恢复 match 段，再按 R-edges 规则恢复 proposed01 段，返回 bitstream 和统计信息。
    """
    header = parse_positions_header(positions_path)
    rows   = parse_positions_rows(positions_path)

    # stego_path = Path(header.get("stego_image", "output/lena_stego_fuse.png"))
    if not stego_path.exists():
        raise FileNotFoundError(str(stego_path.resolve()))

    img = Image.open(stego_path).convert("RGB")
    stego_np = np.array(img, dtype=np.uint8)
    H, W = stego_np.shape[:2]

    # A) 按行级顺序恢复匹配阶段
    # mismatch = oob = 0
    bitsA: List[str] = []
    used_flat = set()
    for r in rows:
        y = int(r["y"]); 
        x = int(r["x"]); 
        ch = r["channel"]; 
        bp = int(r["bit_plane"])
        b = read_bit(stego_np, y, x, ch, bp)
        bitsA.append(str(b)); used_flat.add(y*W+x)


    bitstreamA = "".join(bitsA)
    payload_total = int(header.get("total_bits", "-1"))

    proposed_Ghdr = 100


    # 计算剩余待嵌入的比特数 remain_len
    if payload_total > 0:
        # 如果已知总负载长度（payload_total），则直接计算差值
        remain_len = payload_total - len(bitstreamA)
        # 安全保护：防止因计算误差导致出现负数
        if remain_len < 0:
            remain_len = 0


    # B) 若需要，R-edges 恢复剩余位
    bitstreamB, metaB = ("", {})
    if remain_len > 0:
        bitstreamB, metaB = extract_proposed01_bits(
            stego_np, remain_len, used_flat, start=400, step=10, minimum=110, expect_threshold=proposed_Ghdr
        )

    all_bits = bitstreamA + bitstreamB
    if payload_total > 0 and len(all_bits) > payload_total:
        all_bits = all_bits[:payload_total]

    stats = {
        "match_bits": len(bitstreamA),
        "proposed_bits": len(bitstreamB),
    }
    return all_bits, stats