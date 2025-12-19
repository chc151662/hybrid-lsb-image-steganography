# tests/find_max_secret_for_image.py
# 目标：找到 remain==0 的最大明文大小（bytes/chars），即“最大可嵌入 Secret Message 长度”。

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import math
import numpy as np
from PIL import Image

from core.payload import pack_text
from core.pipeline_fuse import fuse_embed
from core.psnr import calculate_psnr, calculate_mse

# ===================== 配置区 =====================
IMAGE_PATH  = Path("dataset/base/Lena.bmp")
OUT_STEGO   = Path("output/_tmp_max_stego.bmp")
POS_PATH    = Path("output/_tmp_positions_max.txt")
TEXT_PATH   = Path("dataset/The Complete Works of William Shakespeare.txt")

# 明文大小控制：与 base paper 对齐
SECRET_SIZE_MODE   = "bytes"   # "bytes" | "chars"
SEED               = 2025
ZLIB_LEVEL         = 6

# 搜索参数
START_SIZE         = 4096      # 初始尝试的明文大小（B 或 chars）
MAX_ITER           = 11       # 总迭代上限（指数阶段+二分阶段）
# =================================================

# 嵌入管线参数（保持与你主实验一致）
T_NOTR        = 100
PROPOSED_MIN  = 110
START_G       = 400
STEP_G        = 10

def read_full_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def take_prefix_by_bytes(text: str, n_bytes: int) -> str:
    if n_bytes <= 0: return ""
    b = text.encode("utf-8")
    if len(b) <= n_bytes: return text
    hi = n_bytes
    while hi > 0:
        try:
            return b[:hi].decode("utf-8", "strict")
        except UnicodeDecodeError:
            hi -= 1
    return ""

def compute_psnr_mse(orig: Path, steg: Path):
    try:
        psnr_val = calculate_psnr(str(orig), str(steg))
    except Exception:
        o = np.array(Image.open(orig).convert("RGB"))
        s = np.array(Image.open(steg).convert("RGB"))
        diff = o.astype("float32") - s.astype("float32")
        mse = float(np.mean(diff ** 2))
        psnr_val = float("inf") if mse == 0.0 else 10.0 * math.log10((255.0**2)/mse)
        return psnr_val, mse
    o = np.array(Image.open(orig))
    s = np.array(Image.open(steg))
    mse = float(calculate_mse(o, s))
    return psnr_val, mse

def build_text(full_text: str, target: int) -> str:
    if SECRET_SIZE_MODE == "bytes":
        return take_prefix_by_bytes(full_text, target)
    else:
        return full_text[:max(0, min(len(full_text), target))]

def try_embed(full_text: str, size_val: int):
    text = build_text(full_text, size_val)
    raw_B = len(text.encode("utf-8"))
    bits = pack_text(text, level=ZLIB_LEVEL)

    try:
        meta = fuse_embed(
            image_path=IMAGE_PATH,
            out_stego_path=OUT_STEGO,
            positions_path=POS_PATH,
            payload_bits=bits,
            T_notR=T_NOTR, proposed_min=PROPOSED_MIN,
            startG=START_G, stepG=STEP_G,
            seed=SEED
        )
        # 成功：可计算 PSNR/MSE
        psnr, mse = compute_psnr_mse(IMAGE_PATH, OUT_STEGO)
        compB  = (len(bits) + 7) // 8
        bpp    = float(meta.get("bpp", 0.0))
        return {
            "ok": True,
            "remain": int(meta.get("remain", 0)),
            "raw_B": raw_B,
            "comp_B": compB,
            "psnr": psnr,
            "mse": mse,
            "bpp": bpp,
            "planes_used": meta.get("planes_used"),
        }

    except RuntimeError as e:
        # 容量不足 -> 视为失败(hi 边界)
        if "容量不足" in str(e) or "capacity" in str(e) or "不足" in str(e):
            compB  = (len(bits) + 7) // 8
            return {
                "ok": False,
                "remain": 1,            # 标记失败
                "raw_B": raw_B,
                "comp_B": compB,
                "psnr": float("inf"),   # 无意义，占位
                "mse": 0.0,             # 无意义，占位
                "bpp": 0.0,             # 无意义，占位
                "planes_used": None,
            }
        # 其他异常继续抛出，避免吞错
        raise


if __name__ == "__main__":
    full_text = read_full_text(TEXT_PATH)
    Nmax = len(full_text) if SECRET_SIZE_MODE=="chars" else len(full_text.encode("utf-8"))

    # 1) 指数扩张，找到 [lo (ok), hi (fail)] 区间
    if SECRET_SIZE_MODE == "bytes":
        size = min(START_SIZE, Nmax)
    else:
        size = min(START_SIZE, len(full_text))

    lo_size = 0
    lo_res  = None
    hi_size = None
    hi_res  = None

    for _ in range(MAX_ITER):
        res = try_embed(full_text, size)
        print(f"[TRY] size={size} ({'B' if SECRET_SIZE_MODE=='bytes' else 'chars'}) -> remain={res['remain']} ok={res['ok']} comp={res['comp_B']}B bpp={res['bpp']:.4f}")
        if res["ok"]:
            lo_size, lo_res = size, res
            nxt = size*2
            if nxt > Nmax: break
            size = nxt
        else:
            hi_size, hi_res = size, res
            break

    if lo_res is None:
        print("[RESULT] 无法嵌入任何明文（lo_res=None）。")
        sys.exit(0)

    if hi_res is None:
        # 到文本极限都 ok，取 lo_res 为当前最大
        hi_size = min(size, Nmax)

    # 2) 二分在 [lo, hi] 搜最大 ok
    L, R = lo_size, hi_size
    best = lo_res
    for _ in range(MAX_ITER):
        if R <= L+1: break
        mid = (L+R)//2
        res = try_embed(full_text, mid)
        if res["ok"]:
            L = mid; best = res
        else:
            R = mid

    # 3) 输出结果
    print("\n[RESULT] === 最大可嵌入（remain==0）的明文大小 ===")
    print(f"Image        : {IMAGE_PATH.name}")
    print(f"Mode         : {SECRET_SIZE_MODE}")
    print(f"Max Secret   : {L} ({'B' if SECRET_SIZE_MODE=='bytes' else 'chars'})")
    print(f"Raw(B)       : {best['raw_B']}")
    print(f"Compressed(B): {best['comp_B']}")

    psnr_str = "∞" if np.isinf(best["psnr"]) else f"{best['psnr']:.2f}"
    print(f"PSNR(dB)     : {psnr_str}")

    print(f"MSE          : {best['mse']:.6f}")
    print(f"Capacity(bpp): {best['bpp']:.6f}")
    nplanes = (best['planes_used']+1) if best['planes_used'] is not None else None
    print(f"Bit Planes   : { '—' if nplanes is None else ( '1 (LSB)' if nplanes==1 else ('8 (All bits)' if nplanes==8 else f'{nplanes} (LSBs)') ) }")
