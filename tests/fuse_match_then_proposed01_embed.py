# tests/fuse_match_then_proposed01_embed.py
# 组合：读取文本(可选截断) -> Zlib+CRC打包为比特串 -> fuse_match_then_proposed01 嵌入
# 说明：
#   - 不使用命令行参数，所有设置写在“配置区”
#   - 如需回到随机负载对照，只需把 USE_RANDOM_PAYLOAD = True

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from core.payload import pack_text
from core.pipeline_fuse import fuse_embed
import numpy as np
from PIL import Image
from core.psnr import calculate_psnr, calculate_mse


# ===================== 配置区 =====================
IMAGE_PATH  = Path("dataset/lena.png")
OUT_STEGO   = Path("output/lena_stego_fuse.png")
POS_PATH    = Path("output/positions_match_bgr_lsb.txt")

# 待嵌入文本（UTF-8）
TEXT_PATH   = Path("dataset/The Complete Works of William Shakespeare.txt")

# 若为 None 则不截断；否则按字符数截断（便于控量/对照实验）

# 570000是可以使用Embedding阶段的字符长度
# 7000是lena图片的0.1004左右bpp
MAX_CHARS   = 16000

# Zlib 压缩等级 0..9（6 为常用折中）
ZLIB_LEVEL  = 6

# 若设为 True，则忽略文本负载，使用 fuse 内部的随机比特流（做对照实验用）
USE_RANDOM_PAYLOAD = False

# 是否在写完 positions 后额外生成 .xz 压缩版并打印两者大小
COMPRESS_POS_TO_XZ = True

# fuse pipeline 参数（保持你当前实验习惯）
T_NOTR        = 100
PROPOSED_MIN  = 110
START_G       = 400
STEP_G        = 10
SEED          = 2025
# =================================================


def read_text(path: Path, max_chars: int | None) -> str:
    """读取 UTF-8 文本；可选字符截断。"""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if max_chars is not None:
        txt = txt[:max_chars]
    return txt

def compress_to_xz(src: Path, level: int = 9) -> Path:
    """将文本文件压缩为 .xz 并返回路径（若不需要可关闭 COMPRESS_POS_TO_XZ）"""
    import lzma, shutil
    dst = src.with_suffix(src.suffix + ".xz")
    with open(src, "rb") as fin, lzma.open(dst, "wb", preset=level) as fout:
        shutil.copyfileobj(fin, fout)
    return dst

def fmt_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units)-1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


if __name__ == "__main__":
    # 确保输出目录存在
    OUT_STEGO.parent.mkdir(parents=True, exist_ok=True)
    POS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if USE_RANDOM_PAYLOAD:
        payload_bits = None
        print("[PAYLOAD] 使用随机比特流（对照实验）")
    else:
        # 1) 读取并可选截断
        text = read_text(TEXT_PATH, MAX_CHARS)
        if len(text) == 0:
            raise ValueError("输入文本为空：请检查 TEXT_PATH 或 MAX_CHARS 设置。")

        # 2) Zlib 打包为 '0'/'1' 比特串
        payload_bits = pack_text(text, level=ZLIB_LEVEL)
        bits_len = len(payload_bits)
        # comp_bytes_len = bits_len // 8

        raw_bits_len = len(text.encode("utf-8")) * 8

        print("[PAYLOAD] text file:", TEXT_PATH)
        print(f"[PAYLOAD] orig chars: {len(text):,}")
        print(f"[PAYLOAD] text raw bit length: {raw_bits_len:,} bits")
        print(f"[PAYLOAD] compressed bits (after Zlib): {bits_len:,} bit")
        print(f"[PAYLOAD] compression ratio         : {raw_bits_len / bits_len:.2f}x")
        
        if len(text) > 0:
            print(f"[PAYLOAD] bytes-per-char (approx): {bits_len/len(text):.3f} bit/char")

    # 3) 调用 fuse pipeline 进行嵌入
    meta = fuse_embed(
        image_path=IMAGE_PATH,
        out_stego_path=OUT_STEGO,
        positions_path=POS_PATH,
        payload_bits=payload_bits,   # 若为 None 则 fuse 内部会用随机比特流
        T_notR=T_NOTR,
        proposed_min=PROPOSED_MIN,
        startG=START_G,
        stepG=STEP_G,
        seed=SEED,
    )

    print("[EMBED] image:", IMAGE_PATH)
    print("[EMBED] out  :", OUT_STEGO)
    print("[EMBED] pos  :", POS_PATH)
    print("[EMBED] planes_used :", meta["planes_used"])
    print("[EMBED] matched :", meta["matched"])
    print("[EMBED] embedded_R :", meta["embedded_R"])
    print("[EMBED] remain :", meta["remain"])
    print("[EMBED] bpp :", meta["bpp"])
    print("[EMBED] bit_planes_used :", meta["planes_used"])
    print("[DONE] Fuse embedding completed.")

    # 4) === 计算并打印 PSNR 与 MSE ===
    try:
        psnr = calculate_psnr(str(IMAGE_PATH), str(OUT_STEGO))
    except Exception as e:
        # 极少数情况下(色彩通道/位深不一致)可退回到数组级计算
        orig_arr = np.array(Image.open(IMAGE_PATH).convert("RGB"))
        stego_arr = np.array(Image.open(OUT_STEGO).convert("RGB"))
        diff = orig_arr.astype("float") - stego_arr.astype("float")
        mse_fallback = float(np.mean(diff ** 2))
        psnr = float("inf") if mse_fallback == 0 else 10.0 * np.log10((255.0 ** 2) / mse_fallback)

    # 与 calculate_psnr 同源方式：直接读入，不做模式转换，保证一致性
    orig_arr_for_mse = np.array(Image.open(IMAGE_PATH))
    stego_arr_for_mse = np.array(Image.open(OUT_STEGO))
    mse = calculate_mse(orig_arr_for_mse, stego_arr_for_mse)

    # 5) positions 压缩为 .xz 并打印尺寸对比
    try:
        raw_sz = POS_PATH.stat().st_size
        print(f"[POS] raw txt size  : {raw_sz:,} B ({fmt_bytes(raw_sz)})")
        if COMPRESS_POS_TO_XZ:
            xz_path = compress_to_xz(POS_PATH)
            xz_sz = xz_path.stat().st_size
            print(f"[POS] .xz size      : {xz_sz:,} B ({fmt_bytes(xz_sz)})")
            reduction = 100.0 * (1 - xz_sz / raw_sz) if raw_sz > 0 else 0.0
            print(f"[POS] reduction     : {reduction:.2f}%")
    except FileNotFoundError:
        print("[WARN] positions 文件尚未生成，无法统计大小。")

    print(f"[METRIC] PSNR: {psnr:.6f} dB")
    print(f"[METRIC] MSE : {mse:.6f}")