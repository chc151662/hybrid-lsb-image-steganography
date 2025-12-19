# tests/experiment.py
# 一体化脚本：按“明文大小”控制负载 -> 嵌入（记录实验表） -> 可选提取与解包预览
# 表格列：
# Cover-image | Size of Secret Message (B) | Compressed (B) | PSNR(dB) | MSE | Capacity (bpp) | Location File (KB) | Bit Planes

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import math
import csv
import numpy as np
from PIL import Image

from core.payload import pack_text, unpack_to_text  # 去头/无CRC版
from core.pipeline_fuse import fuse_embed, fuse_extract
from core.psnr import calculate_psnr, calculate_mse

# ===================== 配置区 =====================

# 运行模式: "embed" | "extract" | "both"
MODE = "embed"

# 数据与路径
# Airplane.bmp
# Baboon.bmp
# Barbara.bmp
# House.bmp
# Lena.bmp 
# Peppers.bmp
# Tiffany.bmp
# Tree.bmp

IMAGE_PATH  = Path("dataset/base/Airplane.bmp")
OUT_STEGO   = Path("output/Airplane_stego.bmp")
POS_PATH    = Path("output/positions_match_bgr_lsb.txt")

TEXT_PATH   = Path("dataset/The Complete Works of William Shakespeare.txt")
OUT_BITS_PATH  = Path("output/recovered_bits.txt")
OUT_TEXT_PATH  = Path("output/recovered_text_utf8.txt")
OUT_ERROR_PATH = Path("output/recovered_text_error.txt")

# —— Secret Message 尺寸控制方式（与 base paper 对齐：先定明文大小，再压缩）——
# 可选: "bytes" | "chars"
SECRET_SIZE_MODE   = "bytes"   # 推荐 bytes：与 8KB/16KB 一致

# 注意：TARGET_SECRET_SIZE 以 Byte 为单位 (1 KB = 1024 Bytes)
TARGET_SECRET_SIZE =  445.0 * 1024      # 若 mode=bytes 单位B；若 mode=chars 为字符数
ZLIB_LEVEL         = 6

# 若使用随机负载（对照），payload_bits=None，由 fuse 内部生成
USE_RANDOM_PAYLOAD = False

# 位置文件统计：使用 .xz 压缩后的大小作为“Location File (KB)”
COMPRESS_POS_TO_XZ = True

# 嵌入管线参数
T_NOTR        = 100
PROPOSED_MIN  = 110
START_G       = 400
STEP_G        = 10
SEED          = 2025

# 结果持久化（可选）：写入 TSV（Tab 分隔）
WRITE_TSV = True
TSV_PATH  = Path("exp_results.tsv")

# 提取预览
PREVIEW_HEAD_CHARS = 400

# =================================================

def ensure_parents(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def fmt_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units)-1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"

def compress_to_xz(src: Path, level: int = 9) -> Path:
    import lzma, shutil
    dst = src.with_suffix(src.suffix + ".xz")
    with open(src, "rb") as fin, lzma.open(dst, "wb", preset=level) as fout:
        shutil.copyfileobj(fin, fout)
    return dst

def read_full_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def take_prefix_by_bytes(text: str, n_bytes: int) -> str:
    """UTF-8 安全截断到 <= n_bytes 的前缀（不截半个字符）"""
    if n_bytes <= 0:
        return ""
    b = text.encode("utf-8")
    if len(b) <= n_bytes:
        return text
    hi = n_bytes
    while hi > 0:
        try:
            return b[:hi].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            hi -= 1
    return ""

def bit_planes_label(planes_used_raw: int | None) -> str:
    """将程序内部位面编号（0->LSB, 7->MSB）映射为人类可读标签"""
    if planes_used_raw is None:
        return "—"
    n = int(planes_used_raw) + 1
    if n == 1:
        return "1 (LSB)"
    if n == 8:
        return "8 (All bits)"
    return f"{n} (LSBs)"

def compute_psnr_mse(orig_path: Path, stego_path: Path):
    try:
        psnr_val = calculate_psnr(str(orig_path), str(stego_path))
    except Exception:
        orig = np.array(Image.open(orig_path).convert("RGB"))
        steg = np.array(Image.open(stego_path).convert("RGB"))
        diff = orig.astype("float32") - steg.astype("float32")
        mse = float(np.mean(diff ** 2))
        psnr_val = float("inf") if mse == 0.0 else 10.0 * math.log10((255.0 ** 2) / mse)
        return psnr_val, mse
    # MSE
    orig_arr = np.array(Image.open(orig_path))
    steg_arr = np.array(Image.open(stego_path))
    mse_val = calculate_mse(orig_arr, steg_arr)
    return psnr_val, float(mse_val)

def write_tsv_row(path: Path, header: list[str], row: list[str]):
    write_header = (not path.exists()) or (path.stat().st_size == 0)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if write_header:
            w.writerow(header)
        w.writerow(row)

def do_embed() -> dict:
    """执行嵌入与指标统计，返回用于后续 extract 的关键信息"""
    ensure_parents(OUT_STEGO)
    ensure_parents(POS_PATH)
    if WRITE_TSV:
        ensure_parents(TSV_PATH)

    # 1) 负载准备（按明文大小控制）
    if USE_RANDOM_PAYLOAD:
        payload_bits = None
        raw_bytes_len = None
        comp_bytes_len = None
        print("[PAYLOAD] 使用随机比特流（对照实验）")
    else:
        full_text = read_full_text(TEXT_PATH)
        if len(full_text) == 0:
            raise ValueError("输入文本为空：请检查 TEXT_PATH。")

        if SECRET_SIZE_MODE == "bytes":
            target_B = int(TARGET_SECRET_SIZE)
            text = take_prefix_by_bytes(full_text, target_B)
            raw_bytes_len = len(text.encode("utf-8"))
        elif SECRET_SIZE_MODE == "chars":
            n_chars = int(TARGET_SECRET_SIZE)
            text = full_text[:max(0, min(len(full_text), n_chars))]
            raw_bytes_len = len(text.encode("utf-8"))
        else:
            raise ValueError("SECRET_SIZE_MODE 必须是 'bytes' 或 'chars'")

        payload_bits = pack_text(text, level=ZLIB_LEVEL)
        bits_len = len(payload_bits)
        comp_bytes_len = (bits_len + 7) // 8

        print("[PAYLOAD] text file :", TEXT_PATH)
        print(f"[PAYLOAD] mode      : {SECRET_SIZE_MODE}")
        print(f"[PAYLOAD] target    : {TARGET_SECRET_SIZE} ({'B' if SECRET_SIZE_MODE=='bytes' else 'chars'})")
        print(f"[PAYLOAD] raw size  : {raw_bytes_len:,} B")
        print(f"[PAYLOAD] compressed: {comp_bytes_len:,} B")
        if comp_bytes_len > 0:
            print(f"[PAYLOAD] compression ratio: {raw_bytes_len/comp_bytes_len:.2f}x\n")

    # 2) 嵌入
    meta = fuse_embed(
        image_path=IMAGE_PATH,
        out_stego_path=OUT_STEGO,
        positions_path=POS_PATH,
        payload_bits=payload_bits,   # None -> fuse 内部使用随机流
        T_notR=T_NOTR,
        proposed_min=PROPOSED_MIN,
        startG=START_G,
        stepG=STEP_G,
        seed=SEED,
    )

    print("[EMBED] image          :", IMAGE_PATH)
    print("[EMBED] out            :", OUT_STEGO)
    print("[EMBED] positions path :", POS_PATH)
    print("[EMBED] planes_used    :", meta.get("planes_used"))
    print("[EMBED] matched        :", meta.get("matched"))
    print("[EMBED] embedded_R     :", meta.get("embedded_R"))
    print("[EMBED] remain         :", meta.get("remain"))
    print("[EMBED] bpp            :", meta.get("bpp"))
    print("[DONE] Fuse embedding completed.\n")

    # 3) 指标：PSNR / MSE
    psnr_val, mse_val = compute_psnr_mse(IMAGE_PATH, OUT_STEGO)

    # 4) 位置文件（取 .xz）
    try:
        raw_bytes = POS_PATH.stat().st_size
        print(f"[POS] raw txt size  : {raw_bytes:,} B ({fmt_bytes(raw_bytes)})")
        if COMPRESS_POS_TO_XZ and raw_bytes > 0:
            xz_path = compress_to_xz(POS_PATH)
            xz_bytes = xz_path.stat().st_size
            pos_kb = xz_bytes / 1024.0
            print(f"[POS] .xz size      : {xz_bytes:,} B ({fmt_bytes(xz_bytes)})")
            reduction = 100.0 * (1 - xz_bytes / raw_bytes)
            print(f"[POS] reduction     : {reduction:.2f}% (recording .xz size)")
        else:
            pos_kb = raw_bytes / 1024.0
    except FileNotFoundError:
        pos_kb = 0.0
        print("[WARN] positions 文件尚未生成，无法统计大小。")

    # 5) 表格行
    cover_image = IMAGE_PATH.name
    planes_label = bit_planes_label(meta.get("planes_used"))
    capacity_bpp = float(meta.get("bpp", 0.0))

    psnr_is_inf = (np.isinf(psnr_val) or mse_val == 0.0)
    psnr_str_for_table = "∞" if psnr_is_inf else f"{psnr_val:.2f}"
    psnr_for_csv = "inf" if psnr_is_inf else f"{psnr_val:.6f}"
    mse_str_for_table = f"{mse_val:.3f}"
    mse_for_csv = f"{mse_val:.6f}"

    size_secret_B = "" if USE_RANDOM_PAYLOAD else f"{raw_bytes_len}"
    compressed_B  = "" if USE_RANDOM_PAYLOAD else f"{comp_bytes_len}"

    header_cols = [
        "Cover-image",
        "Size of Secret Message (B)",
        "Compressed (B)",
        "PSNR(dB)",
        "MSE",
        "Capacity (bpp)",
        "Location File (KB)",
        "Bit Planes",
    ]
    row_cols = [
        cover_image,
        size_secret_B,
        compressed_B,
        psnr_str_for_table,
        mse_str_for_table,
        f"{capacity_bpp:.4f}",
        f"{pos_kb:.1f}",
        planes_label,
    ]

    print("\n[RESULT-TABLE] -------------------------------")
    print("\t".join(header_cols))
    print("\t".join(row_cols))
    print("[RESULT-TABLE] -------------------------------\n")

    # 可选：写 TSV
    if WRITE_TSV and not USE_RANDOM_PAYLOAD:
        write_tsv_row(TSV_PATH, header_cols, [
            cover_image,
            size_secret_B,
            compressed_B,
            psnr_for_csv,
            mse_for_csv,
            f"{capacity_bpp:.6f}",
            f"{pos_kb:.3f}",
            planes_label,
        ])
        print(f"[SAVE] TSV appended -> {TSV_PATH.resolve()}\n")

    return {
        "psnr": psnr_val,
        "mse": mse_val,
        "pos_kb": pos_kb,
        "stego": OUT_STEGO,
        "positions": POS_PATH,
    }

def do_extract():
    """执行提取与解包预览（使用当前 POS_PATH 与 OUT_STEGO）"""
    # 1) 提取比特串
    bits, stats = fuse_extract(POS_PATH, OUT_STEGO)  # 你的实现签名是 (positions_path, stego_path)
    print("[EXTRACT] stats:", stats)

    # 2) 保存原始比特
    ensure_parents(OUT_BITS_PATH)
    OUT_BITS_PATH.write_text(bits, encoding="utf-8")
    print(f"\n[SAVE] raw bits -> {OUT_BITS_PATH} (len(bits)={len(bits):,})")

    # 3) zlib 解包 -> UTF-8 文本
    try:
        text = unpack_to_text(bits)  # 无 CRC，若比特损坏或顺序错，可能抛 zlib.error
    except Exception as e:
        msg = (
            f"[UNPACK][ERROR] {type(e).__name__}: {e}\n"
            "- 可能原因：\n"
            "  1) 嵌入阶段使用了随机比特流（USE_RANDOM_PAYLOAD=True），不是 zlib 负载；\n"
            "  2) 负载被截断（remain>0 或容量不足），导致 zlib 解压失败；\n"
            "  3) 提取使用的 positions 与 stego 不匹配；\n"
            "  4) 提取顺序/位点不一致，造成 bitstream 破损。\n"
        )
        ensure_parents(OUT_ERROR_PATH)
        OUT_ERROR_PATH.write_text(msg, encoding="utf-8")
        print(msg)
        return

    # 4) 成功：保存文本并做预览
    ensure_parents(OUT_TEXT_PATH)
    OUT_TEXT_PATH.write_text(text, encoding="utf-8")
    print(f"[SAVE] recovered text -> {OUT_TEXT_PATH} (chars={len(text):,})")
    preview = text[:PREVIEW_HEAD_CHARS].replace("\n", "\\n")
    print(f"\n[PREVIEW {min(PREVIEW_HEAD_CHARS, len(text))} chars] {preview}")

if __name__ == "__main__":
    if MODE not in {"embed", "extract", "both"}:
        raise ValueError("MODE 必须是 'embed' | 'extract' | 'both'")

    if MODE in {"embed", "both"}:
        do_embed()

    if MODE in {"extract", "both"}:
        do_extract()