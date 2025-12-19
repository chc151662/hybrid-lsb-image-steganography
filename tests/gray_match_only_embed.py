# tests/gray_match_only_embed.py
# 组合：读取文本(按“明文大小=Bytes/Chars”控量) -> Zlib+CRC打包为比特串 -> 灰度(~R)匹配式嵌入（match-only，零失真）
# 说明：
#   - 与参考论文对齐：灰度图、仅用 LSB（planes=(0,)），可按 bpp/明文字节自行控量
#   - 默认按 Bytes 控制（与“8KB/16KB”口径一致）

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from core.payload import pack_text
from core.pipeline_gray_match import gray_match_embed   # 你的灰度管线

# ===================== 配置区 =====================

# BOWS-2 的灰度 PGM 图
IMAGE_PATH  = Path("dataset/BOWS_2/9000.pgm")
OUT_STEGO   = Path("output/BOWS_2/9000_gray_match_only.pgm")   # 输出“stego”（零失真=与原图一致）
POS_PATH    = Path("output/positions_gray_match_only.txt")

# 待嵌入文本（UTF-8）
TEXT_PATH   = Path("dataset/The Complete Works of William Shakespeare.txt")

# —— Secret Message 尺寸控制方式（先定“明文大小”再压缩）——
# 可选: "bytes" | "chars"
SECRET_SIZE_MODE   = "bytes"          # 推荐 "bytes"：与 8KB/16KB 一致
TARGET_SECRET_SIZE = int(50 * 1024)    # 例如设成 8KB 明文：8*1024 = 8192 B

# Zlib 压缩等级 0..9（6 为常用折中）
ZLIB_LEVEL  = 6

# 若为 True，则忽略文本负载，使用内部随机比特流（对照实验）
USE_RANDOM_PAYLOAD = False

# 灰度匹配参数
T_NOTR      = 200         # Sobel 阈值：~R(T) 为非边缘区域
SEED        = 2025

# 是否在写完 positions 后额外生成 .xz 压缩版并打印两者大小
COMPRESS_POS_TO_XZ = True

# ===================== 工具函数 =====================

def ensure_parents(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

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

def compress_to_xz(src: Path, level: int = 9) -> Path:
    """将文本文件压缩为 .xz 并返回路径"""
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

# ===================== 主流程 =====================

if __name__ == "__main__":
    # 确保输出目录存在
    ensure_parents(OUT_STEGO)
    ensure_parents(POS_PATH)

    if USE_RANDOM_PAYLOAD:
        payload_bits = None
        print("[PAYLOAD] 使用随机比特流（对照实验）")
    else:
        # 1) 读取全文并按 Bytes/Chars 控量
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

        # 2) Zlib+CRC 打包为 '0'/'1' 比特串
        payload_bits = pack_text(text, level=ZLIB_LEVEL)
        bits_len = len(payload_bits)
        comp_bytes_len = (bits_len + 7) // 8

        # 打印 payload 统计
        print("[PAYLOAD] text file :", TEXT_PATH)
        print(f"[PAYLOAD] mode      : {SECRET_SIZE_MODE}")
        print(f"[PAYLOAD] target    : {TARGET_SECRET_SIZE} ({'B' if SECRET_SIZE_MODE=='bytes' else 'chars'})")
        print(f"[PAYLOAD] raw size  : {raw_bytes_len:,} B ({fmt_bytes(raw_bytes_len)})")
        print(f"[PAYLOAD] compressed: {comp_bytes_len:,} B ({fmt_bytes(comp_bytes_len)})")
        if comp_bytes_len > 0:
            print(f"[PAYLOAD] compression ratio: {raw_bytes_len/comp_bytes_len:.2f}x")

    # 3) 调用灰度匹配管线（零失真，仅匹配）
    meta = gray_match_embed(
        image_path=IMAGE_PATH,
        out_stego_path=OUT_STEGO,
        positions_path=POS_PATH,
        payload_bits=payload_bits,   # None -> 内部随机
        T_notR=T_NOTR,
        seed=SEED,
    )

    print("[EMBED] image       :", IMAGE_PATH)
    print("[EMBED] out         :", OUT_STEGO)
    print("[EMBED] pos         :", POS_PATH)
    print("[EMBED] planes_used :", meta.get("planes_used"))
    print("[EMBED] matched     :", meta.get("matched"))
    print("[EMBED] bpp         :", meta.get("bpp"))

    # 4)（可选）positions 压缩为 .xz 并打印尺寸对比
    try:
        raw_sz = POS_PATH.stat().st_size
        print(f"[POS] raw txt size  : {raw_sz:,} B ({fmt_bytes(raw_sz)})")
        if COMPRESS_POS_TO_XZ and raw_sz > 0:
            xz_path = compress_to_xz(POS_PATH)
            xz_sz = xz_path.stat().st_size
            print(f"[POS] .xz size      : {xz_sz:,} B ({fmt_bytes(xz_sz)})")
            reduction = 100.0 * (1 - xz_sz / raw_sz)
            print(f"[POS] reduction     : {reduction:.2f}%")
    except FileNotFoundError:
        print("[WARN] positions 文件尚未生成，无法统计大小。")

    print("[DONE] Gray match-only embedding completed.")
