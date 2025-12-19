# core/payload.py  (no header, no CRC)
from __future__ import annotations
import zlib

# ---------- 基础工具 ----------
def bytes_to_bits(b: bytes) -> str:
    """bytes -> '0'/'1' 比特串（高位在前）。"""
    return "".join(f"{x:08b}" for x in b)

def bits_to_bytes(bits: str) -> bytes:
    """'0'/'1' 比特串 -> bytes。长度必须是 8 的倍数。"""
    if len(bits) % 8 != 0:
        raise ValueError("bitstream length must be a multiple of 8")
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))


# ---------- 核心 API（无头部、无CRC，仅zlib流） ----------
def pack_text(text: str, level: int = 6) -> str:
    """
    文本 -> UTF-8 -> zlib(level) -> 比特串。
    无任何头部/校验字段，便于精确控制压缩后大小。
    :return: '0'/'1' 比特串
    """
    raw = text.encode("utf-8")
    comp = zlib.compress(raw, level=level)
    return bytes_to_bits(comp)

def unpack_to_text(bits: str) -> str:
    """
    从比特串还原：zlib 解压 -> UTF-8 解码。
    由于无CRC与长度字段，若提取顺序错误或比特错误，可能抛出 zlib.error。
    """
    comp = bits_to_bytes(bits)
    raw = zlib.decompress(comp)  # 可能抛出 zlib.error
    return raw.decode("utf-8", errors="strict")

# ---------- 可选：字节级便捷函数 ----------
def pack_bytes(data: bytes, level: int = 6) -> str:
    """任意字节 -> zlib 压缩 -> 比特串。"""
    return bytes_to_bits(zlib.compress(data, level=level))

def unpack_to_bytes(bits: str) -> bytes:
    """比特串 -> zlib 解压后的字节。"""
    return zlib.decompress(bits_to_bytes(bits))

# ---------- 自检 ----------
if __name__ == "__main__":
    msg = "Hello, stego! 你好，隐写！"
    bits = pack_text(msg, level=6)
    back = unpack_to_text(bits)
    print("ok:", back == msg, "len(bits)=", len(bits))
