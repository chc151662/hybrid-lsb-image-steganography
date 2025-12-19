# core/preprocess.py
from typing import Tuple, Optional
import numpy as np
from PIL import Image

# 3.3.1 Cover Image Preparation — BT.601 灰度
# Ig = 0.299*R + 0.587*G + 0.114*B  （报告 3.3.1）
def rgb_to_gray_bt601(img_in) -> np.ndarray:
    """
    将 RGB 图像转为 BT.601 加权灰度 (uint8)。
    支持 PIL.Image 或 np.ndarray 输入；返回 HxW uint8 数组。
    """
    if isinstance(img_in, Image.Image):
        rgb = np.asarray(img_in.convert("RGB"), dtype=np.float64)
    else:
        arr = np.asarray(img_in)
        if arr.ndim == 2:  # 已是灰度
            return arr.astype(np.uint8, copy=False)
        # 强制转 RGB 三通道
        rgb = arr[..., :3].astype(np.float64)
    # BT.601 加权
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray = np.clip(gray, 0, 255).round().astype(np.uint8)
    return gray

def gray_from_channel(image_or_array, channel: str = "R") -> np.ndarray:
    """
    使用单通道(R/G/B)作为灰度图返回（uint8）。
    目的：避免嵌入修改影响边缘检测的一致性。
    """
    if isinstance(image_or_array, Image.Image):
        img = np.array(image_or_array.convert("RGB"), dtype=np.uint8)
    else:
        img = np.asarray(image_or_array, dtype=np.uint8)
        if img.ndim == 2:
            return img  # 已是灰度
    c = {"R": 0, "G": 1, "B": 2}[channel.upper()]
    return img[..., c]


# 3.3.2 Secret Data Preparation — UTF-8 -> bitstream（不做压缩）
def text_to_bitstream_utf8(text: str) -> str:
    """
    将文本按 UTF-8 编码为连续比特流（'0'/'1' 字符串）。
    注意：UTF-8 为变长编码，非 ASCII 字符会占用 >1 字节。
    """
    data = text.encode("utf-8")
    return "".join(f"{b:08b}" for b in data)

def load_text_file_to_bitstream(path: str, max_chars: Optional[int] = None) -> Tuple[str, int]:
    """
    读取文本文件（可选字符截断），输出 (bitstream, char_count)。
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if max_chars is not None:
        txt = txt[:max_chars]
    bitstream = text_to_bitstream_utf8(txt)
    return bitstream, len(txt)

# 3.3.3 Bitstream Validation and Capacity Estimation — 截断策略
# 这里不估计容量本身；容量由 3.4 的边缘坐标 |R(T)| 决定并传入。
def truncate_bitstream_to_capacity(bitstream: str, available_positions: int) -> Tuple[str, bool]:
    """
    根据可用嵌入坐标数（通常为 |R(T)|）对比特流做容量校验与必要的截断。
    返回 (truncated_bitstream, was_truncated)
    """
    if available_positions < 0:
        raise ValueError("available_positions 不能为负数")
    if len(bitstream) <= available_positions:
        return bitstream, False
    return bitstream[:available_positions], True

# 辅助：保存灰度图（便于快速检查预处理是否正确）
def save_gray(gray: np.ndarray, path: str) -> None:
    Image.fromarray(gray, mode="L").save(path)


