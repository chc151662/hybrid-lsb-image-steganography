# 负责 positions.tsv 的标准化读写（含头部元数据）
# 提供：save_positions(), parse_positions_header(), parse_positions_rows()
# core/positions.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# ---- 写入（头部 + 行级）----
def save_positions(
    path: Path,
    header: Dict[str, str],
    rows: Iterable[Tuple[int,int,int,str,int]],  # (y,x,bit,channel,bit_plane)
    width: int,
    note: str = "match_BGR_then_maybe_proposed01",
    position_type: str = "non-edge",
):
    """
    header: 建议包含这些键（字符串化写入）：
      image, 
      stego_image, 
      size(HxW), 
      match_threshold_notR, 
      bit_planes_used,                 # 例如 "0,1" 或 "0-2"
      payload_total_bits, 
      matched_bits_in_match_phase, 
      total_positions_in_non-edge,
      need_proposed01, 
      proposed_minimum, 
      remaining_bits_to_embed_by_proposed01,
      （可选）pipeline, proposed_threshold_R, selector, seed 等
    """
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        # f.write("# Image Steganography Position Map\n")
        for k, v in header.items():
            f.write(f"# {k}: {v}\n")
        # f.write("# -----------------------------------------------------\n")
        f.write("y\tx\tch\tbp\n")
        for (y, x, b, ch, bp) in rows:
            # flat = y * width + x
            # threshold = header.get("match_threshold_notR", "NA")
            # f.write(f"{flat}\t{y}\t{x}\t{ch}\t{bp}\t{b}\t{position_type}\t{threshold}\t{note}\n")
            f.write(f"{y}\t{x}\t{ch}\t{bp}\n")

# ---- 读取（头部）----
def parse_positions_header(path: Path) -> Dict[str, str]:
    import re
    header: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*([^:]+):\s*(.*)$", line.strip())
            if m:
                header[m.group(1).strip()] = m.group(2).strip()
    return header

# ---- 读取（行级）----
def parse_positions_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str,str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0].lower() == "y": # 非常重要的一行代码，需要匹配到"y"这个字符，再往下找
                continue
            # if len(parts) < 9:
            #     raise ValueError(f"TSV 字段数不足 9：{parts}")
            if len(parts) < 4:
                raise ValueError(f"TSV 字段数不足 4：{parts}")
            rows.append({
                # "index": parts[0], 
                "y": parts[0], 
                "x": parts[1],
                "channel": parts[2], 
                "bit_plane": parts[3], 
                # "bit": parts[5],
                # "position_type": parts[6], 
                # "threshold": parts[7], 
                # "note": parts[8],
            })
    return rows

# ===== 灰度专用：写入（头部 + 行级 y x bp） =====
def save_positions_gray(
    path: Path,
    header: Dict[str, str],
    rows: Iterable[Tuple[int, int, int]],   # (y, x, bit_plane)
) -> None:
    """
    灰度 positions：每行仅写 (y, x, bp)，无 channel/bit 字段。
    header 建议包含：image, stego_image, size(HxW), match_threshold_notR,
                    payload_total_bits, matched_bits_in_match_phase, pipeline 等。
    """
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        # 头部
        for k, v in header.items():
            f.write(f"# {k}: {v}\n")
        # 表头（3列）
        f.write("y\tx\tbp\n")
        # 行级
        for (y, x, bp) in rows:
            f.write(f"{int(y)}\t{int(x)}\t{int(bp)}\n")


# ===== 灰度专用：读取（行级 y x bp） =====
def parse_positions_rows_gray(path: Path) -> List[Dict[str, str]]:
    """
    读取灰度版 positions（3列：y x bp）。返回的每行 dict 仅包含：
      - "y", "x", "bit_plane"
    注意：不包含 channel；若你下游代码需要统一键名，可默认当作 'Y' 使用。
    """
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            # 跳过表头
            if parts[0].lower() == "y":
                continue
            if len(parts) < 3:
                raise ValueError(f"TSV 字段数不足 3：{parts}")
            rows.append({
                "y": parts[0],
                "x": parts[1],
                "bit_plane": parts[2],
            })
    return rows
