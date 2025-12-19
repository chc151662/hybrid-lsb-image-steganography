import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gzip, bz2, lzma, zipfile, os
from pathlib import Path

# 原始文件路径
src = Path("output/positions_match_bgr_lsb.txt")
data = src.read_bytes()
orig_size = len(data)

def fmt(n):
    for u in ["B","KB","MB","GB"]:
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} TB"

results = []

# --- gzip ---
gz_path = src.with_suffix(".txt.gz")
with gzip.open(gz_path, "wb", compresslevel=9) as f:
    f.write(data)
results.append(("gzip (.gz)", gz_path.stat().st_size))

# --- bzip2 ---
bz2_path = src.with_suffix(".txt.bz2")
with bz2.open(bz2_path, "wb", compresslevel=9) as f:
    f.write(data)
results.append(("bzip2 (.bz2)", bz2_path.stat().st_size))

# --- xz (LZMA) ---
xz_path = src.with_suffix(".txt.xz")
with lzma.open(xz_path, "wb", preset=9) as f:
    f.write(data)
results.append(("xz (.xz)", xz_path.stat().st_size))

# --- zip ---
zip_path = src.with_suffix(".zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    zf.write(src, arcname=src.name)
results.append(("zip (.zip)", zip_path.stat().st_size))

# 汇总结果
print(f"原始文件大小: {fmt(orig_size)}\n")
for name, size in results:
    ratio = size / orig_size * 100
    print(f"{name:<15}: {fmt(size):>10}  ({ratio:5.2f}% of original)")

# 删除压缩文件（如不需保留可启用）
# for _, path in [("gz", gz_path), ("bz2", bz2_path), ("xz", xz_path), ("zip", zip_path)]:
#     path.unlink(missing_ok=True)
