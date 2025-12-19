# tests/visualize_lena_non_edge.py
# 生成 Cover–Stego Matching 准备阶段的三张效果图：
# 1) 原始 Lena RGB 图
# 2) BT.601 灰度图
# 3) 非边缘区域 mask 图（白 = non-edge, 黑 = edge），T1 = 100

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
from PIL import Image

from core.match_bgr import prepare_bt601_grad_and_masks  # 复用你现有的接口

# ===================== 配置区 =====================
# 注意：如果你的 Lena 路径/文件名不同，这里改一下就行
IMAGE_PATH = Path("dataset/base/Peppers.bmp")      # 输入的 Lena.bmp
OUT_DIR    = Path("output/visualized/peppers_non_edge_vis")  # 输出目录

T_NOTR = 100   # 与论文中的 T1 一致

# ===================== 主流程 =====================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 读入 Lena 并保存“原始 RGB 图”
    img_rgb = Image.open(IMAGE_PATH).convert("RGB")
    out_rgb = OUT_DIR / "step1_lena_rgb.png"
    img_rgb.save(out_rgb)
    print(f"[OK] Saved RGB image to {out_rgb}")

    # 2) 使用项目现有的 BT.601 + Sobel 函数，拿到灰度和 mask
    # 返回值约定：gray_bt, grad, R_mask, notR_mask, ys, xs
    gray_bt, grad, R_mask, notR_mask, ys, xs = prepare_bt601_grad_and_masks(
        img_rgb, T_NOTR
    )

    # 2.1 保存灰度图（如果是 ndarray，就转成 PIL Image）
    if isinstance(gray_bt, np.ndarray):
        # 确保是 uint8
        if gray_bt.dtype != np.uint8:
            gray_bt_norm = np.clip(gray_bt, 0, 255).astype(np.uint8)
        else:
            gray_bt_norm = gray_bt
        img_gray = Image.fromarray(gray_bt_norm, mode="L")
    else:
        # 如果 prepare_bt601_grad_and_masks 直接返回的是 PIL Image
        img_gray = gray_bt.convert("L")

    out_gray = OUT_DIR / "step2_lena_gray_bt601.png"
    img_gray.save(out_gray)
    print(f"[OK] Saved grayscale image to {out_gray}")

    # 3) 生成非边缘区域 mask 图：白 = non-edge, 黑 = edge
    # notR_mask 应该是 bool/0-1 的二维数组
    notR_mask = np.asarray(notR_mask)
    if notR_mask.dtype != np.bool_:
        notR_bool = notR_mask.astype(bool)
    else:
        notR_bool = notR_mask

    # 白色表示 non-edge 区域，黑色表示 edge
    mask_non_edge = (notR_bool.astype(np.uint8)) * 255  # 0 或 255
    img_mask = Image.fromarray(mask_non_edge, mode="L")

    out_mask = OUT_DIR / f"step3_non_edge_mask_T{T_NOTR}.png"
    img_mask.save(out_mask)
    print(f"[OK] Saved non-edge mask image to {out_mask}")

    print("\nDone. You now have three images:")
    print(f"  1) {out_rgb.name}   (RGB original)")
    print(f"  2) {out_gray.name}  (BT.601 grayscale)")
    print(f"  3) {out_mask.name}  (non-edge mask, white = non-edge, black = edge)")


if __name__ == "__main__":
    main()
