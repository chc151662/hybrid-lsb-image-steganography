# tests/fuse_match_then_proposed01_extract.py
# 测试流程：fuse_match_then_proposed01 提取 -> Zlib 解包 -> UTF-8 文本落盘
# 说明：
#   - 不使用命令行参数，所有设置写在“配置区”
#   - 若嵌入阶段使用的是随机比特流或负载被截断，解包将触发校验错误（已做友好提示）

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from core.pipeline_fuse import fuse_extract
from core.payload import unpack_to_text

# ===================== 配置区 =====================
STEGO_FILE          = Path("output/BOWS_2/2.pgm")
POS_PATH            = Path("output/positions_match_bgr_lsb.txt")
OUT_BITS_PATH       = Path("output/recovered_bits.txt")           # 原始'0'/'1'比特
OUT_TEXT_PATH       = Path("output/recovered_text_utf8.txt")      # 解包后的明文
OUT_ERROR_PATH      = Path("output/recovered_text_error.txt")     # 解包失败时的错误信息
PREVIEW_HEAD_CHARS  = 400                                         # 终端打印前几百个字符预览
# =================================================


def main():
    # 1) 执行提取（从 positions 文件反向恢复负载比特串）
    bits, stats = fuse_extract(POS_PATH, STEGO_FILE)
    print("[EXTRACT] stats:", stats)

    # 2) 保存原始比特，便于调试或对照
    OUT_BITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_BITS_PATH.write_text(bits, encoding="utf-8")
    print(f"[SAVE] raw bits -> {OUT_BITS_PATH} (len(bits)={len(bits):,})")

    # 3) Zlib 解包 -> UTF-8 文本
    try:
        text = unpack_to_text(bits)
    except Exception as e:
        # 常见原因提示：a) 嵌入时使用随机比特流；b) 负载容量不足导致截断；c) positions 不匹配
        msg = (
            f"[UNPACK][ERROR] {type(e).__name__}: {e}\n"
            "- 可能原因：\n"
            "  1) 嵌入阶段使用了随机比特流（USE_RANDOM_PAYLOAD=True），不是 Zlib+CRC 负载；\n"
            "  2) 负载被截断（final_remain_bits>0 或容量不足），导致 CRC 校验失败；\n"
            "  3) 提取使用的 positions 文件与嵌入不匹配；\n"
            "  4) 提取顺序/位点不一致，造成 bitstream 破损。\n"
        )
        OUT_ERROR_PATH.write_text(msg, encoding="utf-8")
        print(msg)
        return

    # 4) 成功：保存文本并在终端做一个简短预览
    OUT_TEXT_PATH.write_text(text, encoding="utf-8")
    print(f"[SAVE] recovered text -> {OUT_TEXT_PATH} (chars={len(text):,})")

    preview = text[:PREVIEW_HEAD_CHARS].replace("\n", "\\n")
    print(f"[PREVIEW {min(PREVIEW_HEAD_CHARS, len(text))} chars] {preview}")


if __name__ == "__main__":
    main()
