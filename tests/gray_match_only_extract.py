# tests/gray_match_only_extract.py
# 测试流程：灰度·仅匹配 提取 -> Zlib+CRC 解包 -> UTF-8 文本落盘
# 说明：
#   - 不使用命令行参数，所有设置写在“配置区”
#   - 若嵌入阶段使用随机比特流或负载被截断，解包将触发校验错误（已做友好提示）

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from core.pipeline_gray_match import gray_match_extract
from core.payload import unpack_to_text

# ===================== 配置区 =====================
STEGO_FILE          = Path("output/BOWS_2/2_gray_match_only.pgm")          # 与嵌入脚本一致
POS_PATH            = Path("output/positions_gray_match_only.txt")  # 与嵌入脚本一致
OUT_BITS_PATH       = Path("output/recovered_bits_gray.txt")
OUT_TEXT_PATH       = Path("output/recovered_text_gray_utf8.txt")
OUT_ERROR_PATH      = Path("output/recovered_text_gray_error.txt")
PREVIEW_HEAD_CHARS  = 400
# =================================================

def main():
    # 1) 从 positions 与 stego 恢复 bitstream
    bits, stats = gray_match_extract(POS_PATH, STEGO_FILE)
    print("[EXTRACT] stats:", stats)

    # 2) 保存原始比特
    OUT_BITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_BITS_PATH.write_text(bits, encoding="utf-8")
    print(f"[SAVE] raw bits -> {OUT_BITS_PATH} (len(bits)={len(bits):,})")

    # 3) Zlib+CRC 解包（若嵌入时是随机比特流，这里会失败）
    try:
        text = unpack_to_text(bits)
    except Exception as e:
        msg = (
            f"[UNPACK][ERROR] {type(e).__name__}: {e}\n"
            "- 可能原因：\n"
            "  1) 嵌入阶段使用了随机比特流（USE_RANDOM_PAYLOAD=True），不是 Zlib+CRC 负载；\n"
            "  2) 负载容量不足导致截断（remain>0），CRC 校验失败；\n"
            "  3) positions 与 stego 不匹配或读取顺序有误。\n"
            "  4) 提取顺序/位点不一致，造成 bitstream 破损。\n"
        )
        OUT_ERROR_PATH.write_text(msg, encoding="utf-8")
        print(msg)
        return

    # 4) 保存文本并做预览
    OUT_TEXT_PATH.write_text(text, encoding="utf-8")
    print(f"[SAVE] recovered text -> {OUT_TEXT_PATH} (chars={len(text):,})")

    preview = text[:PREVIEW_HEAD_CHARS].replace("\n", "\\n")
    print(f"[PREVIEW {min(PREVIEW_HEAD_CHARS, len(text))} chars] {preview}")

if __name__ == "__main__":
    main()
