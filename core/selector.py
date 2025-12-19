# core/selector.py
from core.fibonacci import embed_fibonacci_pixel, extract_fibonacci
from core.xnor import embed_xnor_pixel, extract_xnor_pixel
from PIL import Image
import numpy as np

def embed_with_pixel_selection(input_path: str, bitstream: str, output_path: str):
    """
    根据对角线分割策略，对图像应用两种不同的嵌入算法（Fibonacci / XNOR）。
    """

    img = np.array(Image.open(input_path))
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3).copy()

    embedded_count = 0
    for y in range(h):
        for x in range(w):
            if embedded_count >= len(bitstream):
                break

            if y == x:
                continue  # 分割线：跳过

            bit = int(bitstream[embedded_count])
            
            r, g, b = img[y, x]

            if y < x:
                # 上三角：使用 XNOR
                # print(embedded_count, y, x, bit, "xnor") # 打印测试
                r, g, b = embed_xnor_pixel(r, g, b, bit) 
                
            else:
                # 下三角：使用 Fibonacci
                # print(embedded_count, y, x, bit, "fib") # 打印测试
                r, g, b = embed_fibonacci_pixel(r, g, b, bit)

            img[y, x] = [r, g, b]
            embedded_count += 1

    Image.fromarray(img).save(output_path)
    print(f"[INFO] 嵌入完成，共嵌入 {embedded_count} 位，保存为: {output_path}")

def extract_with_pixel_selection(image_path: str, length: int) -> str:
    """
    从图像中提取隐藏的比特流，按照对角线分割策略：
    - 上三角：使用 XNOR 提取（G 或 B 通道，依赖 R.MSB）
    - 下三角：使用 Fibonacci 提取（G 通道）
    - 对角线：跳过
    """
    img = np.array(Image.open(image_path))
    h, w = img.shape[:2]

    extracted_bits = []
    for y in range(h):
        for x in range(w):
            if len(extracted_bits) >= length:
                break

            if y == x:
                continue  # 跳过对角线

            r, g, b = img[y, x]

            if y < x:
                # 上三角：XNOR 提取
                bit = extract_xnor_pixel(r, g, b)
            else:
                # 下三角：Fibonacci 提取
                bit = extract_fibonacci(g)

            extracted_bits.append(str(bit))

    return ''.join(extracted_bits)
