from PIL import Image
import numpy as np

def xnor(a, b):
    """
    执行位级的 XNOR 操作，返回 a XNOR b 的结果（二进制为1或0）

    参数:
    - a: 第一个位（0或1）
    - b: 第二个位（0或1）

    返回:
    - 结果: a XNOR b 的结果，范围为 {0, 1}
    """
    return ~(a ^ b) & 1

def embed_xnor(image_path, message, output_path):
    """
    将 message 使用 XNOR 操作嵌入到 image_path 指定的图像中，并将结果保存为 output_path。

    参数:
    - image_path: 待嵌入的原始图像路径（应为RGB图）
    - message: 要嵌入的秘密文本信息
    - output_path: 嵌入后生成的 stego 图像保存路径

    操作说明:
    - 将消息转换为二进制字符串，每个字符8位，并加上终止符 ‘00000000’
    - 遍历图像的每一个像素，以 XNOR(红通道的 MSB, 消息位) 作为结果写入 绿通道或者蓝通道的 LSB
    - 红通道 MSB (最高有效位) 决定嵌入行为的 key，绿通道或者蓝通道 LSB 是嵌入位
    """
    img = Image.open(image_path).convert("RGB")
    data = np.array(img)
    bits = message  # 不附加终止符

    h, w, _ = data.shape # shape: (H, W, 3)
    flat_idx = 0 # 当前嵌入第几个比特
 
    for y in range(h):
        for x in range(w):
            if flat_idx >= len(bits):
                break
            
            r, g, b = data[y, x]
            msb_r = (r >> 7) & 1       # 获取红通道的 MSB
            bit = int(bits[flat_idx])  # 当前要嵌入的比特

            result = xnor(msb_r, bit)  # 执行 XNOR 操作
            
            if msb_r == 1:
                # 如果红通道 MSB 为 1，嵌入绿色通道的 LSB
                g = np.uint8((int(g) & ~1) | int(result))
            else:
                # 否则嵌入蓝色通道的 LSB
                # b = np.uint8((b & ~1) | result)
                b = np.uint8((int(b) & ~1) | int(result))

            data[y, x] = [r, g, b]
            flat_idx += 1


    # 保存 stego 图像
    Image.fromarray(data).save(output_path)
    print(f"[INFO] XNOR stego image saved to: {output_path}")

def embed_xnor_pixel(r: int, g: int, b: int, bit: int) -> tuple[int, int, int]:
    """
    根据 r 通道的 MSB 选择绿色或蓝色通道嵌入 bit，返回修改后的 r,g,b。
    """
    msb_r = (r >> 7) & 1
    result = xnor(msb_r, bit)

    if msb_r == 1:
        # 嵌入到绿色通道 LSB
        # g = (g & ~1) | result
        g = np.uint8((int(g) & ~1) | int(result))
    else:
        # 嵌入到蓝色通道 LSB
        # b = (b & ~1) | result
        b = np.uint8((int(b) & ~1) | int(result))

    return r, g, b

def extract_xnor_pixel(r: int, g: int, b: int) -> int:
    """
    从图像像素中提取隐藏比特：
    - MSB(r) == 1 → 从 G 的 LSB 提取
    - MSB(r) == 0 → 从 B 的 LSB 提取
    - 用反推 XNOR 的方式恢复 bit
    """
    msb_r = (r >> 7) & 1
    if msb_r == 1:
        lsb = g & 1
    else:
        lsb = b & 1

    bit = ~(msb_r ^ lsb) & 1    # 该写法更直观
    return bit


def extract_xnor(image_path, length):
    """
    从 XNOR 隐写图像中提取指定长度的比特流。

    参数:
    - image_path: stego 图像路径
    - length: 要提取的比特数量（注意单位是 bit）

    返回:
    - 比特流字符串（如 "10010101..."）
    """
    img = Image.open(image_path).convert("RGB")
    data = np.array(img)
    h, w, _ = data.shape
    bits = []

    flat_idx = 0
    for y in range(h):
        for x in range(w):
            if flat_idx >= length:
                break

            r, g, b = data[y, x]
            msb_r = (r >> 7) & 1  # 红通道 MSB

            # 根据 MSB 选择读取哪个通道的 LSB
            if msb_r == 1:
                lsb = g & 1  # 从绿色通道提取
            else:
                lsb = b & 1  # 从蓝色通道提取

            # 逆推原始比特
            bit = msb_r ^ (~lsb & 1)
            bits.append(str(bit))

            flat_idx += 1

    return ''.join(bits)


