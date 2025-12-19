from typing import List

def generate_fibonacci_upto(n: int) -> List[int]:
    """生成不超过 n 的斐波那契数列（从大到小）"""
    fib = [1, 2]
    while fib[-1] + fib[-2] <= n:
        fib.append(fib[-1] + fib[-2])
    return fib[::-1]    # 反转为降序排列

def to_zeckendorf(n: int, fib_seq: List[int]) -> List[int]:
    """
    使用贪心算法将整数 n 转换为 Zeckendorf 编码（二进制形式）
    规则：不能使用相邻的 Fibonacci 数字
    """
    code = []
    for f in fib_seq:
        if f <= n:
            code.append(1)
            n -= f
        else:
            code.append(0)
    return code

def from_zeckendorf(code: List[int], fib_seq: List[int]) -> int:
    """
    将 Zeckendorf 编码还原为整数
    即 sum(对应 Fibonacci 数 * 1)
    """
    return sum(f for b, f in zip(code, fib_seq) if b)

def embed_fibonacci(pixel_value: int, bit: int) -> int:
    """
    将一个 bit 融入 Zeckendorf Fibonacci 表示的倒数第一位（index=-1）
    并修复不合法的连续两个1（末尾的两位）
    """
    fib_seq = generate_fibonacci_upto(255)
    z_bits = to_zeckendorf(pixel_value, fib_seq)

    # 补齐为 12 位（前面补0）
    if len(z_bits) < 12:
        z_bits = [0] * (12 - len(z_bits)) + z_bits
    
    # print(z_bits) # 补位结果测试

    # 嵌入 secret bit 到倒数第一位
    z_bits[-1] = bit

    # print(z_bits)   # 嵌入结果测试

    # 修复非法情况：若末尾两个 bit 都是 1，不合法
    if z_bits[-2] == 1 and z_bits[-1] == 1:
        z_bits[-2] = 0  # 将倒数第二位修正为0，避免连续1

    # print(z_bits)   # 矫正结果测试

    # 将 Fibonacci 编码还原为整数
    new_value = from_zeckendorf(z_bits, fib_seq)

    # 如果超出 8-bit 像素范围，则放弃嵌入
    return new_value if new_value <= 255 else pixel_value

def embed_fibonacci_pixel(r: int, g: int, b: int, bit: int) -> tuple[int, int, int]:
    """
    在绿色通道中嵌入 Fibonacci 编码的 bit，其余通道保持不变。
    返回新的 (r, g, b)
    """

    g_new = embed_fibonacci(g, bit)
    return r, g_new, b


def extract_fibonacci(pixel_value: int) -> int:
    """
    从像素值中提取 Zeckendorf Fibonacci 编码的倒数第一位作为隐藏信息
    """
    fib_seq = generate_fibonacci_upto(255)
    z_bits = to_zeckendorf(pixel_value, fib_seq)

    # 补齐为 12 位（在前面补0）
    if len(z_bits) < 12:
        z_bits = [0] * (12 - len(z_bits)) + z_bits

    # 提取倒数第一位（被 embed_fibonacci 用来嵌入 bit）
    return z_bits[-1]