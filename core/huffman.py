import heapq
from collections import defaultdict, Counter

class Node:
    def __init__(self, char=None, freq=None):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    """
    构建 Huffman 编码树

    参数:
    - text: 要编码的原始文本

    返回:
    - Huffman 树的根节点
    """
    frequency = Counter(text)
    heap = [Node(ch, freq) for ch, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(freq=node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0] if heap else None

def build_codes(root):
    """
    构建字符到 Huffman 编码的映射表（字典）

    参数:
    - root: Huffman 树根节点

    返回:
    - codes: dict, 每个字符对应的二进制编码，如 {'a': '010', 'b': '11'}
    """
    codes = {}
    def generate_code(node, current=""):
        if node:
            if node.char is not None:
                codes[node.char] = current
            generate_code(node.left, current + "0")
            generate_code(node.right, current + "1")
    generate_code(root)
    return codes

def huffman_encode(text):
    """
    对文本进行 Huffman 编码

    参数:
    - text: str，原始要编码的明文

    返回:
    - encoded: str，仅由 '0' 和 '1' 构成的比特流字符串
    - codes: dict，字符到比特串的映射表（解码时需要）
    """
    root = build_huffman_tree(text)
    codes = build_codes(root)
    encoded = ''.join([codes[ch] for ch in text])
    return encoded, codes

def huffman_decode(encoded, codes):
    """
    对 Huffman 编码后的 bit 字符串进行解码

    参数:
    - encoded: str，Huffman 编码后的比特流
    - codes: dict，编码表（必须与编码时使用的保持一致）

    返回:
    - decoded: str，解码后的明文字符串
    """
    reverse_codes = {v: k for k, v in codes.items()}
    current = ""
    decoded = ""
    for bit in encoded:
        current += bit
        if current in reverse_codes:
            decoded += reverse_codes[current]
            current = ""
    return decoded