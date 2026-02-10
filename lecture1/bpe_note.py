# Unicode :1 ~ 154,997  '牛'


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    print([bytes([b]) for b in bytestring])
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
  # 因为utf-8当中1个字符由1-2个字节（数字）存储,不能单独解码合并, 而且用两个字节编码的数字会有规则，比如第一个字节不能是0xe4之类的（有一个范围）,所以直接单个字节字节的解码一定会出现程序上的报错，不单单是不准确的问题。所以，这里要解码必须在整个字符串层面去解码，这样可以应用一些固定的顺序，规则。所以如果非要从list[int]这个中间节点走一下变回去的话正确的解码应该像下面这样 bytes(list("你好".encode("utf-8"))).decode("utf-8")  

def decode_utf8_bytes_to_str_middleway(bytestring: bytes):
    int_list = list(bytestring)
    bytestring_retransform = bytes(int_list)
    assert bytestring_retransform == bytestring
    return  bytestring_retransform.decode("utf-8")

# 最标准的做法就是下面这样
def decode_utf8_bytes_to_str_straightway(bytestring: bytes):
    return bytestring.decode("utf-8")

# 验证
try:
    res = decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
    print(res)
    assert res == "你好"
    print("函数执行成功")
except Exception as e:
    print(e)
    print("函数执行失败")

print(decode_utf8_bytes_to_str_middleway("你好".encode("utf-8")))
print(decode_utf8_bytes_to_str_straightway("你好".encode("utf-8")))


    
# print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
# print(decode_utf8_bytes_to_str_wrong("111".encode("utf-8")))

# print(b"\x0\x0".decode("utf-8"))
# print(b'\xe4\xbd'.decode("utf-8"))

def  get_btyestr_from_binarry(binary_str):
    byte_val = int(binary_str, base=2)
    print("byte_val:", byte_val)
    byte_str = bytes([byte_val])  # 创建单字节序列
    return byte_str

print(get_btyestr_from_binarry("11000001"))
try:
    print(b"\xc1\xc1".decode("utf-8"))
    print("\xc1\xc1 解码成功")
except:
    print("\xc1\xc1 解码失败")


"""
utf-8 编码对两字节的序列编码有规定  第一个字节的二进制必须是110xxxxx, 第二个必须是10xxxxxx
要构建一个不能被解码的两字节序列
首先第一个字节必须用110xxxxx来表示，这样utf-8解码的时候会把这个理解为两字节序列，用两字节的方式做解码，但是第二个字节如果不满足10xxxxxx就会出错了
"""

# 预分词
print("Pre-Tokenizer")
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
text = "some text that i'll pre-tokenize"
pretokenized_seq = re.findall(PAT, text)
print(pretokenized_seq)

match_iter = re.finditer(PAT, text)
for match_obj in match_iter:
    print(f"text {match_obj.group()}, position {match_obj.span()}")



