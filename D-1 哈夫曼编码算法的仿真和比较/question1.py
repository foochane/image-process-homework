# -*- coding: UTF-8 -*-
import numpy as np

import huffman
import utility

# 获取数据
# data = utility.get_uniform_data() # 均匀分布
# data = utility.get_normal_data() # 正态分布
data = utility.get_laplace_data() # 拉普拉斯分布

print("原始数据(前300个):")
print(data[0][:300])

(w,h) = data.shape # 取宽 高

data_1D = data.reshape(w*h)  # 转为1维


# 压缩
compress_data,average_code_length = huffman.compress(data_1D)

# 计算熵
entropy = utility.entropy(data)

# 计算编码效率
coding_efficiency = utility.coding_efficiency(entropy,average_code_length)

#解码
decompress_data = huffman.decompress(compress_data)
decompress_data = np.array(decompress_data).reshape([w,h])

# print("编码数据：")
# print(compress_data[:100])


print("解码数据（前300个）：")
print(decompress_data[0][:300])


print("熵：",entropy)
print("平均码长：",average_code_length)
print("编码效率：",coding_efficiency)