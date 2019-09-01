# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

import huffman
import utility

# filePath = "lena-512.png"

# # 读取 BMP 文件
# img = cv2.imread(filePath) #默认顺序为 B、G、R
# B = img[:,:,0]
# G = img[:,:,1]
# R = img[:,:,2]



data = np.array([[2,2,3,5,0,0,5,5],
                 [5,4,1,1,2,2,1,5],
                 [4,6,5,5,7,2,2,3],
                 [5,2,2,2,3,4,4,4],
                 [6,2,1,4,1,1,2,2],
                 [1,5,7,6,5,5,7,2],
                 [2,4,4,1,2,2,1,5],
                 [2,3,1,2,2,1,5,0]])
# data  = np.array([[0,1,1,1],
#                 [ 2,2,2,2]])

# data = [1,2,2,3,3,3,4,4,4,4,5,5,5,5,5]
# data = np.arange(1,26).reshape([5,5])
(w,h) = data.shape # 取宽 高

# data = list(data)
data_1D = data.reshape(w*h)  # 转为1维

# data = data.reshape([w,h])

# 压缩
compress_data,average_code_length = huffman.compress(data_1D)

# 计算熵
# entropy = utility.entropy(data)
entropy = utility.entropy_1D(data_1D)

# 计算编码效率
coding_efficiency = utility.coding_efficiency(entropy,average_code_length)

decompress_data = huffman.decompress(compress_data)

print("压缩和解压：")

# print(compress_data)
# print(decompress_data)
print("熵：",entropy)
print("平均码长：",average_code_length)
print("编码效率：",coding_efficiency)
