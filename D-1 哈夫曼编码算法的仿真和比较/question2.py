# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

import huffman
import utility

# 获取数据
# path = "lena-512.png"
# path = "cat-512.png"
path = "Sandhill-Crane-512.png"
R,G,B = utility.get_img_data(path) 
origin_img = cv2.merge([R,G,B])




(w,h) = R.shape # 取宽 高

R_1D = R.reshape(w*h)  # 转为1维
G_1D = G.reshape(w*h)  # 转为1维
B_1D = B.reshape(w*h)  # 转为1维


# 压缩
compress_R,average_code_length_R = huffman.compress(R_1D)
compress_G,average_code_length_G = huffman.compress(G_1D)
compress_B,average_code_length_B = huffman.compress(B_1D)

# 计算熵
entropy_R = utility.entropy(R)
entropy_G = utility.entropy(G)
entropy_B = utility.entropy(B)

# 计算编码效率
coding_efficiency_R = utility.coding_efficiency(entropy_R,average_code_length_R)
coding_efficiency_G = utility.coding_efficiency(entropy_G,average_code_length_G)
coding_efficiency_B = utility.coding_efficiency(entropy_B,average_code_length_B)

#解码
decompress_R = huffman.decompress(compress_R)
decompress_G = huffman.decompress(compress_G)
decompress_B = huffman.decompress(compress_B)

decompress_R = np.array(decompress_R, dtype = np.uint8).reshape([w,h])
decompress_G = np.array(decompress_G, dtype = np.uint8).reshape([w,h])
decompress_B = np.array(decompress_B, dtype = np.uint8).reshape([w,h])

decompress_img = cv2.merge([decompress_R,decompress_G,decompress_B])

plt.subplot(1,2,1)
plt.title("origin_img")
plt.imshow(origin_img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("decompress_img")
plt.imshow(decompress_img)
plt.axis("off")
plt.show()

print("R分量：")
print("熵(R)：",entropy_R)
print("平均码长(R)：",average_code_length_R)
print("编码效率(R)：",coding_efficiency_R)

print("G分量：")
print("熵(G)：",entropy_G)
print("平均码长(G)：",average_code_length_G)
print("编码效率(G)：",coding_efficiency_G)

print("B分量：")
print("熵(B)：",entropy_B)
print("平均码长(B)：",average_code_length_B)
print("编码效率(B)：",coding_efficiency_B)


