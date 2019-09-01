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
data = [R,G,B]
data_1D = np.array(data).reshape(3*w*h)



# 压缩
compress,average_code_length = huffman.compress(data_1D)

# 计算熵
entropy = utility.entropy_1D(data_1D)


# 计算编码效率
coding_efficiency = utility.coding_efficiency(entropy,average_code_length)

#解码
decompress = huffman.decompress(compress)

decompress = np.array(decompress, dtype = np.uint8).reshape([3,w,h])

decompress_img = cv2.merge([decompress[0],decompress[1],decompress[2]])

plt.subplot(1,2,1)
plt.title("origin_img")
plt.imshow(origin_img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("decompress_img")
plt.imshow(decompress_img)
plt.axis("off")
plt.show()

print("结果：")
print("熵：",entropy)
print("平均码长：",average_code_length)
print("编码效率：",coding_efficiency)


