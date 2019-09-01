# -*- coding: UTF-8 -*-
import numpy as np
import sys
import cv2
import math
import convert
import matplotlib.pyplot as plt
import utility

filePath = "lena.bmp"
# filePath = "Sandhill-Crane-256.png"

# 读取 BMP 文件
img = cv2.imread(filePath) #默认顺序为 B、G、R
B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]

zeros = np.zeros(R.shape, dtype = np.uint8)

img_RGB = cv2.merge([R,G,B]) 
img_R = cv2.merge([R,zeros,zeros])
img_G = cv2.merge([zeros,G,zeros])
img_B = cv2.merge([zeros,zeros,B])

plt.subplot(2,2,1)
plt.imshow(img_RGB)
plt.axis("off")#去除坐标轴
plt.title("img_RGB")
plt.subplot(2,2,2)
plt.imshow(img_R)
plt.axis("off")#去除坐标轴
plt.title("img_G")
plt.subplot(2,2,3)
plt.imshow(img_G)
plt.axis("off")#去除坐标轴
plt.title("img_G")
plt.subplot(2,2,4)
plt.imshow(img_B)
plt.axis("off")#去除坐标轴
plt.title("img_B")
plt.show()

#YUV
Y1, U, V = convert.RGB2YUV(R, G, B)
Y1 = np.array(Y1, dtype = np.uint8)
U = np.array(U, dtype = np.uint8)
V = np.array(V, dtype = np.uint8)
img_YUV = cv2.merge([Y1, U, V]) 
img_Y = cv2.merge([Y1,zeros,zeros])
img_U = cv2.merge([zeros,U,zeros])
img_V = cv2.merge([zeros,zeros,V])
plt.subplot(2,2,1)
plt.imshow(img_YUV)
plt.axis("off")#去除坐标轴
plt.title("img_YUV")
plt.subplot(2,2,2)
plt.imshow(img_Y)
plt.axis("off")#去除坐标轴
plt.title("img_Y")
plt.subplot(2,2,3)
plt.imshow(img_U)
plt.axis("off")#去除坐标轴
plt.title("img_U")
plt.subplot(2,2,4)
plt.imshow(img_V)
plt.axis("off")#去除坐标轴
plt.title("img_V")
plt.show()



# YIQ
Y2, I, Q = convert.RGB2YIQ(R, G, B)
for row in range(len(Y2)) :
    for col in range(len(Y2[row])) :
        Y2[row][col] = np.int8(Y2[row][col])
        I[row][col] = np.int8(I[row][col] + 128)
        Q[row][col] = np.int8(Q[row][col] + 128)
Y2 = np.array(Y2, dtype = np.uint8)
I = np.array(I, dtype = np.uint8)
Q = np.array(Q, dtype = np.uint8)
img_YIQ = cv2.merge([Y2, I, Q]) 
img_Y = cv2.merge([Y2,zeros,zeros])
img_I = cv2.merge([zeros,I,zeros])
img_Q = cv2.merge([zeros,zeros,Q])
plt.subplot(2,2,1)
plt.imshow(img_YIQ)
plt.axis("off")#去除坐标轴
plt.title("img_YIQ")
plt.subplot(2,2,2)
plt.imshow(img_Y)
plt.axis("off")#去除坐标轴
plt.title("img_Y")
plt.subplot(2,2,3)
plt.imshow(img_I)
plt.axis("off")#去除坐标轴
plt.title("img_I")
plt.subplot(2,2,4)
plt.imshow(img_Q)
plt.axis("off")#去除坐标轴
plt.title("img_Q")
plt.show()

# YCbCr
Y3, Cb, Cr = convert.RGB2YCbCr(R, G, B)
Y3 = np.array(Y3, dtype = np.uint8)
Cb = np.array(Cb, dtype = np.uint8)
Cr = np.array(Cr, dtype = np.uint8)
img_YCbCr = cv2.merge([Y3, Cb, Cr])
img_Y = cv2.merge([Y3,zeros,zeros])
img_Cb = cv2.merge([zeros,Cb,zeros])
img_Cr = cv2.merge([zeros,zeros,Cr])
plt.subplot(2,2,1)
plt.imshow(img_YCbCr)
plt.axis("off")#去除坐标轴
plt.title("img_YCbCr")
plt.subplot(2,2,2)
plt.imshow(img_Y)
plt.axis("off")#去除坐标轴
plt.title("img_Y")
plt.subplot(2,2,3)
plt.imshow(img_Cb)
plt.axis("off")#去除坐标轴
plt.title("img_Cb")
plt.subplot(2,2,4)
plt.imshow(img_Cr)
plt.axis("off")#去除坐标轴
plt.title("img_Cr")
plt.show()

#计算相关系数和熵
print("RGB:")
print("相关系数(R,G):",utility.corrcoef(R,G))
print("相关系数(R,B):",utility.corrcoef(R,B))
print("相关系数(G,B):",utility.corrcoef(G,B))
print("熵(R):",utility.entropy(R))
print("熵(G):",utility.entropy(G))
print("熵(B):",utility.entropy(B))

print("YUV:")
print("相关系数(Y,U):",utility.corrcoef(Y1,U))
print("相关系数(Y,V):",utility.corrcoef(Y1,V))
print("相关系数(U,V):",utility.corrcoef(U,V))
print("熵(Y):",utility.entropy(Y1))
print("熵(U):",utility.entropy(U))
print("熵(V):",utility.entropy(V))

print("YIQ:")
print("相关系数(Y,I):",utility.corrcoef(Y2,I))
print("相关系数(Y,Q):",utility.corrcoef(Y2,Q))
print("相关系数(I,Q):",utility.corrcoef(I,Q))
print("熵(Y):",utility.entropy(Y2))
print("熵(I):",utility.entropy(I))
print("熵(Q):",utility.entropy(Q))

print("YCbCr:")
print("相关系数(Y,Cb):",utility.corrcoef(Y3,Cb))
print("相关系数(Y,Cr):",utility.corrcoef(Y3,Cr))
print("相关系数(Cb,Cr):",utility.corrcoef(Cb,Cr))
print("熵(Y):",utility.entropy(Y3))
print("熵(Cb):",utility.entropy(Cb))
print("熵(Cr):",utility.entropy(Cr))