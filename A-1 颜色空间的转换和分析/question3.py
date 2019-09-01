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

#RGB2YUV
Y1, U, V = convert.RGB2YUV(R, G, B)
Y1 = np.array(Y1, dtype = np.uint8)
U = np.array(U, dtype = np.uint8)
V = np.array(V, dtype = np.uint8)


#YUV2YIQ
Y2, I, Q = convert.YUV2YIQ(Y1, U, V)
Y2 = np.array(Y2, dtype = np.uint8)
I = np.array(I, dtype = np.uint8)
Q = np.array(Q, dtype = np.uint8)
img_YIQ = cv2.merge([Y2, I, Q]) 
img_Y2 = cv2.merge([Y2,zeros,zeros])
img_I = cv2.merge([zeros,I,zeros])
img_Q = cv2.merge([zeros,zeros,Q])
plt.suptitle("YUV2YIQ")
plt.subplot(2,2,1)
plt.imshow(img_YIQ)
plt.axis("off")#去除坐标轴
plt.title("img_YIQ")
plt.subplot(2,2,2)
plt.imshow(img_Y2)
plt.axis("off")#去除坐标轴
plt.title("img_Y2")
plt.subplot(2,2,3)
plt.imshow(img_I)
plt.axis("off")#去除坐标轴
plt.title("img_I")
plt.subplot(2,2,4)
plt.imshow(img_Q)
plt.axis("off")#去除坐标轴
plt.title("img_Q")
plt.show()

#YIQ2YUV
Y1, U, V = convert.YIQ2YUV(Y2, I, Q)
Y1 = np.array(Y1, dtype = np.uint8)
U = np.array(U, dtype = np.uint8)
V = np.array(V, dtype = np.uint8)
img_YUV = cv2.merge([Y1, U, V]) 
img_Y = cv2.merge([Y1,zeros,zeros])
img_U = cv2.merge([zeros,U,zeros])
img_V = cv2.merge([zeros,zeros,V])
plt.suptitle("YIQ2YUV")
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

#YUV2YCbCr
Y3, Cb, Cr = convert.YUV2YCbCr(Y1, U, V)
Y3 = np.array(Y3, dtype = np.uint8)
Cb = np.array(Cb, dtype = np.uint8)
Cr = np.array(Cr, dtype = np.uint8)
img_YCbCr = cv2.merge([Y2, I, Q]) 
img_Y3 = cv2.merge([Y3,zeros,zeros])
img_Cb = cv2.merge([zeros,Cb,zeros])
img_Cr = cv2.merge([zeros,zeros,Cr])
plt.suptitle("YUV2YCbCr")
plt.subplot(2,2,1)
plt.imshow(img_YCbCr)
plt.axis("off")#去除坐标轴
plt.title("img_YCbCr")
plt.subplot(2,2,2)
plt.imshow(img_Y3)
plt.axis("off")#去除坐标轴
plt.title("img_Y3")
plt.subplot(2,2,3)
plt.imshow(img_Cb)
plt.axis("off")#去除坐标轴
plt.title("img_Cb")
plt.subplot(2,2,4)
plt.imshow(img_Cr)
plt.axis("off")#去除坐标轴
plt.title("img_Cr")
plt.show()

#YCbCr2YUV
Y1, U, V = convert.YCbCr2YUV(Y3, Cb, Cr)
Y1 = np.array(Y1, dtype = np.uint8)
U = np.array(U, dtype = np.uint8)
V = np.array(V, dtype = np.uint8)
img_YUV = cv2.merge([Y1, U, V]) 
img_Y = cv2.merge([Y1,zeros,zeros])
img_U = cv2.merge([zeros,U,zeros])
img_V = cv2.merge([zeros,zeros,V])
plt.suptitle("YCbCr2YUV")
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

#YIQ2YCbCr
Y3, Cb, Cr = convert.YIQ2YCbCr(Y2, I, Q)
Y3 = np.array(Y3, dtype = np.uint8)
Cb = np.array(Cb, dtype = np.uint8)
Cr = np.array(Cr, dtype = np.uint8)
img_YCbCr = cv2.merge([Y2, I, Q]) 
img_Y3 = cv2.merge([Y3,zeros,zeros])
img_Cb = cv2.merge([zeros,Cb,zeros])
img_Cr = cv2.merge([zeros,zeros,Cr])
plt.suptitle("YIQ2YCbCr")
plt.subplot(2,2,1)
plt.imshow(img_YCbCr)
plt.axis("off")#去除坐标轴
plt.title("img_YCbCr")
plt.subplot(2,2,2)
plt.imshow(img_Y3)
plt.axis("off")#去除坐标轴
plt.title("img_Y3")
plt.subplot(2,2,3)
plt.imshow(img_Cb)
plt.axis("off")#去除坐标轴
plt.title("img_Cb")
plt.subplot(2,2,4)
plt.imshow(img_Cr)
plt.axis("off")#去除坐标轴
plt.title("img_Cr")
plt.show()


#YCbCr2YIQ
Y2, I, Q = convert.YCbCr2YIQ(Y3,Cb,Cr)
Y2 = np.array(Y2, dtype = np.uint8)
I = np.array(I, dtype = np.uint8)
Q = np.array(Q, dtype = np.uint8)
img_YIQ = cv2.merge([Y2, I, Q]) 
img_Y2 = cv2.merge([Y2,zeros,zeros])
img_I = cv2.merge([zeros,I,zeros])
img_Q = cv2.merge([zeros,zeros,Q])
plt.suptitle("YCbCr2YIQ")
plt.subplot(2,2,1)
plt.imshow(img_YIQ)
plt.axis("off")#去除坐标轴
plt.title("img_YIQ")
plt.subplot(2,2,2)
plt.imshow(img_Y2)
plt.axis("off")#去除坐标轴
plt.title("img_Y2")
plt.subplot(2,2,3)
plt.imshow(img_I)
plt.axis("off")#去除坐标轴
plt.title("img_I")
plt.subplot(2,2,4)
plt.imshow(img_Q)
plt.axis("off")#去除坐标轴
plt.title("img_Q")
plt.show()


#计算相关系数
print("相关系数(Y1,Y2):",utility.corrcoef(Y1,Y2))
print("相关系数(Y1,Y3):",utility.corrcoef(Y1,Y3))
print("相关系数(Y2,Y3):",utility.corrcoef(Y2,Y3))

print("相关系数(U,I):",utility.corrcoef(U,I))
print("相关系数(U,Cb):",utility.corrcoef(U,Cb))
print("相关系数(I,Cb):",utility.corrcoef(I,Cb))

print("相关系数(V,Q):",utility.corrcoef(V,Q))
print("相关系数(V,Cr):",utility.corrcoef(V,Cr))
print("相关系数(Q,Cb):",utility.corrcoef(Q,Cb))