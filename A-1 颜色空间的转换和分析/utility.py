
import cv2
import numpy as np
import math

# 计算相关系数
def corrcoef(data1,data2):
    a = data1.reshape(data1.size)
    b = data2.reshape(data2.size)
    return np.corrcoef(a, b)[0, 1]

# 计算熵,输入w*h矩阵
def entropy(data):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0

    data = np.array(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            val = data[i][j]
            tmp[val] = float(tmp[val] + 1)  #获取每个分量的值，并统计数量
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)  #计算概率
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))   #求熵
    return res

