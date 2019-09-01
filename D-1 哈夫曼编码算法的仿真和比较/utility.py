
import cv2
import numpy as np
import math

def get_img_data(path):
    img = cv2.imread(path) #默认顺序为 B、G、R
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    return R,G,B

# 均匀分布
def get_uniform_data():
    data = np.random.uniform(0,255,1920*1080)
    data = np.array(data, dtype = np.uint8).reshape(1920,1080)
    # np.savetxt("data.txt", data)
    return data

#正态分布
def get_normal_data():
    data = np.random.randn(1920*1080)*10
    data = np.array(data, dtype = np.uint8).reshape(1920,1080)
    return data

#拉普拉斯分布
def get_laplace_data():
    data = np.random.laplace(0,1,1920*1080)*10
    data = np.array(data, dtype = np.uint8).reshape(1920,1080)
    return data

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

# 计算熵,输入矩阵
def entropy_1D(data):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0

    data = np.array(data)
    for i in range(len(data)):
        val = data[i]
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

#计算编码效率 熵/平均码长
def coding_efficiency(H,L):
    return H/L