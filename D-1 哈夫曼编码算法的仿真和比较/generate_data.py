import numpy as np

# 均匀分布
data1 = np.random.uniform(0,255,1920*1080)
data1 = np.array(data1, dtype = np.uint8).reshape(1920,1080)
np.savetxt("data1.txt", data1)

#正态分布
data2 = np.random.randn(1920*1080)*10
data2 = np.array(data2, dtype = np.uint8).reshape(1920,1080)
np.savetxt("data2.txt", data2)

# 拉普拉斯分布
data3 = np.random.laplace(0,1,1920*1080)*10
data3 = np.array(data3, dtype = np.uint8).reshape(1920,1080)
np.savetxt("data3.txt", data3)