import cv2
import numpy as np
import matplotlib.pyplot as plt

def dct(data,block_size):
    m,n = data.shape  
    # block_size*block_size 的块
    hdata = np.vsplit(data,n//block_size) # 垂直分成高度度为block_size的块（按行分割）
    # 第i行
    dct = None
    idct = None
    for i in range(0, n//block_size):
        block_rows = None
        iblock_row = None
        blockdata = np.hsplit(hdata[i],m//block_size)  #（按列分割）
        for j in range(0, m//block_size): # 第i行，第j列个块
            block = cv2.dct(blockdata[j].astype(np.float))
            iblock = cv2.idct(block)
            if j == 0:
                block_rows = block
                iblock_rows = iblock
            else:
                block_rows = np.hstack((block_rows,block))  # 按列合并
                iblock_rows = np.hstack((iblock_rows,iblock))
        if i == 0:
            dct = block_rows
            idct = iblock_rows
        else:
            dct = np.vstack((dct,block_rows))  # 按行合并
            idct = np.vstack((idct,iblock_rows))
    return np.array(dct,dtype=np.uint8),np.array(idct,dtype=np.uint8)

def dct_image(img_data,block_size):
    dct_R,idct_R = dct(img_data[0],block_size)
    dct_G,idct_G = dct(img_data[1],block_size)
    dct_B,idct_B = dct(img_data[2],block_size)
    dct_img = cv2.merge([dct_R,dct_G,dct_B])
    idct_img = cv2.merge([idct_R,idct_G,idct_B])
    return dct_img,idct_img

def get_image_data(filePath):
    img = cv2.imread(filePath)
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    return [R,G,B]

def corrcoef(img_data):
    data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    data = data.reshape(data.size)
    a = data[::2]
    b = data[1::2]   
    return np.corrcoef(a, b)[0, 1]



# main
block_size = 4 #分块

filePath = "Sandhill-Crane-256.png"
# filePath = "lena-128.png"
img_data = get_image_data(filePath)

origin_img = cv2.merge(img_data)
origin_img_corrcoef = corrcoef(origin_img) #计算相关系数

dct_img,idct_img = dct_image(img_data,block_size)
dct_img_corrcoef = corrcoef(dct_img)
# idct_img_corrcoef = corrcoef(idct_img)

print("变换前图像相关系数:",origin_img_corrcoef)
print("变换后图像相关系数:",dct_img_corrcoef)
# print("反变换后图像相关系数:",idct_img_corrcoef)

plt.subplot(1,3,1)
plt.title("origin_img_256")
plt.imshow(origin_img)
plt.axis("off")
plt.subplot(1,3,2)
plt.title("dct_img")
plt.imshow(dct_img)
plt.axis("off")
plt.subplot(1,3,3)
plt.title("idct_img")
plt.imshow(idct_img)
plt.axis("off")

plt.show()