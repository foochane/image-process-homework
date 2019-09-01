# -*- coding: UTF-8 -*-
import numpy as np
import math

#YUV
def RGB2YUV(R,G,B):
    matrix_tran = np.array([[0.299, 0.587, 0.114], \
                            [-0.14713, -0.28886, 0.436], \
                            [0.615, -0.51499, -0.10001]])
    Y = []
    U = []
    V = []
    for row in range(len(R)) :
        Y_row = []
        U_row = []
        V_row = []
        for col in range(len(R[row])) :
            Y_row.append(matrix_tran[0][0] * R[row][col] + matrix_tran[0][1] * G[row][col] + matrix_tran[0][2] * B[row][col])
            U_row.append(matrix_tran[1][0] * R[row][col] + matrix_tran[1][1] * G[row][col] + matrix_tran[1][2] * B[row][col])
            V_row.append(matrix_tran[2][0] * R[row][col] + matrix_tran[2][1] * G[row][col] + matrix_tran[2][2] * B[row][col])
        Y.append(Y_row)
        U.append(U_row)
        V.append(V_row)
    # 标准化
    return Y, U, V	

def YUV2RGB(Y,U,V):
    matrix_tran = np.array([[1, 0, 1.13983], \
                            [1, -0.39465, -0.58060], \
                            [1, 2.03211, 0]])
    R = []
    G = []
    B = []
    for row in range(len(Y)) :
        R_row = []
        G_row = []
        B_row = []
        for col in range(len(Y[row])) :
            R_row.append(matrix_tran[0][0] * Y[row][col] + matrix_tran[0][1] * U[row][col] + matrix_tran[0][2] * V[row][col])
            G_row.append(matrix_tran[1][0] * Y[row][col] + matrix_tran[1][1] * U[row][col] + matrix_tran[1][2] * V[row][col])
            B_row.append(matrix_tran[2][0] * Y[row][col] + matrix_tran[2][1] * U[row][col] + matrix_tran[2][2] * V[row][col])
        R.append(R_row)
        G.append(G_row)
        B.append(B_row)
    # 标准化
    return R, G, B	

# YIQ 变换
def RGB2YIQ(R, G, B) :
    matrix_tran = np.array([[0.299, 0.587, 0.114], \
                            [0.596, -0.275, -0.321], \
                            [0.212, -0.523, 0.311]])
    Y = []
    I = []
    Q = []
    for row in range(len(R)) :
        Y_row = []
        I_row = []
        Q_row = []
        for col in range(len(R[row])) :
            Y_row.append(matrix_tran[0][0] * R[row][col] + matrix_tran[0][1] * G[row][col] + matrix_tran[0][2] * B[row][col])
            I_row.append(matrix_tran[1][0] * R[row][col] + matrix_tran[1][1] * G[row][col] + matrix_tran[1][2] * B[row][col])
            Q_row.append(matrix_tran[2][0] * R[row][col] + matrix_tran[2][1] * G[row][col] + matrix_tran[2][2] * B[row][col])
        Y.append(Y_row)
        I.append(I_row)
        Q.append(Q_row)
    # 标准化
    return Y, I, Q

def YIQ2RGB(Y,I,Q):
    matrix_tran = np.array([[1, 0.956, 0.62], \
                            [1, -0.272, -0.647], \
                            [1, -1.108, 1.7]])
    R = []
    G = []
    B = []
    for row in range(len(Y)) :
        R_row = []
        G_row = []
        B_row = []
        for col in range(len(Y[row])) :
            R_row.append(matrix_tran[0][0] * Y[row][col] + matrix_tran[0][1] * I[row][col] + matrix_tran[0][2] * Q[row][col])
            G_row.append(matrix_tran[1][0] * Y[row][col] + matrix_tran[1][1] * I[row][col] + matrix_tran[1][2] * Q[row][col])
            B_row.append(matrix_tran[2][0] * Y[row][col] + matrix_tran[2][1] * I[row][col] + matrix_tran[2][2] * Q[row][col])
        R.append(R_row)
        G.append(G_row)
        B.append(B_row)
    # 标准化
    return R, G, B

# YCbCr 变换
def RGB2YCbCr(R, G, B) :
    matrix_tran = np.array([[0.299, 0.587, 0.114], \
                            [-0.169, -0.331, 0.5], \
                            [0.5, -0.419, -0.081]])
    # matrix_tran = np.array([[0.257, 0.504, 0.098], \
    #                         [-0.148, -0.291, 0.439], \
    #                         [0.439, -0.368, -0.071]])
    Y = []
    Cb = []
    Cr = []
    for row in range(len(R)) :
        Y_row = []
        Cb_row = []
        Cr_row = []
        for col in range(len(R[row])) :
            Y_row.append(matrix_tran[0][0] * R[row][col] + matrix_tran[0][1] * G[row][col] + matrix_tran[0][2] * B[row][col]+16)
            Cb_row.append(matrix_tran[1][0] * R[row][col] + matrix_tran[1][1] * G[row][col] + matrix_tran[1][2] * B[row][col] + 128)
            Cr_row.append(matrix_tran[2][0] * R[row][col] + matrix_tran[2][1] * G[row][col] + matrix_tran[2][2] * B[row][col] + 128)
        Y.append(Y_row)
        Cb.append(Cb_row)
        Cr.append(Cr_row)
    # 标准化
    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr) :
    matrix_tran = np.array([[1.164, 0, 1.596], \
                            [1.164, -0.392, 0.813], \
                            [1.164, 2.017, 0]])
    R = []
    G = []
    B = []
    for row in range(len(Y)) :
        R_row = []
        G_row = []
        B_row = []
        for col in range(len(Y[row])) :
            R_row.append(matrix_tran[0][0] * Y[row][col] + matrix_tran[0][1] * Cb[row][col] + matrix_tran[0][2] * Cr[row][col]-16)
            G_row.append(matrix_tran[1][0] * Y[row][col] + matrix_tran[1][1] * Cb[row][col] + matrix_tran[1][2] * Cr[row][col] - 128)
            B_row.append(matrix_tran[2][0] * Y[row][col] + matrix_tran[2][1] * Cb[row][col] + matrix_tran[2][2] * Cr[row][col] - 128)
        R.append(R_row)
        G.append(G_row)
        B.append(B_row)
    # 标准化
    return R, G, B

def YUV2YIQ(Y,U,V):
    R,G,B = YUV2RGB(Y,U,V)
    Y,I,Q = RGB2YIQ(R,G,B)
    return Y,I,Q

def YIQ2YUV(Y,I,Q):
    R,G,B = YIQ2RGB(Y,I,Q)
    Y,U,V = RGB2YUV(R,G,B)
    return Y,U,V

def YUV2YCbCr(Y,U,V):
    R,G,B = YUV2RGB(Y,U,V)
    Y,Cb,Cr = RGB2YCbCr(R,G,B)
    return Y,Cb,Cr

def YCbCr2YUV(Y,Cb,Cr):
    R,G,B = YCbCr2RGB(Y,Cb,Cr)
    Y,U,V = RGB2YUV(R,G,B)
    return Y,U,V

def YIQ2YCbCr(Y,I,Q):
    R,G,B = YIQ2RGB(Y,I,Q)
    Y,Cb,Cr = RGB2YCbCr(R,G,B)
    return Y,Cb,Cr

def YCbCr2YIQ(Y,Cb,Cr):
    R,G,B = YCbCr2RGB(Y,Cb,Cr)
    Y,I,Q = RGB2YIQ(R,G,B)
    return Y,I,Q