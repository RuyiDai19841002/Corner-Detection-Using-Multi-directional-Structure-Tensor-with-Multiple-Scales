import cv2
import numpy as np
from scipy import signal
from pylab import *
def improved1(im,threshold):
    rho = 1.5
    p = 8
    sigma1 = 1.5
    sigma2 = 3
    sigma3 = 4.5
    radius = 5
    esp = 2.220 * 10 ** (-16)
    if im.shape[2]==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image',im)
    rows, cols = np.shape(im)
    wid = 7
    width = 15
    extend = cv2.copyMakeBorder(im, 7, 7, 7, 7, cv2.BORDER_REFLECT)
    extend_image = np.array(extend, np.float32)
    # cv2.imshow('extend',extend_image)
    # cv2.waitKey()
    V1 = []
    V2 = []
    V3 = []
    M = []
    M2 = []
    M3 = []
    M1 = np.mat([[rho, 0], [0, 1 / rho]])
    for direction in range(1, p + 1):
        theta = (direction - 1) * np.pi / (p)
        anigs_direction1 = np.zeros((2 * width + 1, 2 * width + 1))
        anigs_direction2 = np.zeros((2 * width + 1, 2 * width + 1))
        anigs_direction3 = np.zeros((2 * width + 1, 2 * width + 1))
        R = np.mat([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        R1 = np.mat([[np.cos(-theta), np.sin(-theta)], [-np.sin(-theta), np.cos(-theta)]])
        for x in range(-width, width + 1):
            for y in range(-width, width + 1):
                X1 = np.mat([x, y])
                X2 = X1.T
                anigs_direction1[x + width, y + width] = -rho * (x * np.cos(theta) + y * np.sin(theta)) * 1 / (
                            2 * np.pi * sigma1 * sigma1) * np.exp(-1 / (2 * sigma1) * X1 * R1 * M1 * R * X2)
                anigs_direction2[x + width, y + width] = -rho * (x * np.cos(theta) + y * np.sin(theta)) * 1 / (
                            2 * np.pi * sigma2 * sigma2) * np.exp(-1 / (2 * sigma2) * X1 * R1 * M1 * R * X2)
                anigs_direction3[x + width, y + width] = -rho * (x * np.cos(theta) + y * np.sin(theta)) * 1 / (
                            2 * np.pi * sigma3 * sigma3) * np.exp(-1 / (2 * sigma3) * X1 * R1 * M1 * R * X2)
                # print(type(anigs_direction3))
        V1.append(anigs_direction1)
        V2.append(anigs_direction2)
        V3.append(anigs_direction3)
    # print(np.shape(V1[7]))
    # print(V2)
    # print(V3)
    # smooth the input image
    template1 = np.zeros((2 * wid + rows, cols + 2 * wid))
    template2 = np.zeros((2 * wid + rows, cols + 2 * wid))
    template3 = np.zeros((2 * wid + rows, cols + 2 * wid))
    for d in range(0, p):
        a = V1[d]
        b = V2[d]
        c = V3[d]
        oa = a - (a.sum()) / (a.size)
        ob = b - (b.sum()) / (b.size)
        oc = c - (c.sum()) / (c.size)
        template1 = signal.convolve2d(extend_image, oa, mode='same')
        template2 = signal.convolve2d(extend_image, ob, mode='same')
        template3 = signal.convolve2d(extend_image, oc, mode='same')
        # print(type(template3))
        # print(np.shape(template3))
        M.append(template1)
        M2.append(template2)
        M3.append(template3)
    # print(np.shape(M2[7]))
    measure = np.zeros((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            Tem = []
            for d in range(0, p):
                A = M[d]
                a1 = A[i + wid - 1, j + wid + 3]
                a2 = A[i + wid, j + wid + 3]
                a3 = A[i + wid + 1, j + wid + 3]
                a4 = A[i + wid - 2, j + wid + 2]
                a5 = A[i + wid - 1, j + wid + 2]
                a6 = A[i + wid, j + wid + 2]
                a7 = A[i + wid + 1, j + wid + 2]
                a8 = A[i + wid + 2, j + wid + 2]
                a9 = A[i + wid - 3, j + wid + 1]
                a10 = A[i + wid - 2, j + wid + 1]
                a11 = A[i + wid - 1, j + wid + 1]
                a12 = A[i + wid, j + wid + 1]
                a13 = A[i + wid + 1, j + wid + 1]
                a14 = A[i + wid + 2, j + wid + 1]
                a15 = A[i + wid + 3, j + wid + 1]
                a16 = A[i + wid - 3, j + wid]
                a17 = A[i + wid - 2, j + wid]
                a18 = A[i + wid - 1, j + wid]
                a19 = A[i + wid, j + wid]
                a20 = A[i + wid + 1, j + wid]
                a21 = A[i + wid + 2, j + wid]
                a22 = A[i + wid + 3, j + wid]
                a23 = A[i + wid - 3, j + wid - 1]
                a24 = A[i + wid - 2, j + wid - 1]
                a25 = A[i + wid - 1, j + wid - 1]
                a26 = A[i + wid, j + wid - 1]
                a27 = A[i + wid + 1, j + wid - 1]
                a28 = A[i + wid + 2, j + wid - 1]
                a29 = A[i + wid + 3, j + wid - 1]
                a30 = A[i + wid - 2, j + wid - 2]
                a31 = A[i + wid - 1, j + wid - 2]
                a32 = A[i + wid, j + wid - 2]
                a33 = A[i + wid + 1, j + wid - 2]
                a34 = A[i + wid + 2, j + wid - 2]
                a35 = A[i + wid - 1, j + wid - 3]
                a36 = A[i + wid, j + wid - 3]
                a37 = A[i + wid + 1, j + wid - 3]
                s = np.mat([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21,
                     a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37]).T
                Tem.append(s)
            T = np.transpose(Tem)
            M4 = (np.mat(np.abs(Tem))) * (np.mat(np.abs(T)))
            measure[i, j] = np.linalg.det(M4) / (np.trace(M4) + esp)
    marked_img_one = np.zeros((rows, cols))
    marked_img_two = np.zeros((rows, cols))
    marked_img_three = np.zeros((rows, cols))
    r = nonma(measure, threshold, radius)
    # figure(1)
    # gray()
    # imshow(im)
    # plot([p[1] for p in r],[p[0]for p in r],'w.')
    # axis('off')
    # show()
    for i in range(len(r)):
        Tem2 = []
        for d in range(0, p):
            A2 = M2[d]
            a1 = A2[r[i, 0] + wid - 1, r[i, 1] + wid + 3]
            a2 = A2[r[i, 0] + wid, r[i, 1] + wid + 3]
            a3 = A2[r[i, 0] + wid + 1, r[i, 1] + wid + 3]
            a4 = A2[r[i, 0] + wid - 2, r[i, 1] + wid + 2]
            a5 = A2[r[i, 0] + wid - 1, r[i, 1] + wid + 2]
            a6 = A2[r[i, 0] + wid, r[i, 1] + wid + 2]
            a7 = A2[r[i, 0] + wid + 1, r[i, 1] + wid + 2]
            a8 = A2[r[i, 0] + wid + 2, r[i, 1] + wid + 2]
            a9 = A2[r[i, 0] + wid - 3, r[i, 1] + wid + 1]
            a10 = A2[r[i, 0] + wid - 2, r[i, 1] + wid + 1]
            a11 = A2[r[i, 0] + wid - 1, r[i, 1] + wid + 1]
            a12 = A2[r[i, 0] + wid, r[i, 1] + wid + 1]
            a13 = A2[r[i, 0] + wid + 1, r[i, 1] + wid + 1]
            a14 = A2[r[i, 0] + wid + 2, r[i, 1] + wid + 1]
            a15 = A2[r[i, 0] + wid + 3, r[i, 1] + wid + 1]
            a16 = A2[r[i, 0] + wid - 3, r[i, 1] + wid]
            a17 = A2[r[i, 0] + wid - 2, r[i, 1] + wid]
            a18 = A2[r[i, 0] + wid - 1, r[i, 1] + wid]
            a19 = A2[r[i, 0] + wid, r[i, 1] + wid]
            a20 = A2[r[i, 0] + wid + 1, r[i, 1] + wid]
            a21 = A2[r[i, 0] + wid + 2, r[i, 1] + wid]
            a22 = A2[r[i, 0] + wid + 3, r[i, 1] + wid]
            a23 = A2[r[i, 0] + wid - 3, r[i, 1] + wid - 1]
            a24 = A2[r[i, 0] + wid - 2, r[i, 1] + wid - 1]
            a25 = A2[r[i, 0] + wid - 1, r[i, 1] + wid - 1]
            a26 = A2[r[i, 0] + wid, r[i, 1] + wid - 1]
            a27 = A2[r[i, 0] + wid + 1, r[i, 1] + wid - 1]
            a28 = A2[r[i, 0] + wid + 2, r[i, 1] + wid - 1]
            a29 = A2[r[i, 0] + wid + 3, r[i, 1] + wid - 1]
            a30 = A2[r[i, 0] + wid - 2, r[i, 1] + wid - 2]
            a31 = A2[r[i, 0] + wid - 1, r[i, 1] + wid - 2]
            a32 = A2[r[i, 0] + wid, r[i, 1] + wid - 2]
            a33 = A2[r[i, 0] + wid + 1, r[i, 1] + wid - 2]
            a34 = A2[r[i, 0] + wid + 2, r[i, 1] + wid - 2]
            a35 = A2[r[i, 0] + wid - 1, r[i, 1] + wid - 3]
            a36 = A2[r[i, 0] + wid, r[i, 1] + wid - 3]
            a37 = A2[r[i, 0] + wid + 1, r[i, 1] + wid - 3]
            t = np.mat([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22,
                 a23,a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37]).T
            Tem2.append(t)
        T2 = np.transpose(Tem2)
        M5 = (np.mat(np.abs(Tem2))) * (np.mat(np.abs(T2)))
        KKK = np.linalg.det(M5) / (np.trace(M5) + esp)
        if KKK > threshold:
            marked_img_two[r[i, 0], r[i, 1]] = 1
    rr = np.array(marked_img_two.nonzero()).T
    # figure(2)
    # gray()
    # imshow(im)
    # plot([p[1] for p in rr],[p[0]for p in rr],'w.')
    # axis('off')
    # show()
    for i in range(len(rr)):
        Tem3 = []
        for d in range(0, p):
            A3 = M3[d]
            a1 = A3[rr[i, 0] + wid - 1, rr[i, 1] + wid + 3]
            a2 = A3[rr[i, 0] + wid, rr[i, 1] + wid + 3]
            a3 = A3[rr[i, 0] + wid + 1, rr[i, 1] + wid + 3]
            a4 = A3[rr[i, 0] + wid - 2, rr[i, 1] + wid + 2]
            a5 = A3[rr[i, 0] + wid - 1, rr[i, 1] + wid + 2]
            a6 = A3[rr[i, 0] + wid, rr[i, 1] + wid + 2]
            a7 = A3[rr[i, 0] + wid + 1, rr[i, 1] + wid + 2]
            a8 = A3[rr[i, 0] + wid + 2, rr[i, 1] + wid + 2]
            a9 = A3[rr[i, 0] + wid - 3, rr[i, 1] + wid + 1]
            a10 = A3[rr[i, 0] + wid - 2, rr[i, 1] + wid + 1]
            a11 = A3[rr[i, 0] + wid - 1, rr[i, 1] + wid + 1]
            a12 = A3[rr[i, 0] + wid, rr[i, 1] + wid + 1]
            a13 = A3[rr[i, 0] + wid + 1, rr[i, 1] + wid + 1]
            a14 = A3[rr[i, 0] + wid + 2, rr[i, 1] + wid + 1]
            a15 = A3[rr[i, 0] + wid + 3, rr[i, 1] + wid + 1]
            a16 = A3[rr[i, 0] + wid - 3, rr[i, 1] + wid]
            a17 = A3[rr[i, 0] + wid - 2, rr[i, 1] + wid]
            a18 = A3[rr[i, 0] + wid - 1, rr[i, 1] + wid]
            a19 = A3[rr[i, 0] + wid, rr[i, 1] + wid]
            a20 = A3[rr[i, 0] + wid + 1, rr[i, 1] + wid]
            a21 = A3[rr[i, 0] + wid + 2, rr[i, 1] + wid]
            a22 = A3[rr[i, 0] + wid + 3, rr[i, 1] + wid]
            a23 = A3[rr[i, 0] + wid - 3, rr[i, 1] + wid - 1]
            a24 = A3[rr[i, 0] + wid - 2, rr[i, 1] + wid - 1]
            a25 = A3[rr[i, 0] + wid - 1, rr[i, 1] + wid - 1]
            a26 = A3[rr[i, 0] + wid, rr[i, 1] + wid - 1]
            a27 = A3[rr[i, 0] + wid + 1, rr[i, 1] + wid - 1]
            a28 = A3[rr[i, 0] + wid + 2, rr[i, 1] + wid - 1]
            a29 = A3[rr[i, 0] + wid + 3, rr[i, 1] + wid - 1]
            a30 = A3[rr[i, 0] + wid - 2, rr[i, 1] + wid - 2]
            a31 = A3[rr[i, 0] + wid - 1, rr[i, 1] + wid - 2]
            a32 = A3[rr[i, 0] + wid, rr[i, 1] + wid - 2]
            a33 = A3[rr[i, 0] + wid + 1, rr[i, 1] + wid - 2]
            a34 = A3[rr[i, 0] + wid + 2, rr[i, 1] + wid - 2]
            a35 = A3[rr[i, 0] + wid - 1, rr[i, 1] + wid - 3]
            a36 = A3[rr[i, 0] + wid, rr[i, 1] + wid - 3]
            a37 = A3[rr[i, 0] + wid + 1, rr[i, 1] + wid - 3]
            pp =np.mat([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22,
                  a23,a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37]).T
            Tem3.append(pp)
        T3 = np.transpose(Tem3)
        M6 = (np.mat(np.abs(Tem3))) * (np.mat(np.abs(T3)))
        KKKK = np.linalg.det(M6) / (np.trace(M6) + esp)
        if KKKK > threshold:
            marked_img_two[rr[i, 0], rr[i, 1]] = 1
    rrr = np.array(marked_img_two.nonzero()).T
    return rrr
def nonma(cim,threshold,radius):
    rows,cols=np.shape(cim)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
    mx=cv2.dilate(cim,kernel)
    bordermask=np.zeros(np.shape(cim),np.int)
    bordermask[radius+1:rows-radius,radius+1:cols-radius]=1
    t = (cim == mx) + 0
    t2 = (cim > threshold) + 0
    cimmx=t & t2 & bordermask
    r = np.array(cimmx.nonzero()).T
    return r