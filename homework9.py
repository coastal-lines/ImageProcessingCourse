import skimage.io
from skimage.io import imread, imsave
import numpy as np
from numpy import dstack

img0 = imread('00.png')
img1 = imread('01.png')
img2 = imread('02.png')
img3 = imread('03.png')
img4 = imread('04.png')
img5 = imread('05.png')
img6 = imread('06.png')
img7 = imread('07.png')
img8 = imread('08.png')
img9 = imread('09.png')

def roll(channel1, channel2):
    best_x = 0
    best_y = 0
    max_corr = 0.0
    for i in range(15, -15, -1):
        channel1_temp = np.roll(channel1, i, axis=0)
        correlation = (channel1_temp * channel2).sum()
        if(correlation > max_corr):
            max_corr = correlation
            best_x = i
        for j in range(15, -15, -1):
            channel1_temp = np.roll(channel1, j, axis=1)
            correlation = (channel1_temp * channel2).sum()
            if(correlation > max_corr):
                max_corr = correlation
                best_y = j

    return best_y, best_x

def align(img, g_coord, name):
    #load and convert to float
    img_f = skimage.img_as_float(img)

    #divide into three parts
    h = img_f.shape[0]
    w = img_f.shape[1]
    delimeter = h // 3
    b = img_f[0 : delimeter, 0 : w]
    g = img_f[delimeter : delimeter * 2, 0: w]
    r = img_f[delimeter * 2 : h, 0: w]

    #cut borders 5%
    h = b.shape[0]
    w = b.shape[1]
    cutH = (h // 100) * 5
    cutW = (w // 100) * 5
    b = b[cutH: h - cutH, cutW: w - cutW]
    g = g[cutH: h - cutH, cutW: w - cutW]
    r = r[cutH: h - cutH, cutW: w - cutW]

    #roll blue
    roll(b, g)
    best_b_y, best_b_x = 
    max_corr_b = 0.0
    for i in range(15, -15, -1):
        b_temp = np.roll(b, i, axis=0)
        correlation = (b_temp * g).sum()
        if(correlation > max_corr_b):
            max_corr_b = correlation
            best_b_x = i
        for j in range(15, -15, -1):
            b_temp = np.roll(b, j, axis=1)
            correlation = (b_temp * g).sum()
            if(correlation > max_corr_b):
                max_corr_b = correlation
                best_b_y = j
    #print(max_corr_b)

    #roll red
    max_corr_r = 0.0
    best_r_x = 0
    best_r_y = 0
    for i in range(15, -15, -1):
        r_temp = np.roll(r, i, axis=0)
        correlation = (r_temp * g).sum()
        if (correlation > max_corr_r):
            max_corr_r = correlation
            best_r_x = i
        for j in range(15, -15, -1):
            r_temp = np.roll(r, j, axis=1)
            correlation = (r_temp * g).sum()
            if (correlation > max_corr_r):
                max_corr_r = correlation
                best_r_y = j

    #shift the best layers
    #b = np.roll(b, best_b_x, axis=0)
    #b = np.roll(b, best_b_y, axis=1)
    #r = np.roll(r, best_r_x, axis=0)
    #r = np.roll(r, best_r_y, axis=1)

    #combine channels
    #b = skimage.img_as_ubyte(b)
    #g = skimage.img_as_ubyte(g)
    #r = skimage.img_as_ubyte(r)
    #img_result = dstack((r, g, b))
    #imsave("out/" + str(name) + "b" + ".png", b)
    #imsave("out/" + str(name) + "g" + ".png", g)
    #imsave("out/" + str(name) + "r" + ".png", r)

    #test
    y_g, x_g = g_coord
    true_col_b = x_g
    true_row_b = y_g - delimeter
    true_col_r = x_g
    true_row_r = y_g + delimeter

    col_b = x_g + best_b_x
    row_b = y_g + best_b_y - delimeter
    col_r = x_g + best_r_x
    row_r = y_g + best_r_y + delimeter
    print(row_b, col_b, row_r, col_r)
    print(153, 236, 858, 238)

    diff =  abs(row_b - true_row_b) + abs(col_b - true_col_b) + abs(row_r - true_row_r) + abs(col_r - true_col_r)
    print(diff)

    #imsave("out/" + str(name) + ".png", img_result)

    return (row_b, col_b), (row_r, col_r)

align(img0, (508, 237),0)
#align(img1, (508, 237),1)
#align(img2, (508, 237),2)
#align(img3, (508, 237),3)
#align(img4, (508, 237),4)
#align(img5, (508, 237),5)
#align(img6, (508, 237),6)
#align(img7, (508, 237),7)
#align(img8, (508, 237),8)
#align(img9, (508, 237),9)