import skimage.io
from skimage.io import imread, imsave, imshow
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

def roll(ch1, ch2):
    #axis = 0 row строка y
    #axis = 1 column столбец x
    
    best_b_x = 0
    best_b_y = 0
    max_corr_b = 0.0
    best_b_channel = None
    for i in range(15, -15, -1):
        b_temp = np.roll(ch1, i, axis=1)
        correlation = (b_temp * ch2).sum()
        if(correlation > max_corr_b):
            max_corr_b = correlation
            best_b_x = i
            best_b_channel = b_temp

    for j in range(15, -15, -1):
        b_temp = np.roll(ch1, j, axis=0)
        correlation = (b_temp * ch2).sum()
        if(correlation > max_corr_b):
            max_corr_b = correlation
            best_b_y = j
            best_b_channel = b_temp

    return best_b_x, best_b_y, best_b_channel

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

    best_b_x, best_b_y, best_b_channel = roll(b, g)
    best_r_x, best_r_y, best_r_channel = roll(r, g)

    #test
    y_g = g_coord[0]
    x_g = g_coord[1]

    row_b = best_b_y + (y_g - delimeter)
    col_b = best_b_x + x_g
    true_row_b = y_g - delimeter
    true_col_b = x_g

    row_r = best_r_y + (y_g + delimeter)
    col_r = best_r_x + x_g
    true_row_r = y_g + delimeter
    true_col_r = x_g

    diff =  abs(row_b - true_row_b) + abs(col_b - true_col_b) + abs(row_r - true_row_r) + abs(col_r - true_col_r)
    print(diff)

    #combine channels
    b = skimage.img_as_ubyte(best_b_channel)
    g = skimage.img_as_ubyte(g)
    r = skimage.img_as_ubyte(best_r_channel)
    img_result = dstack((r, g, b))
    imsave("out/" + str(name) + ".png", img_result)

    #return (row_b, col_b), (row_r, col_r)

align(img0, (508, 237),0)