import skimage.io
from skimage.io import imread
import numpy as np

def align(img, g_coord):
    img_f = skimage.img_as_float(img)

    h = img_f.shape[0]
    w = img_f.shape[1]
    delimeter = h // 3
    b = img_f[0 : delimeter, 0 : w]
    g = img_f[delimeter : delimeter * 2, 0: w]
    r = img_f[delimeter * 2 : h, 0: w]

    h = b.shape[0]
    w = b.shape[1]
    cutH = (h // 100) * 3
    cutW = (w // 100) * 5
    b = b[cutH: h - cutH, cutW: w - cutW]
    g = g[cutH: h - cutH, cutW: w - cutW]
    r = r[cutH: h - cutH, cutW: w - cutW]

    best_b_x = 0
    best_b_y = 0
    max_corr_b = 0.0
    for x in range(15):
        temp = np.roll(b, x, axis=1)
        correlation = (temp * g).sum()
        if (correlation > max_corr_b):
            max_corr_b = correlation
            best_b_x = x

    for x in range(-15, 0, 1):
        temp = np.roll(b, x, axis=1)
        correlation = (temp * g).sum()
        if (correlation > max_corr_b):
            max_corr_b = correlation
            best_b_x = x

    for y in range(15):
        temp = np.roll(b, y, axis=0)
        correlation = (temp * g).sum()
        if (correlation > max_corr_b):
            max_corr_b = correlation
            best_b_y = y

    for y in range(-15, 0, 1):
        temp = np.roll(b, y, axis=0)
        correlation = (temp * g).sum()
        if (correlation > max_corr_b):
            max_corr_b = correlation
            best_b_y = y

    best_r_x = 0
    best_r_y = 0
    max_corr_r = 0.0
    for x in range(15):
        temp = np.roll(r, x, axis=1)
        correlation = (temp * g).sum()
        if (correlation > max_corr_r):
            max_corr_r = correlation
            best_r_x = x

    for x in range(-15, 0, 1):
        temp = np.roll(r, x, axis=1)
        correlation = (temp * g).sum()
        if (correlation > max_corr_r):
            max_corr_r = correlation
            best_r_x = x

    for y in range(15):
        temp = np.roll(r, y, axis=0)
        correlation = (temp * g).sum()
        if (correlation > max_corr_r):
            max_corr_r = correlation
            best_r_y = y

    for y in range(-15, 0, 1):
        temp = np.roll(r, y, axis=0)
        correlation = (temp * g).sum()
        if (correlation > max_corr_r):
            max_corr_r = correlation
            best_r_y = y

    y_g = g_coord[0]
    x_g = g_coord[1]
    row_b = (y_g - best_b_y) - delimeter
    col_b = (x_g - best_b_x)
    #true_row_b = y_g - delimeter
    #true_col_b = x_g
    row_r = (y_g - best_r_y) + delimeter
    col_r = x_g - best_r_x
    #true_row_r = y_g + delimeter
    #true_col_r = x_g

    #diff = abs(true_row_b - row_b) + abs(true_col_b - col_b) + abs(true_row_r - row_r) + abs(true_col_r - col_r)
    #print("diff:      " + str(diff))
    #print("actual:    " + str([row_b, col_b, row_r, col_r]))
    #print("===============================")

    return (row_b, col_b), (row_r, col_r)

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

align(img0, (508, 237))
align(img1, (483, 218))
align(img2, (557, 141))
align(img3, (627, 179))
align(img4, (540, 96))
align(img5, (641, 369))
align(img6, (527, 196))
align(img7, (430, 140))
align(img8, (502, 254))
align(img9, (493, 238))