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

def align(img, g_coord):
    #200 150

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
    best_b_x = 0
    best_b_y = 0
    max_corr_b = 0.0
    for i in range(15):
        for j in range(15):
            b_temp = np.roll(b, i, axis=0)
            b_temp = np.roll(b, j, axis=1)
            correlation = (b_temp * g).sum()
            if(correlation > max_corr_b):
                max_corr_b = correlation
                best_b_x = i
                best_b_y = j
    print(max_corr_b)

    #roll red
    max_corr_r = 0.0
    best_r_x = 0
    best_r_y = 0
    for i in range(15):
        for j in range(15):
            r_temp = np.roll(r, i, axis=0)
            r_temp = np.roll(r, j, axis=1)
            correlation = (r_temp * g).sum()
            if(correlation > max_corr_r):
                max_corr_r = correlation
                best_r_x = i
                best_r_y = j
    print(max_corr_r)

    #shift the best layers
    b = np.roll(b, best_b_x, axis=0)
    b = np.roll(b, best_b_y, axis=1)
    r = np.roll(r, best_r_x, axis=0)
    r = np.roll(r, best_r_y, axis=1)

    #combine channels
    b = skimage.img_as_ubyte(b)
    g = skimage.img_as_ubyte(g)
    r = skimage.img_as_ubyte(r)
    img = dstack((r, g, b))

    #test
    diff =

    imsave('out/img.png', img)

align(img3, (200, 150))