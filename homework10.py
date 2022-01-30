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

    best_x = 0
    best_y = 0
    max_corr_b = 0.0
    best_channel = None

    for x in range(15):
        temp = np.roll(ch1, x, axis=1)
        correlation = (temp * ch2).sum()
        if (correlation > max_corr_b):
            max_corr_b = correlation
            best_x = x

    for x in range(-15, 0, 1):
        temp = np.roll(ch1, x, axis=1)
        correlation = (temp * ch2).sum()
        if (correlation > max_corr_b):
            max_corr_b = correlation
            best_x = x

    for y in range(15):
        temp = np.roll(ch1, y, axis=0)
        correlation = (temp * ch2).sum()
        if (correlation > max_corr_b):
            max_corr_b = correlation
            best_y = y

    for y in range(-15, 0, 1):
        temp = np.roll(ch1, y, axis=0)
        correlation = (temp * ch2).sum()
        if (correlation > max_corr_b):
            max_corr_b = correlation
            best_y = y

    best_channel = np.roll(np.roll(ch1, best_y, axis=0), best_x, axis=1)
    return best_x, best_y, best_channel

def align(img, g_coord, name, scores):
    #load and convert to float
    img_f = skimage.img_as_float(img)

    #divide into three parts
    h = img_f.shape[0]
    w = img_f.shape[1]
    delimeter = h // 3

    b = img_f[0 : delimeter, 0 : w]
    g = img_f[delimeter : delimeter * 2, 0: w]
    r = img_f[delimeter * 2 : h, 0: w]

    imsave("out/b.png", skimage.img_as_ubyte(b))
    imsave("out/g.png", skimage.img_as_ubyte(g))
    imsave("out/r.png", skimage.img_as_ubyte(r))

    #cut borders 5%
    h = b.shape[0]
    w = b.shape[1]
    cutH = (h // 100) * 3
    cutW = (w // 100) * 5
    b = b[cutH: h - cutH, cutW: w - cutW]
    g = g[cutH: h - cutH, cutW: w - cutW]
    r = r[cutH: h - cutH, cutW: w - cutW]

    best_b_x, best_b_y, best_b_channel = roll(b, g)
    best_r_x, best_r_y, best_r_channel = roll(r, g)

    #test
    y_g = g_coord[0]
    x_g = g_coord[1]

    #best_b_x = best_b_x + cutW*2
    #best_b_y = best_b_y + cutH*2
    #best_r_x = best_r_x + cutW*2
    #best_r_y = best_b_y + cutH*2

    row_b = (y_g - best_b_y) - delimeter
    col_b = (x_g - best_b_x)
    true_row_b = y_g - delimeter
    true_col_b = x_g

    row_r = (y_g - best_r_y) + delimeter
    col_r = x_g - best_r_x
    true_row_r = y_g + delimeter
    true_col_r = x_g

    diff = abs(row_b - true_row_b) + abs(col_b - true_col_b) + abs(row_r - true_row_r) + abs(col_r - true_col_r)

    print("diff:      " + str(diff))
    print("actual:    " + str([row_b, col_b, row_r, col_r]))
    print("excpected: " + str(scores))
    print("===============================")

    #combine channels
    b = skimage.img_as_ubyte(best_b_channel)
    g = skimage.img_as_ubyte(g)
    r = skimage.img_as_ubyte(best_r_channel)

    img_result = dstack((r, g, b))
    imsave("out/" + str(name) + ".png", img_result)
    imsave("out/b.png", b)
    imsave("out/g.png", g)
    imsave("out/r.png", r)

    #return (row_b, col_b), (row_r, col_r)

###align(img0, (508, 237), 0, [153,236,858,238])
###align(img1, (483, 218), 1, [145, 219,817, 218])
###align(img2, (557, 141), 2, [204, 143, 908, 140])
###align(img3, (627, 179), 3, [243, 179, 1010, 176])
###align(img4, (540, 96), 4, [154, 95, 922, 94])
###align(img5, (641, 369), 5, [258, 372, 1021, 368])
###align(img6, (527, 196), 6, [144, 198, 908, 193])
###align(img7, (430, 140), 7, [82, 140, 777, 141])
###align(img8, (502, 254), 8, [123, 259, 880, 251])
align(img9, (493, 238), 9, [114, 240, 871, 235])