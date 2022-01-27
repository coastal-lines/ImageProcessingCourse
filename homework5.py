from skimage.io import imread, imsave
import numpy as np

img = imread('img.png')
h = img.shape[0]
w = img.shape[1]

for i in range(h):
    for j in range(w):
        img[i, j, 0] = 255 - img[i, j, 0]
        img[i, j, 1] = 255 - img[i, j, 1]
        img[i, j, 2] = 255 - img[i, j, 2]

imsave('out_img.png', img)