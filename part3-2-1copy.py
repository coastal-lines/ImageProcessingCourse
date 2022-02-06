import skimage
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
import numpy as np
from numpy import histogram
from matplotlib import pyplot as plt
from numpy import dstack
from numpy import ravel

img = imread('tiger-color.png')

#1
img_f = skimage.img_as_float(img)

#2
#YUV
r = img_f[:,:,0]
g = img_f[:,:,1]
b = img_f[:,:,2]
Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
U = -0.0999 * r - 0.3360 * g + 0.4360 * b
V = 0.6150 * r - 0.5586 * g + 0.0563 * b
#img_yuv = dstack((Y, U, V))

#3
#min color and max color
threshold = round(Y.size * 0.05)

values = sorted(np.reshape(Y, -1))
x_min = values[threshold+1]
x_max = values[-(threshold - 1)]

#4
value = (x_max - x_min)
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        pixel = (Y[i,j] - x_min) / value
        if(pixel < 0):
            Y[i,j] = 0.0
            continue

        if(pixel > 255):
            Y[i,j] = 1.0
            continue

        Y[i,j] = pixel

#5
#
#Y = np.clip(Y, 0, 1.0)

#6
R = Y + 1.2803 * V
G = Y - 0.2148 * U - 0.3805 * V
B = Y + 2.1279 * U

#7
R = (np.clip(R, 0, 1))
G = (np.clip(G, 0, 1))
B = (np.clip(B, 0, 1))
img_rgb_f = np.dstack((R, G, B))

#8
img_final = skimage.img_as_ubyte(img_rgb_f)
imsave('out_img.png', img_final)

#test
print(np.array_equal(img_final, imread(r'C:\Users\User\Downloads\tiger-stable-contrast.png')))