import skimage
from skimage.io import imread, imsave
import numpy as np
from numpy import histogram
from matplotlib import pyplot as plt
from numpy import dstack

img = imread('img.png')

img_f = skimage.img_as_float(img)

#YUV
Y = 0.2126 * img_f[:,:,0] + 0.7152 * img_f[:,:,1] + 0.0722 * img_f[:,:,2]
U = -0.0999 * img_f[:,:,0] - 0.3360 * img_f[:,:,1] + 0.4360 * img_f[:,:,2]
V = 0.6150 * img_f[:,:,0] - 0.5586 * img_f[:,:,1] + 0.0563 * img_f[:,:,2]
img_yuv = dstack((Y, U, V))

#imsave('out_img.png', img)