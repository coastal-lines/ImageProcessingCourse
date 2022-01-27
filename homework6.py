from skimage.io import imread, imsave
from numpy import dstack

img = imread('img.png')
r = img[:, :, 0]
g = img[:, :, 1]
b = img[:, :, 2]
img = dstack((b, r, g))

imsave('out_img.png', img)