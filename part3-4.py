import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
from numpy import histogram
import datetime

print(datetime.datetime.now())

img = imread('img.png')

def calculate_fx(x):
    
    return round(((np.sum(values[0:x+1]) - np.min(np.nonzero(np.sum(values[0:x+1])))) / (img.size - 1)) * 255)

values, bin_edges = histogram(img.ravel(), bins=range(257))
cdf_min = np.min(values[np.nonzero(values)])

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j] = calculate_fx(img[i,j])

imsave("out_img.png", img)
print(np.array_equal(img, imread('landscape-histeq.png')))

print(datetime.datetime.now())
