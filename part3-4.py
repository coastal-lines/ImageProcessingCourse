import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
from numpy import histogram

img = imread(r'landscape.png')
values, bin_edges = histogram(img.ravel(), bins=range(257))

cdf = 0
for i in range(values.size):
    cdf = cdf + values[i]

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i][j] = 

print("")