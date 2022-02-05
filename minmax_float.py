import skimage
from skimage.io import imread, imsave
import numpy as np
from numpy import histogram
from matplotlib import pyplot as plt
from numpy import dstack
from numpy import ravel

#int
print("int: ")
img = imread('img.png')
threshold1 = round(img[:,:,0].size * 0.05)
vector = np.reshape(img[:,:,0], -1)
vector.sort()
values, bin_edges = histogram(vector.ravel(), bins=range(257))

count = 0
x_min_shifted = 0
for i in range(256):
    count += values[i]
    if count > threshold1:
        x_min_shifted = i
        print(x_min_shifted)
        break

count = 0
x_max_shifted = 0
for i in range(255, 0, -1):
    count += values[i]
    if count > threshold1:
        x_max_shifted = i
        print(x_max_shifted)
        break

#float
print("float: ")
img_f = skimage.img_as_float(img[:,:,0])
threshold2 = round(img_f.size * 0.05)
values2 = sorted(ravel(img_f))
#print(values2)