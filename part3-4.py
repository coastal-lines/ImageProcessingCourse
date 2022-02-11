import numpy as np
from skimage.io import imread, imsave
from numpy import histogram
from functools import reduce
import datetime

print(datetime.datetime.now())

def calc(x):
    min = np.min(np.nonzero(np.sum(values[0:x + 1])))
    if(min == 0):
        min = cdf_min_main
    return round(((np.sum(values[0:x + 1]) - min) / i_size) * 255)

img = imread('img.png')
i_size = img.size - 1

values, bin_edges = histogram(img.ravel(), bins=range(257))

unique = np.unique(img)

cdf_min_main_arr = []
for i in range(unique.shape[0]):
    cdf_min_main_arr.append(np.sum(values[0:unique[i] + 1]))
cdf_min_main = np.min(cdf_min_main_arr)

cdf_unique = []
for i in range(unique.shape[0]):
    cdf_unique.append([unique[i], calc(unique[i])])

cdf_unique = np.stack(cdf_unique, axis=0)
cdf_min_main = np.min(np.nonzero(cdf_unique[:,1]))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i, j] = cdf_unique[np.where(cdf_unique[:,0] == img[i, j])[0][0]][1]

print(np.array_equal(img, imread('landscape-histeq.png')))
imsave("out_img.png", img)

print(datetime.datetime.now())
print(np.array_equal(img, imread('landscape-histeq.png')))

print(datetime.datetime.now())
