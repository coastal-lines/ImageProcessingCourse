import numpy as np
import math

import scipy
from numpy import uint8
from skimage.io import imread, imsave

img = imread("tiger-gray-small.png")

kernel = np.array([[-1, -2, -1],
                   [-2, 22, -2],
                   [-1, -2, -1]])

kernel = kernel * 0.1

border = kernel[0].size // 2

def calculate(arr, kernel):
    sum = 0

    arr = np.reshape(arr, -1)
    arr = arr[::-1]
    kernel = np.reshape(kernel, -1)

    for i in range(arr.size):
            sum += arr[i] * kernel[i]

    arr = np.array([arr],dtype=uint8)
    s = round((arr*kernel).sum().astype('uint8'))

    #return sum
    if(s < 0):
        s = 0

    if(s > 255):
        s = 255

    return s

temp = []
for i in range(border, img.shape[0] - border):
    for j in range(border, img.shape[1] - border):
        matrix = img[i - border : i + border + 1, j - border: j + border + 1]
        res = calculate(matrix, kernel)
        temp.append(res)

temp = np.reshape(temp, [img.shape[0] - (border*2), img.shape[1] - (border*2)])
#temp = np.clip(temp,0,255)
#temp = np.ndarray.astype(temp, np.uint8)

imsave('out_img.png', temp)
test = imread('unsharp-tiger.png')
print(np.array_equal(temp, test))