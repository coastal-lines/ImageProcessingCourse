import numpy as np
import math
from skimage.io import imread, imsave

img = imread("tiger-gray-small.png")

def gauss(arr):
    o = arr[0]
    x = arr[1]
    y = arr[2]

    o_pow = math.pow(o, 2)
    x_pow = math.pow(x,2)
    y_pow = math.pow(y,2)

    p1 = 1 / (2 * math.pi * o_pow)
    p2 = (-x_pow - y_pow) / (2 * o_pow)
    p3 = p1 * (math.e ** p2)

    return p3

def k_width(o):
    s = (o * 6) + 1
    k = round(s)

    if(k % 2 == 0):
        k = k+1

    return k

def calculate_sum(o, k):
    sum = 0
    radius = k // 2
    for i in range(0 - radius, radius + 1, 1):
        for j in range(0 - radius, radius + 1, 1):
            sum += gauss([o, i, j])

    return sum

def get_kernel(o, k, sum):
    kernel = []
    radius = k // 2

    for i in range(0 - radius, radius + 1, 1):
        for j in range(0 - radius, radius + 1, 1):
            p = gauss([o, i, j]) / sum
            kernel.append(p)

    return np.reshape(kernel, [k, k])

def calculate_kernel(o):
    k = k_width(o)
    sum = calculate_sum(o, k)
    kernel = get_kernel(o, k, sum)

    return kernel

def apply_kernel(arr, kernel):
    sum = 0

    arr = np.reshape(arr, -1)
    arr = arr[::-1]
    kernel = np.reshape(kernel, -1)

    for i in range(arr.size):
            temp = arr[i] * kernel[i]
            sum += temp

    return sum

def make_blur(img,kernel):
    border = kernel[0].size // 2
    temp = []
    for i in range(border, img.shape[0] - border):
        for j in range(border, img.shape[1] - border):
            matrix = img[i - border: i + border + 1, j - border: j + border + 1]
            res = apply_kernel(matrix, kernel)
            temp.append(res)

    temp = np.reshape(temp, [img.shape[0] - (border * 2), img.shape[1] - (border * 2)])
    #temp = temp.astype(int)
    temp = np.ndarray.astype(temp, np.uint8)
    #temp = np.clip(temp, 0, 255)

    imsave('out_img.png', temp)
    test = imread('gaussian-tiger.png')
    print(np.array_equal(temp, test))

kernel = calculate_kernel(0.66)

#kernel = np.array([[0.00004,0.00117,0.00370,0.00117,0.00004],[0.00117,0.03677,0.11586,0.03677,0.00117],[0.00370,0.11586,0.36513,0.11586,0.00370],[0.00117,0.03677,0.11586,0.03677,0.00117],[0.00004,0.00117,0.00370,0.00117,0.00004]])

make_blur(img,kernel)