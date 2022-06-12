import numpy as np
import skimage.io
from skimage.io import imread,imsave
import math
import matplotlib.pyplot as plt

def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def extend_image(img, kernel_width):

    #extend image if kernel equals 7*7
    #it means that we have to extend each border to 3 pixels
    #and corners

    #the mirror method
    #get radius
    r = kernel_width // 2

    #make new extended image
    new_img = np.zeros((img.shape[0] + (r * 2), img.shape[1] + (r * 2), img.shape[2]) ,np.uint8)

    #top
    top = img[0:r, 0:img.shape[1]]
    top = top[::-1]
    new_img[0:r, r:img.shape[1] + r] = top
    #corners on the top
    new_img[0:r, 0:r] = top[0:r, 0:r]
    new_img[0:r, new_img.shape[1] - r: new_img.shape[1]] = img[0:r, img.shape[1] - r: img.shape[1]]

    #bottom
    bottom = img[img.shape[0] - r:img.shape[0], 0:img.shape[1]]
    bottom = bottom[::-1]
    new_img[new_img.shape[0] - r:new_img.shape[0], r:img.shape[1] + r] = bottom
    #corners on the bottom
    new_img[new_img.shape[0] - r:new_img.shape[0], 0:r] = bottom[bottom.shape[0] - r:new_img.shape[0], 0:r]
    new_img[new_img.shape[0] - r:new_img.shape[0], new_img.shape[1] - r:new_img.shape[1]] = bottom[bottom.shape[0] - r:new_img.shape[0], img.shape[1] - r:img.shape[1]]

    #left
    left = img[0:img.shape[0], 0:r]
    left = np.flip(left, axis=1)
    new_img[r:img.shape[0] + r, 0:r] = left

    #right
    right = img[0:img.shape[0], img.shape[1] - r:img.shape[1]]
    right = np.flip(right, axis=1)
    new_img[r:img.shape[0] + r, new_img.shape[1] - r:new_img.shape[1]] = right

    #center
    new_img[r: new_img.shape[0] - r, r: new_img.shape[1] - r] = img

    return new_img

def apply_gauss_blur(img, sigma):

    #calculate kernel width
    kernel_width = (sigma * 6) + 1
    kernel_width = round(kernel_width)

    if(kernel_width % 2 == 0):
        kernel_width = kernel_width + 1

    sum = 0
    radius = kernel_width // 2
    for i in range(0 - radius, radius + 1, 1):
        for j in range(0 - radius, radius + 1, 1):
            o_pow = math.pow(sigma, 2)
            x_pow = math.pow(i, 2)
            y_pow = math.pow(j, 2)

            result1 = 1 / (2 * math.pi * o_pow)
            result2 = (-x_pow - y_pow) / (2 * o_pow)
            sum += result1 * (math.e ** result2)

    result = 0
    kernel = []
    radius = kernel_width // 2
    for i in range(0 - radius, radius + 1, 1):
        for j in range(0 - radius, radius + 1, 1):
            o_pow = math.pow(sigma, 2)
            x_pow = math.pow(i, 2)
            y_pow = math.pow(j, 2)

            result1 = 1 / (2 * math.pi * o_pow)
            result2 = (-x_pow - y_pow) / (2 * o_pow)
            result += result1 * (math.e ** result2)

            sum = result / sum
            kernel.append(sum)

    kernel = np.reshape(kernel, [kernel_width, kernel_width])

    border = kernel[0].size // 2
    temp = []
    for i in range(border, img.shape[0] - border):
        for j in range(border, img.shape[1] - border):
            matrix = img[i - border: i + border + 1, j - border: j + border + 1]

            sum = 0
            arr = np.reshape(matrix, -1)
            arr = arr[::-1]
            kernel = np.reshape(kernel, -1)

            for k in range(arr.size):
                sum += arr[k] * kernel[k]

            temp.append(sum)

    temp = np.reshape(temp, [img.shape[0] - (border * 2), img.shape[1] - (border * 2)])
    temp = np.ndarray.astype(temp, np.uint8)

    return temp

# step 1
# Подготовьте одно изображение для экспериментов с гауссовской и лапласовской пирамидой.
apple = imread("apple_bw.bmp")
orange = imread("orange.bmp")


i = apply_gauss_blur(apple, 0.66)
show(i)


