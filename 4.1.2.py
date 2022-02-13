import numpy as np
import skimage.io
from skimage.io import imread,imsave

img = skimage.io.imread("tiger-gray-small.png")

kernel = np.array([[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]])

border = kernel[0].size // 2

def calculate(arr, kernel):
    sum = 0

    arr = np.reshape(arr, -1)
    arr = arr[::-1]
    kernel = np.reshape(kernel, -1)

    for i in range(arr.size):
            temp = arr[i] * kernel[i]
            sum += temp

    return sum

temp = []
for i in range(border, img.shape[0] - border):
    for j in range(border, img.shape[1] - border):
        matrix = img[i - border : i + border + 1, j - border: j + border + 1]
        res = calculate(matrix, kernel)
        temp.append(res)

temp = np.reshape(temp, [img.shape[0] - (border*2), img.shape[1] - (border*2)])
temp = temp / 25
temp = temp.astype(int)

#temp = scipy.signal.convolve2d(img, kernel, mode='valid', boundary='fill', fillvalue=1)
#temp = temp / 25
#temp = temp.astype(int)

imsave('out_img.png', temp)
test = imread('box-tiger.png')
print(np.array_equal(temp, test))