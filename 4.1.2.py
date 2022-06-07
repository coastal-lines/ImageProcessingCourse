import numpy as np
import skimage.io
from skimage.io import imread,imsave

#Box filter for image

img = skimage.io.imread("tiger-gray-small.png")
kernel = np.array([[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]])

#prepare border size - we are not going to extend the image or any other solution for image's borders
#it means that result image will be smaller than original
border = kernel[0].size // 2

#sum of calculating elements - see how Box filter works
def calculate(arr, kernel):
    sum = 0

    #array to vector
    arr = np.reshape(arr, -1)
    #reverse vector
    arr = arr[::-1]
    kernel = np.reshape(kernel, -1)

    #multiply
    for i in range(arr.size):
            temp = arr[i] * kernel[i]
            sum += temp

    return sum

temp = []
for i in range(border, img.shape[0] - border):
    for j in range(border, img.shape[1] - border):
        #prepare roi
        matrix = img[i - border : i + border + 1, j - border: j + border + 1]
        #convolution
        res = calculate(matrix, kernel)
        temp.append(res)

temp = np.reshape(temp, [img.shape[0] - (border*2), img.shape[1] - (border*2)])

#now we have to divide result into size of kerner (25 pixels)
temp = temp / 25
temp = temp.astype(int)

#temp = scipy.signal.convolve2d(img, kernel, mode='valid', boundary='fill', fillvalue=1)
#temp = temp / 25
#temp = temp.astype(int)

imsave('out_img_box_filter.png', temp)
test = imread('box-tiger.png')
print(np.array_equal(temp, test))