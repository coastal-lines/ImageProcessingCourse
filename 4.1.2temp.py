import numpy as np
import scipy.signal
import skimage.io
from skimage.io import imread,imsave

img = skimage.io.imread("tiger-gray-small.png")

a = np.array([[255,255,222,222,234],
                [255,255,222,222,234],
                [255,255,222,222,234],
                [255,255,222,222,234],
                [255,255,222,222,234]])

kernel = np.array([[1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1]])

def calculate(arr):
    l1 = (arr[0][] + arr[0][] + arr[0][] + arr[0][] + arr[0][])
    l2 = (arr[5][] + arr[6][] + arr[7][] + arr[8][] + arr[9][])
    l3 = (arr[10][] + arr[11][] + arr[12][] + arr[13][] + arr[14][])
    l4 = (arr[15][] + arr[16][] + arr[17][] + arr[18][] + arr[19][])
    l5 = (arr[20][] + arr[21][] + arr[22][] + arr[23][] + arr[24][])
    p = (l1 + l2 + l3 + l4 + l5) / 9

    return p

t = calculate(a)

print("")

#res = scipy.signal.convolve2d(img, kernel, mode='valid', boundary='fill', fillvalue=1)
#res = res / res.max() * 255
#res = res.astype(int)

#test = imread('box-tiger.png')
#imsave('out_img.png', res)
#print(np.array_equal(res, test))