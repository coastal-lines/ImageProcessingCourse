import numpy as np
import math
import scipy
from numpy import uint8
from skimage import img_as_ubyte
from skimage.io import imread, imsave
import scipy.signal

img = imread("tiger-gray-small.png")

kernel = np.array([[-1, -2, -1],[-2, 22, -2],[-1, -2, -1]], dtype=int)*0.1

res = scipy.signal.convolve2d(img, kernel, mode='valid', boundary='fill', fillvalue=1)
res = np.clip(res,0,255)
res = res.astype('uint8')
test = imread('unsharp-tiger.png')

a = []
for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        if(res[i,j] != test[i,j]):
            a.append([i,j,test[i,j].astype('uint8')])
            print([i,j,test[i,j].astype('uint8')])

np.savetxt("t3.txt", a, delimiter=",", fmt='%s')

