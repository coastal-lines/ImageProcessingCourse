import numpy as np
import scipy.signal

#this is image
arr = np.array([[0, -8, -3, -2], [1, 9, -8, 0], [9, -4, 5, -9], [6, -4, 6, 3]])
#this is kerner
kernel = np.array([[4, -5, 4], [-6, -8, -2], [1, 5, 5]])

#make convolution of image and kernel
#this is result
res = scipy.signal.convolve2d(arr, kernel, mode='valid')
print(res)