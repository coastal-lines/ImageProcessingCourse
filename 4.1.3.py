import numpy as np
import skimage.io
from skimage.io import imread,imsave
import math

def gauss(arr):
    o = arr[0]
    x = arr[1]
    y = arr[2]

    x_pow = math.pow(x,2)
    y_pow = math.pow(y,2)
    o_pow = math.pow(o,2)

    p1 = 1 / (2 * math.pi * o_pow)
    p2 = (-x_pow - y_pow) / (2 * o_pow)
    p3 = p1 * (math.e ** p2)

    return p3

print(gauss([1,1,1]))
print(gauss([1,1,1]) == 0.05854983152431917)

print(gauss([5,1,1]))
print(gauss([5,1,1]) == 0.006116575540463281)