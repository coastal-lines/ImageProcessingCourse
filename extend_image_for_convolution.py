import numpy as np
import math
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

def show(img):
    plt.imshow(img)
    plt.show()

img = imread("apple.bmp")
print(img.shape)

#extend image if kernel equals 7*7
#it means that we have to extend each border to 3 pixels

#the mirror method


