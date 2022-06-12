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
w = 7

#the mirror method
#get radius
r = w // 2

#make new extended image
new_img = np.zeros((img.shape[0] + (r * 2), img.shape[1] + (r * 2), img.shape[2]) ,np.uint8)
print(new_img.shape)

#top
top = img[0:r, 0:img.shape[1]]
print(top.shape)
#bottom
bottom = img[0:r, 0:img.shape[1]]