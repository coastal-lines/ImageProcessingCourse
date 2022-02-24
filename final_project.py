import numpy as np
import skimage.io
from skimage.io import imread,imsave
import math
import matplotlib.pyplot as plt

def show(img):
    plt.imshow(img)
    plt.show()

#step 1
apple = imread("apple.bmp")
orange = imread("orange.bmp")

#step 2
#Постройте гауссовскую пирамиду изображения из не менее чем пяти слоев
g1 = skimage.filters.gaussian(apple, sigma=2, mode='nearest')
g2 = skimage.filters.gaussian(g1, sigma=2, mode='nearest')
g3 = skimage.filters.gaussian(g2, sigma=3, mode='nearest')
g4 = skimage.filters.gaussian(g3, sigma=3, mode='nearest')
g5 = skimage.filters.gaussian(g4, sigma=4, mode='nearest')

freq = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(g1))))

show(freq[:,:,0])