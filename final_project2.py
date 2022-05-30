import numpy as np
import skimage.io
from skimage.io import imread,imsave
import math
import matplotlib.pyplot as plt

def show(img):
    plt.imshow(img)
    plt.show()

# step 1
# Подготовьте одно изображение для экспериментов с гауссовской и лапласовской пирамидой.
apple = imread("apple.bmp")
orange = imread("orange.bmp")

#step 2.1
#Постройте гауссовскую пирамиду изображения из не менее чем пяти слоев
g1 = skimage.filters.gaussian(apple, sigma=1, mode='nearest')
g2 = skimage.filters.gaussian(g1, sigma=3, mode='nearest')
g3 = skimage.filters.gaussian(g2, sigma=5, mode='nearest')
g4 = skimage.filters.gaussian(g3, sigma=7, mode='nearest')
g5 = skimage.filters.gaussian(g4, sigma=9, mode='nearest')


# step 2.3
# Постройте пирамиду для трех различных значения сигмы гауссовского ядра
def gauss_pyramide(img, sigma, n_layers):

    result = []
    sigma_count = 0
    temp_gauss = skimage.filters.gaussian(img, sigma=sigma[sigma_count], mode='nearest')
    sigma_count += 1

    for l in range(1, n_layers):
        temp_gauss = skimage.filters.gaussian(temp_gauss, sigma=sigma[sigma_count], mode='nearest')
        result.append(temp_gauss)

        if(sigma_count == len(sigma)):
            sigma == 0

    return result

r = gauss_pyramide(apple, [1,3,5], 6)

print("")