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


fig = plt.figure(figsize=(16, 8), dpi=100)
columns = 3
rows = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(apple)
fig.add_subplot(rows, columns, 2)
plt.imshow(g1)
fig.add_subplot(rows, columns, 3)
plt.imshow(g2)
fig.add_subplot(rows, columns, 4)
plt.imshow(g3)
fig.add_subplot(rows, columns, 5)
plt.imshow(g4)
fig.add_subplot(rows, columns, 6)
plt.imshow(g5)

plt.show()
print("")
plt.clf()