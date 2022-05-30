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

# step 2.2
# Визуализируйте полученные изображения и амплитуды частот изображений пирамиды
freq1 = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(g1))))
freq2 = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(g2))))
freq3 = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(g3))))
freq4 = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(g4))))
freq5 = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(g5))))

fig = plt.figure(figsize=(16, 8), dpi=100)
columns = 3
rows = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(freq1[:,:,0])

fig.add_subplot(rows, columns, 2)
plt.imshow(freq2[:,:,0])

fig.add_subplot(rows, columns, 3)
plt.imshow(freq3[:,:,0])

fig.add_subplot(rows, columns, 4)
plt.imshow(freq4[:,:,0])

fig.add_subplot(rows, columns, 5)
plt.imshow(freq5[:,:,0])

plt.show()

# и убедитесь, что на каждом слое диапазон частот сужается
#print(np.mean(freq1))
#print(np.mean(freq2))
#print(np.mean(freq3))
#print(np.mean(freq4))
#print(np.mean(freq5))
print(np.mean(freq1) > np.mean(freq2) and np.mean(freq2) > np.mean(freq3) and np.mean(freq3) > np.mean(freq4) and np.mean(freq4) > np.mean(freq5))

# step 2.3
# Постройте пирамиду для трех различных значения сигмы гауссовского ядра