import skimage
from skimage.io import imread, imshow
import matplotlib as plt
from matplotlib import pyplot as plt
from matplotlib.pyplot import hist
import numpy as np

img = imread('tiger-low-contrast.png')
#imshow(img)
#plt.show()

values, bin_edges, patches = hist(img.ravel(), bins=range(257))

x_min = np.min(img)
x_max = np.max(img)
value = 255 / (x_max - x_min)

#can be done by one line - faster that by loop
#img = (img - x_min) * (value)

h = img.shape[0]
w = img.shape[1]

for j in range(img.shape[0]):
    for i in range(img.shape[1]):
        temp = (img[j,i] - x_min) * value
        img[j,i] = temp

img = img.astype('uint8')

print(np.array_equal(img, imread('tiger-high-contrast.png')))

imshow(img)
plt.show()