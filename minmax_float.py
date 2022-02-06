import skimage.io
from skimage.io import imread, imshow
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt
import numpy as np

#int
print("int: ")
img = imread('img_.png')
values, bin_edges, patches = hist(img[:,:,0].ravel(), bins=range(257))
#plt.show()

threshold1 = round(img[:,:,0].size * 0.05)
vector = np.reshape(img[:,:,0], -1)
vector.sort()
values, bin_edges, patches = hist(vector, bins=range(257))
#plt.show()

count = 0
x_min_shifted = 0
for i in range(256):
    count += values[i]
    if count > threshold1:
        x_min_shifted = i
        print(x_min_shifted)
        break

count = 0
x_max_shifted = 0
for i in range(255, 0, -1):
    count += values[i]
    if count > threshold1:
        x_max_shifted = i
        print(x_max_shifted)
        break

#float
print("float: ")
img_f = skimage.img_as_float(img[:,:,0])
vector2 = np.reshape(img_f, -1)
threshold2 = round(img_f.size * 0.05)
vector2 = sorted(vector2)

min_float = vector2[threshold2 + 1]
max_float = vector2[-(threshold2 + 1)]
print(min_float)
print(max_float)