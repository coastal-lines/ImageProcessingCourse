import skimage.io
from skimage.io import imread, imsave
import numpy as np
from numpy import histogram

img = imread('img.png')
threshold = round(img.size * 0.05)

vector = np.reshape(img, -1)
vector.sort()
values, bin_edges = histogram(vector.ravel(), bins=range(257))

count = 0
x_max_shifted = 0
for i in range(255, 0, -1):
    count += values[i]
    if count > threshold:
        x_max_shifted = i
        break

count = 0
x_min_shifted = 0
for i in range(256):
    count += values[i]
    if count > threshold:
        x_min_shifted = i
        break

img = imread('img.png')
img_f = skimage.img_as_float(img)

value = 255 / (x_max_shifted - x_min_shifted)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pixel = (img[i,j] - x_min_shifted) * value
        if(pixel < 0):
            img[i,j] = 0
            continue

        if(pixel > 255):
            img[i,j] = 255
            continue

        img[i,j] = pixel

img = img.astype('uint8')
imsave("out.png", img)

print(np.array_equal(img, imread('tiger-stable-contrast.png')))