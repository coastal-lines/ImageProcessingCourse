from skimage.io import imread
import numpy as np
from numpy import histogram

img = imread('img.png')

threshold = round(img.size * 0.05)

vector = np.reshape(img, -1)
vector.sort()
v_min = vector[0]
v_max = vector[-1]
#print(v_min)
#print(v_max)

values, bin_edges = histogram(vector.ravel(), bins=range(257))

count = 0
x_min_shifted = 0
for i in range(256):
    count += values[i]
    if count > threshold:
        x_min_shifted = i
        #print(x_min_shifted)
        break

count = 0
x_max_shifted = 0
for i in range(255, 0, -1):
    count += values[i]
    if count > threshold:
        x_max_shifted = i
        #print(x_max_shifted)
        break

print(str(x_min_shifted) + " " + str(x_max_shifted))