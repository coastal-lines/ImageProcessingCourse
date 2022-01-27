from skimage.io import imread
import numpy as np

img = imread('img.png')
h = img.shape[0]
w = img.shape[1]
color = img[0, 0]
centerH = h // 2
centerW = w // 2

leftBorder = 0
for i in range(centerW):
    if(np.array_equal(img[centerH, i], color)):
        leftBorder += 1

rightBorder = 0
for i in range(centerW, w):
    if(np.array_equal(img[centerH, i], color)):
        rightBorder += 1

topBorder = 0
for i in range(centerH):
    if(np.array_equal(img[i, centerW], color)):
        topBorder += 1


bottomBorder = 0
for i in range(centerH, h):
    if(np.array_equal(img[i, centerW], color)):
        bottomBorder += 1

print(str(leftBorder) + " " + str(topBorder) + " " + str(rightBorder) + " " + str(bottomBorder))