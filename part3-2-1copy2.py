import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.util import img_as_float, img_as_ubyte
from collections import Counter

im = imread('tiger-color.png')
red = img_as_float(im[:,:,0])
green = img_as_float(im[:,:,1])
blue = img_as_float(im[:,:,2])

Y = 0.2126 * red + 0.7152 * green + 0.0722 * blue
U = -0.0999 * red - 0.3360 * green + 0.4360 * blue
V = 0.6150 * red - 0.5586 * green - 0.0563 * blue

min_p = np.percentile(Y.reshape(-1), 5)
max_p = np.percentile(Y.reshape(-1), 95)


Y = (Y - min_p) * (1 /(max_p - min_p))

Y = np.clip(Y, 0, 1.0)


R = Y + 1.2803 * V
G = Y - 0.2148 * U - 0.3805 * V
B = Y + 2.1279 * U

R = img_as_ubyte(np.clip(R, 0, 1.0))
G = img_as_ubyte(np.clip(G, 0, 1.0))
B = img_as_ubyte(np.clip(B, 0, 1.0))


rgb = np.dstack((R,G,B))
imsave('out_img.png', rgb)


#test
print(np.array_equal(rgb, imread(r'C:\Users\User\Downloads\tiger-stable-contrast.png')))