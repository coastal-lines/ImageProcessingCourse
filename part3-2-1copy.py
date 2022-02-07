import skimage
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
import numpy as np

img = imread('tiger-color.png')
r = img_as_float(img[:,:,0])
g = img_as_float(img[:,:,1])
b = img_as_float(img[:,:,2])
Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
U = -0.0999 * r - 0.3360 * g + 0.4360 * b
V = 0.6150 * r - 0.5586 * g + 0.0563 * b

#Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
#U = -0.0999 * r - 0.3360 * g + 0.4360 * b
#V = 0.6150 * r - 0.5586 * g - 0.0563 * b

x_min = np.percentile(Y.reshape(-1), 5)
x_max = np.percentile(Y.reshape(-1), 95)
Y = (Y - x_min) * (1 /(x_max - x_min))
Y = np.clip(Y, 0, 1.0)

R = Y + 1.2803 * V
G = Y - 0.2148 * U - 0.3805 * V
B = Y + 2.1279 * U
R = img_as_ubyte(np.clip(R, 0, 1.0))
G = img_as_ubyte(np.clip(G, 0, 1.0))
B = img_as_ubyte(np.clip(B, 0, 1.0))
rgb = np.dstack((R,G,B))
imsave('out_img.png', rgb)
print(np.array_equal(rgb, imread(r'C:\Users\User\Downloads\tiger-stable-contrast.png')))