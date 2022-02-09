import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte

img = imread(r'3.2\img.png')
img_f = skimage.img_as_float(img)

R_averaged = img_f[:,:,0] // img_f[:,:,0].size
G_averaged = img_f[:,:,1]
B_averaged = img_f[:,:,2]