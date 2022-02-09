import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte

img = imread(r'3.2\img.png')
img_f = skimage.img_as_float(img)

R_averaged = img_f[:,:,0].ravel().mean()
G_averaged = img_f[:,:,1].ravel().mean()
B_averaged = img_f[:,:,2].ravel().mean()

RGB_averaged = R_averaged + G_averaged + B_averaged
AVG = RGB_averaged / 3

r_w = R_averaged / AVG
g_w = G_averaged / AVG
b_w = B_averaged / AVG

img_f = skimage.img_as_float(img)
R = img_f[:,:,0] / r_w
G = img_f[:,:,1] / g_w
B = img_f[:,:,2] / b_w

R = np.clip(R, 0, 1.0)
G = np.clip(G, 0, 1.0)
B = np.clip(B, 0, 1.0)

R = img_as_ubyte(R)
G = img_as_ubyte(G)
B = img_as_ubyte(B)
rgb = np.dstack((R,G,B))
imsave(r'3.2\out_img.png', rgb)

print(np.array_equal(rgb, imread(r'3.2\railroad-gray-world.png')))