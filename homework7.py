import skimage.io
from skimage.io import imread, imsave
import warnings
warnings.simplefilter("ignore")

img = imread('img.png')
img_f = skimage.img_as_float(img)
y = skimage.img_as_ubyte((img_f[:,:,0] * 0.2126) + (img_f[:,:,1] * 0.7152) + (img_f[:,:,2] * 0.0722))
imsave('out_img.png', y)