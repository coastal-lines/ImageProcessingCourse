from skimage.io import imread, imsave
img = imread('img.png')
h = img.shape[0] // 2
w = img.shape[1] // 2
img[h - 3: h + 4, w - 7:w + 8] = [255, 192, 203]
imsave('out_img.png', img)