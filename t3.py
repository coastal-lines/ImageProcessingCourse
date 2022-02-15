from skimage.io import imread, imsave
from numpy import array, ndarray, clip


def calculate(img, kernel):
    temp = ndarray(shape=(img.shape[0] - kernel.shape[0] // 2 * 2, img.shape[1] - kernel.shape[0] // 2 * 2), dtype=int)
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            temp[i, j] = int((img[i: i + kernel.shape[0], j: j + kernel.shape[0]] * kernel).sum())
    return clip(temp, 0, 255)


img = imread('img.png')
kernel = 0.1 * array([[-1, -2, -1], [-2, 22, -2], [-1, -2, -1]])
res_img = calculate(img, kernel)

imsave("out_img.png", res_img)
