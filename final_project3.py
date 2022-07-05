import numpy as np
import skimage.io
from skimage.io import imread,imsave
import math
import matplotlib.pyplot as plt
from numpy import dstack
from skimage.color import rgb2gray

def show(img):
    plt.imshow(img)
    plt.show()

def show_frequencies_and_images(array_of_images):
    fig = plt.figure(figsize=(16, 8), dpi=100)
    columns = len(array_of_images)
    #rows = 2

    for i in range(len(array_of_images)):
        #1 - first line for images
        fig.add_subplot(1, columns, i + 1)
        plt.imshow(array_of_images[i])

        #2 - second line for frequencies
        freq = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(array_of_images[i]))))
        fig.add_subplot(2, columns, i + 1)
        plt.imshow(freq)

    plt.show()
    plt.clf()

def show_images(array_of_images):
    fig = plt.figure(figsize=(16, 8), dpi=100)
    columns = len(array_of_images)
    #rows = 2

    for i in range(len(array_of_images)):
        #1 - first line for images
        fig.add_subplot(1, columns, i + 1)
        plt.imshow(array_of_images[i])

    plt.show()
    plt.clf()

def check_frequencies(array_of_images):

    #freq_array = []
    for i in range(len(array_of_images) - 1):
        freq = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(array_of_images[i]))))
        freq_next = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(array_of_images[i + 1]))))
        if(np.mean(freq) < np.mean(freq_next)):
            print("!!! something wrong with freq !!!")

def prepare_kernel(sigma):
    #prepare kernel
    kernel_width = round((sigma * 6) + 1)
    kernel_radius = kernel_width // 2
    kernel = []
    for x in range(0 - kernel_radius, kernel_radius + 1, 1):
        for y in range(0 - kernel_radius, kernel_radius + 1, 1):
            result = (1 / (2 * math.pi * math.pow(sigma, 2))) * (math.e ** ((-math.pow(x, 2) - math.pow(y, 2)) / (2 * math.pow(sigma, 2))))
            result = format(result, ".5f")
            kernel.append(float(result))

    return kernel, kernel_radius, kernel_width

def extend_image(img, kernel_width):

    #img = skimage.img_as_uint(img)

    #extend image if kernel equals 7*7
    #it means that we have to extend each border to 3 pixels
    #and corners

    #the mirror method
    #get radius
    r = kernel_width // 2

    #make new extended image
    if(len(img.shape) == 3):
        #rgb
        new_img = np.zeros((img.shape[0] + (r * 2), img.shape[1] + (r * 2), img.shape[2]) ,np.uint8)
    else:
        #bw
        new_img = np.zeros((img.shape[0] + (r * 2), img.shape[1] + (r * 2)), np.uint8)

    #top
    top = img[0:r, 0:img.shape[1]]
    top = top[::-1]
    new_img[0:r, r:img.shape[1] + r] = top
    #corners on the top
    new_img[0:r, 0:r] = top[0:r, 0:r]
    new_img[0:r, new_img.shape[1] - r: new_img.shape[1]] = img[0:r, img.shape[1] - r: img.shape[1]]

    #bottom
    bottom = img[img.shape[0] - r:img.shape[0], 0:img.shape[1]]
    bottom = bottom[::-1]
    new_img[new_img.shape[0] - r:new_img.shape[0], r:img.shape[1] + r] = bottom
    #corners on the bottom
    new_img[new_img.shape[0] - r:new_img.shape[0], 0:r] = bottom[bottom.shape[0] - r:new_img.shape[0], 0:r]
    new_img[new_img.shape[0] - r:new_img.shape[0], new_img.shape[1] - r:new_img.shape[1]] = bottom[bottom.shape[0] - r:new_img.shape[0], img.shape[1] - r:img.shape[1]]

    #left
    left = img[0:img.shape[0], 0:r]
    left = np.flip(left, axis=1)
    new_img[r:img.shape[0] + r, 0:r] = left

    #right
    right = img[0:img.shape[0], img.shape[1] - r:img.shape[1]]
    right = np.flip(right, axis=1)
    new_img[r:img.shape[0] + r, new_img.shape[1] - r:new_img.shape[1]] = right

    #center
    new_img[r: new_img.shape[0] - r, r: new_img.shape[1] - r] = img

    return new_img

def blur(img, sigma):

    #prepare kernel
    kernel, kernel_radius, kernel_width = prepare_kernel(sigma)

    img = extend_image(img, kernel_width)

    #apply kernel to image
    border = kernel_radius

    temp = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)

    for c in range(img.shape[2]):
        for i in range(border, img.shape[0] - border):
            for j in range(border, img.shape[1] - border):
                matrix = img[i - border: i + border + 1, j - border: j + border + 1]
                sum = 0
                test = matrix[:, :, c]
                arr = np.reshape(test, -1)
                k = np.reshape(kernel, -1)
                for m in range(arr.size):
                    sum += arr[m] * k[m]
                temp[i, j, c] = np.clip(sum, 0, 255)

    blur = temp[border : img.shape[0] - border, border : img.shape[1] - border, :]
    return blur

def gauss_pyramid(img, sigma, n_layers):

    images_gauss_pyramid = []
    temp_img = None

    for i in range(n_layers):
        if(i == 0):
            temp_img = blur(img, sigma[i])
        else:
            temp_img = blur(temp_img, sigma[i])
        images_gauss_pyramid.append(temp_img)

    return images_gauss_pyramid

def laplacian_pyramid(img, sigma, n_layers):

    images_laplac_pyramid = []
    images_gauss_pyramid = gauss_pyramid(img, sigma, n_layers)

    #show_images(images_gauss_pyramid)

    for i in range(n_layers):
        if(i == 0):
            temp_img = img - images_gauss_pyramid[i]
            temp_img = np.int32(img) - np.int32(images_gauss_pyramid[i])
            temp_img2 = np.invert(np.array(temp_img, dtype=np.int8))
        else:
            temp_img = images_gauss_pyramid[i - 1] - images_gauss_pyramid[i]
            temp_img = np.int32(images_gauss_pyramid[i - 1]) - np.int32(images_gauss_pyramid[i])
            temp_img2 = np.invert(np.array(temp_img, dtype=np.int8))
        images_laplac_pyramid.append(temp_img2)

    return images_laplac_pyramid, images_gauss_pyramid

def CombineTwoImages(img1, img2, mask, sigma, n_layers):
    # step 1

    LA, gauss_LA = laplacian_pyramid(img1, sigma, layers)
    LB, gauss_LB = laplacian_pyramid(img2, sigma, layers)

    #show_images(gauss_LA)
    #show_images(gauss_LB)

    last_sigma = sigma[-1]
    sigma.append(last_sigma)
    GM = gauss_pyramid(mask, sigma, layers)

    LA.append(gauss_LA[-1])
    LB.append(gauss_LB[-1])
    GM.append(GM[-1])

    temp = []
    for i in range(layers + 1):
        #t = None
        t = (LA[i] * GM[i]) + LB[i] * (255 - GM[i])
        temp.append(t)

    LS = 0
    for i in range(n_layers + 1):
        LS = LS + temp[i]

    LS = np.clip(LS,0,255)

    return LS


img1 = imread("data3/img1.bmp")
img2 = imread("data3/img2.bmp")
mask = imread("data3/mask.bmp")
mask_bw = rgb2gray(mask)
sigma = [0.5, 1, 1.5]
layers = len(sigma)


result = CombineTwoImages(img1, img2, mask, sigma, layers)
show(result)

#sigma - – стандартное отклонение нормального распределения
# check_frequencies(images_gauss_pyramid)
# show_frequencies_and_images(images_laplac_pyramid)
