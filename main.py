import numpy as np
import skimage.io
from skimage.io import imread,imsave
import math
import matplotlib.pyplot as plt
from numpy import dstack
from skimage.color import rgb2gray

def show(img):
    plt.imshow(img, cmap='gray')
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
        plt.imshow(array_of_images[i], cmap='gray')

    plt.show()
    plt.clf()

def check_frequencies(array_of_images):

    #freq_array = []
    for i in range(len(array_of_images) - 1):
        freq = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(array_of_images[i]))))
        freq_next = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(array_of_images[i + 1]))))
        if(np.mean(freq) < np.mean(freq_next)):
            print("!!! something wrong with freq !!!")

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

    #new_img = skimage.img_as_float(new_img)

    return new_img

def blur(img, sigma):

    #prepare kernel
    kernel_width = round((sigma * 6) + 1)
    kernel_radius = kernel_width // 2
    kernel = []
    for x in range(0 - kernel_radius, kernel_radius + 1, 1):
        for y in range(0 - kernel_radius, kernel_radius + 1, 1):
            result = (1 / (2 * math.pi * math.pow(sigma, 2))) * (math.e ** ((-math.pow(x, 2) - math.pow(y, 2)) / (2 * math.pow(sigma, 2))))
            result = format(result, ".5f")
            kernel.append(float(result))

    #extend image
    #temp_img = extend_image(img, kernel_width)

    temp_img = extend_image(img, kernel_width)

    #apply kernel to image
    #border = kernel[0].size // 2
    border = kernel_radius
    temp = []
    for i in range(border, temp_img.shape[0] - border):
        for j in range(border, temp_img.shape[1] - border):
            matrix = temp_img[i - border: i + border + 1, j - border: j + border + 1]
            matrix = np.reshape(matrix, -1)
            sum_all_pixels = np.sum(matrix)
            sum_all_items_of_kernel = np.sum(kernel)
            sum_for_current_pixel = sum_all_pixels * sum_all_items_of_kernel
            current_pixel = sum_for_current_pixel / kernel_width
            temp.append(current_pixel)

    blur = np.reshape(temp, [temp_img.shape[0] - (border * 2), img.shape[1] - (border * 2)])

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

    for i in range(n_layers):
        if(i == 0):
            #temp_img = skimage.img_as_float(img) - skimage.img_as_float(images_gauss_pyramid[i])
            temp_img = img - images_gauss_pyramid[i]
        else:
            #temp_img = skimage.img_as_float(images_gauss_pyramid[i - 1]) - skimage.img_as_float(images_gauss_pyramid[i])
            temp_img = images_gauss_pyramid[i - 1] - images_gauss_pyramid[i]
        images_laplac_pyramid.append(temp_img)

    return images_laplac_pyramid, images_gauss_pyramid

def CombineTwoImages(img1, img2, mask, sigma, n_layers):
    # step 1

    LA, gauss_LA = laplacian_pyramid(img1, sigma, layers)
    LB, gauss_LB = laplacian_pyramid(img2, sigma, layers)

    last_sigma = sigma[-1]
    sigma.append(last_sigma)
    GM = gauss_pyramid(mask, sigma, layers + 1)

    #how_images(gauss_LA)
    #show_images(LA)
    #show_images(gauss_LB)
    #show_images(LB)
    #show_images(GM)

    LA.append(gauss_LA[-1])
    LB.append(gauss_LB[-1])

    temp = []
    for i in range(n_layers + 1):
        t = LA[i] * mask + LB[i] * (1.0 - mask)
        #t = np.clip(t, 0.0, 1.0)
        temp.append(t)

    LS = 0
    for i in range(n_layers + 1):
        LS = LS + temp[i]

    return LS

#g = tuple(skimage.transform.pyramid_gaussian(img1, downscale=1.1, max_layer = 3))
#l = tuple(skimage.transform.pyramid_laplacian(img1, downscale=1.1, max_layer = 3))
#show_images(l)
#show_images(g)


img1 = imread("data/img1.bmp", 0)
#img1 = skimage.img_as_float(img1)
img2 = imread("data/img2.bmp", 0)
#img2 = skimage.img_as_float(img2)
mask = imread("data/mask.bmp", 0)
#mask = skimage.img_as_float(mask)
sigma = [1, 1, 1]
layers = len(sigma)

b = gauss_pyramid(img1, sigma, layers)
show_images(b)

#result = CombineTwoImages(img1, img2, mask, sigma, layers)
#show(result)


#sigma - – стандартное отклонение нормального распределения
# check_frequencies(images_gauss_pyramid)
# show_frequencies_and_images(images_laplac_pyramid)