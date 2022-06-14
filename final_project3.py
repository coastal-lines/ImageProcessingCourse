import numpy as np
import skimage.io
from skimage.io import imread,imsave
import math
import matplotlib.pyplot as plt

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

def check_frequencies(array_of_images):

    #freq_array = []
    for i in range(len(array_of_images) - 1):
        freq = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(array_of_images[i]))))
        freq_next = np.log(1 + abs(np.fft.fftshift(np.fft.fft2(array_of_images[i + 1]))))
        if(np.mean(freq) < np.mean(freq_next)):
            print("!!! something wrong with freq !!!")

def extend_image(img, kernel_width):

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

    #calculate kernel width
    kernel_width = round((sigma * 6) + 1)

    img = extend_image(img, kernel_width)

    if(kernel_width % 2 == 0):
        kernel_width = kernel_width + 1

    #calculate sum of all pixels in kernel
    sum_of_all_kernel_items = 0
    kernel_radius = kernel_width // 2
    for x in range(0 - kernel_radius, kernel_radius + 1, 1):
        for y in range(0 - kernel_radius, kernel_radius + 1, 1):
            o_pow = math.pow(sigma, 2)
            x_pow = math.pow(x, 2)
            y_pow = math.pow(y, 2)
            sum_in_current_pixel = (1 / (2 * math.pi * o_pow)) * (math.e ** ((-x_pow - y_pow) / (2 * o_pow)))
            sum_of_all_kernel_items += sum_in_current_pixel

    # prepare kernel
    kernel = []

    for x in range(0 - kernel_radius, kernel_radius + 1, 1):
        for y in range(0 - kernel_radius, kernel_radius + 1, 1):
            o_pow = math.pow(sigma, 2)
            x_pow = math.pow(x, 2)
            y_pow = math.pow(y, 2)
            sum_in_current_pixel = (1 / (2 * math.pi * o_pow)) * (math.e ** ((-x_pow - y_pow) / (2 * o_pow)))
            item = sum_in_current_pixel / sum_of_all_kernel_items
            kernel.append(item)

    kernel = np.reshape(kernel, [kernel_width, kernel_width])

    #apply kernel
    border = kernel[0].size // 2
    temp = []
    for i in range(border, img.shape[0] - border):
        for j in range(border, img.shape[1] - border):
            matrix = img[i - border: i + border + 1, j - border: j + border + 1]
            #res = apply_kernel(matrix, kernel)
            sum = 0
            arr = np.reshape(matrix, -1)
            arr = arr[::-1]
            k = np.reshape(kernel, -1)

            for m in range(matrix.size):
                sum += arr[m] * k[m]

            temp.append(sum)

    blur = np.reshape(temp, [img.shape[0] - (border * 2), img.shape[1] - (border * 2)])
    blur = np.ndarray.astype(blur, np.uint8)
    return blur

def gauss_pyramid(img, sigma, n_layers):

    images_gauss_pyramid = []
    #temp_img = blur(img, sigma[0])

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

    for i in range(n_layers - 1):
        if(i == 0):
            temp_img = img - images_gauss_pyramid[i]
        else:
            temp_img = images_gauss_pyramid[i] - images_gauss_pyramid[i + 1]
        images_laplac_pyramid.append(temp_img)

    return images_laplac_pyramid

# step 1
#
apple = imread("lena_bw2.jpg")
#apple = imread("apple_bw.bmp")
orange = imread("orange.bmp")


images_laplac_pyramid = laplacian_pyramid(apple, [1,3,5], 3)
#check_frequencies(images_gauss_pyramid)
#show_frequencies_and_images(images_laplac_pyramid)

