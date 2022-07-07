import numpy as np
from skimage.io import imread
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
        else:
            print("freq is fine")

def extend_image(img, kernel_width, kernel_radius):

    if(len(img.shape) == 3):
        new_img = np.zeros((img.shape[0] + (kernel_radius * 2), img.shape[1] + (kernel_radius * 2), img.shape[2]))
    else:
        new_img = np.zeros((img.shape[0] + (kernel_radius * 2), img.shape[1] + (kernel_radius * 2)))

    #top
    top = img[0:kernel_radius, 0:img.shape[1]]
    top = top[::-1]
    new_img[0:kernel_radius, kernel_radius:img.shape[1] + kernel_radius] = top
    #corners on the top
    new_img[0:kernel_radius, 0:kernel_radius] = top[0:kernel_radius, 0:kernel_radius]
    new_img[0:kernel_radius, new_img.shape[1] - kernel_radius: new_img.shape[1]] = img[0:kernel_radius, img.shape[1] - kernel_radius: img.shape[1]]

    #bottom
    bottom = img[img.shape[0] - kernel_radius:img.shape[0], 0:img.shape[1]]
    bottom = bottom[::-1]
    new_img[new_img.shape[0] - kernel_radius:new_img.shape[0], kernel_radius:img.shape[1] + kernel_radius] = bottom
    #corners on the bottom
    new_img[new_img.shape[0] - kernel_radius:new_img.shape[0], 0:kernel_radius] = bottom[bottom.shape[0] - kernel_radius:new_img.shape[0], 0:kernel_radius]
    new_img[new_img.shape[0] - kernel_radius:new_img.shape[0], new_img.shape[1] - kernel_radius:new_img.shape[1]] = bottom[bottom.shape[0] - kernel_radius:new_img.shape[0], img.shape[1] - kernel_radius:img.shape[1]]

    #left
    left = img[0:img.shape[0], 0:kernel_radius]
    left = np.flip(left, axis=1)
    new_img[kernel_radius:img.shape[0] + kernel_radius, 0:kernel_radius] = left

    #right
    right = img[0:img.shape[0], img.shape[1] - kernel_radius:img.shape[1]]
    right = np.flip(right, axis=1)
    new_img[kernel_radius:img.shape[0] + kernel_radius, new_img.shape[1] - kernel_radius:new_img.shape[1]] = right

    #center
    new_img[kernel_radius: new_img.shape[0] - kernel_radius, kernel_radius: new_img.shape[1] - kernel_radius] = img

    return new_img

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

    return kernel, kernel_width, kernel_radius

def blur(img, sigma):

    #prepare kernel
    kernel, kernel_width, kernel_radius = prepare_kernel(sigma)

    #extend image
    img = extend_image(img, kernel_width, kernel_radius)

    #apply kernel to image
    border = kernel_radius
    temp = []
    for i in range(border, img.shape[0] - border):
        for j in range(border, img.shape[1] - border):
            matrix = img[i - border: i + border + 1, j - border: j + border + 1]

            sum = 0
            arr = np.reshape(matrix, -1)
            k = np.reshape(kernel, -1)

            for m in range(matrix.size):
                sum += arr[m] * k[m]

            temp.append(sum)

    blur = np.reshape(temp, [img.shape[0] - (border * 2), img.shape[1] - (border * 2)])

    return blur

def gauss_pyramid(img, sigma_arr, n_layers):

    images_gauss_pyramid = []
    temp_img = None

    for i in range(n_layers):
        if(i == 0):
            temp_img = blur(img, sigma_arr[i])
        else:
            temp_img = blur(temp_img, sigma_arr[i])
        images_gauss_pyramid.append(temp_img)

    return images_gauss_pyramid

def laplacian_pyramid(img, sigma_arr, n_layers):

    images_laplacian_pyramid = []
    images_gauss_pyramid = gauss_pyramid(img, sigma_arr, n_layers)

    for i in range(n_layers):
        if(i == 0):
            temp_img = img - images_gauss_pyramid[i]
        else:
            temp_img = images_gauss_pyramid[i - 1] - images_gauss_pyramid[i]
        images_laplacian_pyramid.append(temp_img)

    return images_laplacian_pyramid, images_gauss_pyramid

def CombineTwoImages(img1, img2, mask, sigma_arr, n_layers):

    LA, gauss_LA = laplacian_pyramid(img1, sigma_arr, n_layers)
    LB, gauss_LB = laplacian_pyramid(img2, sigma_arr, n_layers)
    GM = gauss_pyramid(mask, sigma_arr, n_layers)

    LA.append(gauss_LA[-1])
    LB.append(gauss_LB[-1])
    GM.append(GM[-1])

    laplacian_combinations = []
    temp_img = 0
    for i in range(n_layers + 1):
        temp = (LA[i] * GM[i]) + LB[i] * (255 - GM[i])
        print("temp")
        laplacian_combinations.append(temp)
        temp_img += temp

    return temp_img, laplacian_combinations

def Run(img1, img2, mask, sigma_arr):

    layers = len(sigma_arr)

    temp_img = []
    laplacian_pyramid = []
    if(len(img1.shape)) == 3:

        for channel in range(3):
            im, l = CombineTwoImages(img1[:, :, channel].astype(np.float64), img2[:, :, channel].astype(np.float64), mask[:, :, channel].astype(np.float64), sigma_arr, layers)
            temp_img.append(np.clip(im / 255, 0, 255))
            laplacian_pyramid.append(l)
    else:
        im, l = CombineTwoImages(img1.astype(np.float64), img2.astype(np.float64), mask.astype(np.float64), sigma_arr, layers)
        temp_img.append(np.clip(im / 255, 0, 255))
        laplacian_pyramid.append(l)

    img = (np.dstack((temp_img))).astype(np.uint8)

    return img, l


    #sigma - – стандартное отклонение нормального распределения
    # check_frequencies(images_gauss_pyramid)
    # show_frequencies_and_images(images_laplac_pyramid)

def test():
    img1 = imread("data/img1.bmp")
    img2 = imread("data/img2.bmp")
    mask = imread("data/mask.bmp")
    sigma_arr = [1.5, 1.6, 1.7]
    result = Run(img1, img2, mask, sigma_arr)
    show(result)

def step_1():
    return imread("data/img1.bmp")

def step_2():
    img = step_1()
    sigma = [0.5, 0.7, 0.9, 1.1, 1.3]
    g = gauss_pyramid(img, sigma, len(sigma))
    check_frequencies(g)
    show_frequencies_and_images(g)

def step_3():
    img = step_1()
    sigma = [0.5, 0.7, 0.9, 1.1, 1.3]
    l, g = laplacian_pyramid(img, sigma, len(sigma))
    check_frequencies(l) #TODO
    show_frequencies_and_images(l)

def step_4():
    img1 = imread("data3/img1.bmp")
    img2 = imread("data3/img2.bmp")
    mask = imread("data3/mask.bmp")

    #set №1
    sigma_arr = [0.5, 1.5, 3]
    i, l = Run(img1, img2, mask, sigma_arr)
    show(i)
    show_images(l)

    # set №2
    sigma_arr = [1.0, 1.0, 1.0]
    i, l = Run(img1, img2, mask, sigma_arr)
    show(i)
    show_images(l)

    # set №3
    sigma_arr = [2.0, 2.0, 2.0, 2.0]
    i, l = Run(img1, img2, mask, sigma_arr)
    show(i)
    show_images(l)

    # set №4
    sigma_arr = [3.0, 3.0, 3.0, 3.0, 3.0]
    i, l = Run(img1, img2, mask, sigma_arr)
    show(i)
    show_images(l)

    # set №5
    sigma_arr = [4.0, 4.0]
    i, l = Run(img1, img2, mask, sigma_arr)
    show(i)
    show_images(l)

def step_5():
    #prepare 3 combinations
    pass
