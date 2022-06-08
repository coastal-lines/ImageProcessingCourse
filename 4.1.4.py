import numpy as np
import math

#calculating kernel of Gauss filter by sigma value

def gauss(arr):
    o = arr[0]
    x = arr[1]
    y = arr[2]

    o_pow = math.pow(o, 2)
    x_pow = math.pow(x,2)
    y_pow = math.pow(y,2)

    p1 = 1 / (2 * math.pi * o_pow)
    p2 = (-x_pow - y_pow) / (2 * o_pow)
    p3 = p1 * (math.e ** p2)

    return p3

#calculate radius of the filter
#it equals sigma * 3
#so if sigma == 1 then radius == 3 and kernel == 7 pixels
#if sigma == 2 then radius == 6 and kernel == 13 pixels
#"k" is width of filter or size of kernel
def k_width(o):
    s = (o * 6) + 1
    k = round(s)

    if(k % 2 == 0):
        k = k+1

    #print("k: " + str(k))
    return k

#sum of all pixels in kernel
def calculate_sum(sigma, kernel_width):
    sum = 0
    radius = kernel_width // 2
    for i in range(0 - radius, radius + 1, 1):
        for j in range(0 - radius, radius + 1, 1):
            sum += gauss([sigma, i, j])

    return sum

def get_kernel(sigma, kernel_width, sum):
    kernel = []
    radius = kernel_width // 2

    for i in range(0 - radius, radius + 1, 1):
        for j in range(0 - radius, radius + 1, 1):
            p = gauss([sigma, i, j]) / sum
            kernel.append(p)

    #if(k % 2 == 0):
    #    k = k + 1

    return np.reshape(kernel, [kernel_width, kernel_width])

def calculate_kernel(sigma):
    kernel_width = k_width(sigma)
    sum = calculate_sum(sigma, kernel_width)
    kernel = get_kernel(sigma, kernel_width, sum)

    lines = []
    for row in kernel:
        lines.append(' '.join(str("{:.5f}".format(x)) for x in row))
    print('\n'.join(lines))

calculate_kernel(input())
#calculate_kernel(0.19)
#calculate_kernel(0.23)
#calculate_kernel(0.5)
#calculate_kernel(0.52)
#calculate_kernel(0.553)
#calculate_kernel(0.86)
#calculate_kernel(0.9)
calculate_kernel(2)