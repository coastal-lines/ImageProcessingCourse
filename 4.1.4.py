import numpy as np
import math

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

def k_width(o):
    s = (o * 6) + 1
    k = round(s)

    if(k % 2 == 0):
        k = k+1

    #print("k: " + str(k))
    return k


def calculate_sum(o, k):
    sum = 0
    radius = k // 2
    for i in range(0 - radius, radius + 1, 1):
        for j in range(0 - radius, radius + 1, 1):
            sum += gauss([o, i, j])

    return sum

def get_kernel(o, k, sum):
    kernel = []
    radius = k // 2

    for i in range(0 - radius, radius + 1, 1):
        for j in range(0 - radius, radius + 1, 1):
            p = gauss([o, i, j]) / sum
            kernel.append(p)

    #if(k % 2 == 0):
    #    k = k + 1

    return np.reshape(kernel, [k, k])

def calculate_kernel(o):
    k = k_width(o)
    sum = calculate_sum(o, k)
    kernel = get_kernel(o, k, sum)

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
