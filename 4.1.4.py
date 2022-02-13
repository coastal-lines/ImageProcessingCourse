#На стандартный вход подается параметр σ гауссовского фильтра.
#Подсчитайте и напечатайте на стандартый вывод элементы ядра.
#Для подсчета значений функции Гаусса используйте функцию из предыдущего задания.
#σ может быть нецелым, тогда округлите K K с помощью функции round.
#Элементы ядра выводите с 5 цифрами после запятой.

import math

def gauss(arr):
    o = arr[0]
    x = arr[1]
    y = arr[2]

    x_pow = math.pow(x,2)
    y_pow = math.pow(y,2)
    o_pow = math.pow(o,2)

    p1 = 1 / (2 * math.pi * o_pow)
    p2 = (-x_pow - y_pow) / (2 * o_pow)
    p3 = p1 * (math.e ** p2)

    return p3

def k_width(o):
    k = round((o * 6) + 1)
    return k

#arr = [1,1,1]
#print(gauss(arr))

o = k_width(0.33)
all = gauss([o,-1,1]) + gauss([o,-1,0]) + gauss([o,-1,-1]) + gauss([o,0,1]) + gauss([o,0,0]) + gauss([o,0,-1]) + gauss([o,1,1]) + gauss([o,1,0]) + gauss([o,1,-1])
#print(gauss([o,-1,1]))
#print(gauss([o,-1,0]))
#print(gauss([o,-1,-1]))

#print(gauss([o,0,1]))
print(gauss([o,0,0]) / all)
#print(gauss([o,0,-1]))

#print(gauss([o,1,1]))
#print(gauss([o,1,0]))
#print(gauss([o,1,-1]))

#print(k_width(2))
#print(k_width(5))