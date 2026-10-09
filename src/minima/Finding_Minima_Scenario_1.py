# Finding minima of f(x) = x^2

import random
import matplotlib.pyplot as plt


def find_x_new(x_old, learning_rate):
    x_new = x_old - (learning_rate * 2 * x_old)
    return x_new


x_o = float(random.randint(-10, 10))
f_x = x_o ** 2

x_list = []
f_x_list = []
x_list.append(x_o)
f_x_list.append(f_x)

x_n = find_x_new(x_o, 0.1)

while abs(x_n - x_o) > 10 ** (-10):
    x_o = x_n
    x_n = find_x_new(x_o, 0.1)
    f_x = x_n ** 2
    x_list.append(x_n)
    f_x_list.append(f_x)

plt.title("Minima of f(x) = x^2")
plt.plot(x_list, f_x_list)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
