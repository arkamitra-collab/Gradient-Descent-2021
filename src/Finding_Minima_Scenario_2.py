# Finding minima of y = cos(3 * pi * x) / x

import numpy as np
import matplotlib.pyplot as plt


def find_x_new(x_old, learning_rate):
    x_new = x_old + (learning_rate * (((3 * np.pi * x_old * np.sin(3 * np.pi * x_old)) + np.cos(3 * np.pi * x_old))
                                      / (x_old ** 2)))
    return x_new


x_o_initials = np.random.rand(5) * 2
# x_o_initials = np.arange(0.01, 2.0, 0.5)
# x_o_initials = np.array([0.5])

x_array = np.array([])
y_array = np.array([])

for x_o in x_o_initials:
    y = np.cos(3 * np.pi * x_o) / x_o

    x_array = np.append(x_array, x_o)
    y_array = np.append(y_array, y)

    x_n = find_x_new(x_o, 0.001)

    while abs(x_n - x_o) > 10 ** (-10):
        x_o = x_n
        x_n = find_x_new(x_o, 0.001)
        y = np.cos(3 * np.pi * x_n) / x_n
        x_array = np.append(x_array, x_n)
        y_array = np.append(y_array, y)

y_min = np.min(y_array)
y_min_index = np.where(y_array == y_min)
print("Value of x where y has global minima: ", x_array[y_min_index])

plt.title("Minima of y = cos(3 * pi * x) / x")
plt.scatter(x_array, y_array)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
