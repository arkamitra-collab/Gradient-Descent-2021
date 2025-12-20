# Capture running time of single addition, multiplication and division of first 10^6 positive integers

import time


def add(num1, num2):
    return num1 + num2


def multiply(num1, num2):
    return num1 * num2


def divide(num1, num2):
    return num1 / num2


running_time = 0
sum_result = 0
mult_result = 1
div_result = 1

for x in range(1, 10 ** 6 + 1):
    time_1 = time.time()
    sum_result = add(sum_result, x)
    time_2 = time.time()
    running_time = running_time + (time_2 - time_1)

print("Total sum running time: ", running_time)

running_time = 0
for x in range(1, 10 ** 6 + 1):
    time_1 = time.time()
    mult_result = multiply(sum_result, x)
    time_2 = time.time()
    running_time = running_time + (time_2 - time_1)

print("Total multiplication running time: ", running_time)

running_time = 0
for x in range(1, 10 ** 6 + 1):
    time_1 = time.time()
    div_result = divide(sum_result, x)
    time_2 = time.time()
    running_time = running_time + (time_2 - time_1)

print("Total division running time: ", running_time)
