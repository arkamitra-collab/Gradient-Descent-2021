# Perform backpropagation for AND Gate Truth Table values.

# Let bias = 0.35, learning rate = 0.5, and initial weights = [0.15, 0.20]

# Truth Table of AND Gate:
# Input 1 | Input 2 | Output
# -----------------------------
#    0     |    0     |   0
#    0     |    1     |   0
#    1     |    0     |   0
#    1     |    1     |   1

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_error(t, out):
    e = (1 / 2) * ((t - out) ** 2)
    return e

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
target = np.array([0, 0, 0, 1])
learning_rate = 0.5
b = 0.35
w1 = 0.15
w2 = 0.20

for epoch in range(1, 30001):
    total_error = 0

    for i in range(4):
        # Forward Pass
        y_in = x1[i] * w1 + x2[i] * w2 + b
        y_out = sigmoid(y_in)
        error = calc_error(target[i], y_out)
        total_error += error

        if epoch % 1000 == 0:
            print(f"i={i} | epoch={epoch} | total_error={total_error}")

        # Backward Pass
        # Find updated w1
        w1_error = (-(target[i] - y_out)) * (y_out * (1 - y_out)) * x1[i]
        w1_new = w1 - learning_rate * w1_error

        # Find updated w2
        w2_error = (-(target[i] - y_out)) * (y_out * (1 - y_out)) * x2[i]
        w2_new = w2 - learning_rate * w2_error

        # Find updated bias (b)
        b_error = (-(target[i] - y_out)) * (y_out * (1 - y_out))
        b_new = b - learning_rate * b_error

        # Update w1, w2 and b
        w1 = w1_new
        w2 = w2_new
        b = b_new

print(f"Final weights: w1={w1} | w2={w2} | b={b}")

for i in range(4):
    y_in = x1[i] * w1 + x2[i] * w2 + b
    y_out = sigmoid(y_in)

    print(f"Input: {x1[i]} {x2[i]} | Output: {target[i]} ({y_out})")
