# Perform backpropagation for OR Gate Truth Table values.

# Let bias = 0.35, learning rate = 0.5, and initial weights = [0.15, 0.20]

# Truth Table of OR Gate:
# Input 1 | Input 2 | Output
# -----------------------------
#   -1     |   -1     |  -1
#   -1     |    1     |   1
#    1     |   -1     |   1
#    1     |    1     |   1

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_error(t, out):
    e = (1 / 2) * ((t - out) ** 2)
    return e

x1 = np.array([-1, -1, 1, 1])
x2 = np.array([-1, 1, -1, 1])
target = np.array([-1, 1, 1, 1])
bias = 0.35
learning_rate = 0.5
w1 = 0.15
w2 = 0.20

for i in range(0, 4):
    print(f"i={i} | x1={x1[i]} | x2={x2[i]} | target={target[i]}")

    while True:
        print(f"i={i} | w1={w1} | w2={w2}")

        # Forward Pass
        y_in = x1[i] * w1 + x2[i] * w2 + bias
        y_out = sigmoid(y_in)
        print(f"i={i} | y_in={y_in:.4f} | y_out={y_out:.4f}")

        error = calc_error(target[i], y_out)
        print(f"i={i} | error={error:.4f}")

        if error < 0.001:
            break

        # Backward Pass
        # Find updated w1
        w1_error = (-(target[i] - y_out)) * (y_out * (1 - y_out)) * x1[i]
        w1_new = w1 - learning_rate * w1_error
        print(f"i={i} | w1_error={w1_error:.4f} | w1_new={w1_new:.4f}")

        # Find updated w2
        w2_error = (-(target[i] - y_out)) * (y_out * (1 - y_out)) * x2[i]
        w2_new = w2 - learning_rate * w2_error
        print(f"i={i} | w2_error={w2_error:.4f} | w2_new={w2_new:.4f}")

        # Update w1 and w2
        w1 = w1_new
        w2 = w2_new
