# Perform backpropagation for XOR Gate Truth Table values.

# Let bias (b1) = 0.35, bias (b2) = 0.40, bias (b3) = 0.60, learning rate = 0.5, and initial weights = [0.15, 0.20, 0.25, 0.30, 0.40, 0.45]

# Truth Table of XOR Gate:
# Input 1 | Input 2 | Output
# -----------------------------
#   -1     |   -1     |  -1
#   -1     |    1     |   1
#    1     |   -1     |   1
#    1     |    1     |  -1

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_error(t, out):
    e = (1 / 2) * ((t - out) ** 2)
    return e

x1 = np.array([-1, -1, 1, 1])
x2 = np.array([-1, 1, -1, 1])
target = np.array([-1, 1, 1, -1])
b1 = 0.35
b2 = 0.40
b3 = 0.60
learning_rate = 0.5
w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45

for i in range(0, 4):
    print(f"i={i} | x1={x1[i]} | x2={x2[i]} | target={target[i]}")

    while True:
        print(f"i={i} | w1={w1} | w2={w2} | w3={w3} | w4={w4} | w5={w5} | w6={w6}")

        # Forward Pass
        h1_in = x1[i] * w1 + x2[i] * w2 + b1
        h1_out = sigmoid(h1_in)
        print(f"i={i} | h1_in={h1_in:.4f} | h1_out={h1_out:.4f}")

        h2_in = x1[i] * w3 + x2[i] * w4 + b2
        h2_out = sigmoid(h2_in)
        print(f"i={i} | h2_in={h2_in:.4f} | h2_out={h2_out:.4f}")

        y_in = h1_out * w5 + h2_out * w6 + b3
        y_out = sigmoid(y_in)
        print(f"i={i} | y_in={y_in:.4f} | y_out={y_out:.4f}")

        error = calc_error(target[i], y_out)
        print(f"i={i} | error={error:.4f}")

        if error < 0.001:
            break

        # Backward Pass
        # Find updated w5
        w5_error = (-(target[i] - y_out)) * (y_out * (1 - y_out)) * h1_out
        w5_new = w5 - learning_rate * w5_error
        print(f"i={i} | w5_error={w5_error:.4f} | w5_new={w5_new:.4f}")

        # Find updated w6
        w6_error = (-(target[i] - y_out)) * (y_out * (1 - y_out)) * h2_out
        w6_new = w6 - learning_rate * w6_error
        print(f"i={i} | w6_error={w6_error:.4f} | w6_new={w6_new:.4f}")

        # Find updated w1
        w1_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w5) * (h1_out * (1 - h1_out)) * x1[i]
        w1_new = w1 - learning_rate * w1_error
        print(f"i={i} | w1_error={w1_error:.4f} | w1_new={w1_new:.4f}")

        # Find updated w2
        w2_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w5) * (h1_out * (1 - h1_out)) * x2[i]
        w2_new = w2 - learning_rate * w2_error
        print(f"i={i} | w2_error={w2_error:.4f} | w2_new={w2_new:.4f}")

        # Find updated w3
        w3_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w6) * (h2_out * (1 - h2_out)) * x1[i]
        w3_new = w3 - learning_rate * w3_error
        print(f"i={i} | w3_error={w3_error:.4f} | w3_new={w3_new:.4f}")

        # Find updated w4
        w4_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w6) * (h2_out * (1 - h2_out)) * x2[i]
        w4_new = w4 - learning_rate * w4_error
        print(f"i={i} | w4_error={w4_error:.4f} | w4_new={w4_new:.4f}")

        # Update w1, w2, w3, w4, w5 and w6
        w1 = w1_new
        w2 = w2_new
        w3 = w3_new
        w4 = w4_new
        w5 = w5_new
        w6 = w6_new
