# Perform backpropagation for XOR Gate Truth Table values.

# Let bias (b1) = 0.35, bias (b2) = 0.40, bias (b3) = 0.60, learning rate = 0.5, and initial weights = [0.15, 0.20, 0.25, 0.30, 0.40, 0.45]

# Truth Table of XOR Gate:
# Input 1 | Input 2 | Output
# -----------------------------
#    0     |    0     |   0
#    0     |    1     |   1
#    1     |    0     |   1
#    1     |    1     |   0

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_error(t, out):
    e = (1 / 2) * ((t - out) ** 2)
    return e

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
target = np.array([0, 1, 1, 0])
learning_rate = 0.5
b1 = 0.35
b2 = 0.40
b3 = 0.60
w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45

for epoch in range(1, 60001):
    total_error = 0

    for i in range(4):
        # Forward Pass
        h1_in = x1[i] * w1 + x2[i] * w2 + b1
        h1_out = sigmoid(h1_in)

        h2_in = x1[i] * w3 + x2[i] * w4 + b2
        h2_out = sigmoid(h2_in)

        y_in = h1_out * w5 + h2_out * w6 + b3
        y_out = sigmoid(y_in)

        error = calc_error(target[i], y_out)
        total_error += error

        if epoch % 1000 == 0:
            print(f"i={i} | epoch={epoch} | total_error={total_error}")

        # Backward Pass
        # Find updated w5
        w5_error = (-(target[i] - y_out)) * (y_out * (1 - y_out)) * h1_out
        w5_new = w5 - learning_rate * w5_error

        # Find updated w6
        w6_error = (-(target[i] - y_out)) * (y_out * (1 - y_out)) * h2_out
        w6_new = w6 - learning_rate * w6_error

        # Find updated b3
        b3_error = (-(target[i] - y_out)) * (y_out * (1 - y_out))
        b3_new = b3 - learning_rate * b3_error

        # Find updated w1
        w1_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w5) * (h1_out * (1 - h1_out)) * x1[i]
        w1_new = w1 - learning_rate * w1_error

        # Find updated w2
        w2_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w5) * (h1_out * (1 - h1_out)) * x2[i]
        w2_new = w2 - learning_rate * w2_error

        # Find updated b1
        b1_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w5) * (h1_out * (1 - h1_out))
        b1_new = b1 - learning_rate * b1_error

        # Find updated w3
        w3_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w6) * (h2_out * (1 - h2_out)) * x1[i]
        w3_new = w3 - learning_rate * w3_error

        # Find updated w4
        w4_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w6) * (h2_out * (1 - h2_out)) * x2[i]
        w4_new = w4 - learning_rate * w4_error

        # Find updated b2
        b2_error = ((-(target[i] - y_out)) * (y_out * (1 - y_out)) * w6) * (h2_out * (1 - h2_out))
        b2_new = b2 - learning_rate * b2_error

        # Update w1, w2, w3, w4, w5 and w6
        w1 = w1_new
        w2 = w2_new
        w3 = w3_new
        w4 = w4_new
        w5 = w5_new
        w6 = w6_new

print(f"Final weights: w1={w1} | w2={w2} | w3={w3} | w4={w4} | w5={w5} | w6={w6} | b1={b1} | b2={b2} | b3={b3}")

for i in range(4):
    h1_in = x1[i] * w1 + x2[i] * w2 + b1
    h1_out = sigmoid(h1_in)

    h2_in = x1[i] * w3 + x2[i] * w4 + b2
    h2_out = sigmoid(h2_in)

    y_in = h1_out * w5 + h2_out * w6 + b3
    y_out = sigmoid(y_in)

    print(f"Input: {x1[i]} {x2[i]} | Output: {target[i]} ({y_out})")
