# Plotting y = cos(3 * pi * x) / x

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.1, 2.0, 0.01)
y = np.cos(3 * np.pi * x) / x

plt.plot(x, y)
plt.show()
