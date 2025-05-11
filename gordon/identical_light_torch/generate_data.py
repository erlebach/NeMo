import numpy as np
a = 1.0
n = 1000
m = 200
np.random.seed(42)
x_train = np.random.uniform(-5, 5, size=(n, 1))
y_train = a * np.sin(x_train)
x_val = np.random.uniform(-5, 5, size=(m, 1))
y_val = a * np.sin(x_val)
np.savez('sine_data.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
