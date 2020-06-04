import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = a

a[0, 0] = 10

print(b)