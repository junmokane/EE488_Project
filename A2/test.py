import numpy as np

confidences = np.array([2.9, 4.5, 1.0, 1.0, 1.0], dtype=np.float)
x = np.array([1, 2, 3, 4, 5], dtype=np.int)
y = np.array([5, 4, 3, 2, 1], dtype=np.int)

x = x[::-1]
print(x, type(x))
x = x.flatten()
print(x)

coord = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
sort_ = np.argsort(confidences)
print(sort_)
print(coord[sort_])

x = np.array(range(100))
x = x[::-1]

print(x[24], x)