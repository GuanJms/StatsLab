import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
print(np.linalg.norm(X, ord='fro') ** 2)