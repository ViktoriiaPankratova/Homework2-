##Завдання1
import numpy as np
def vector_lengths(vectors):
    squared_norms = np.sum(vectors ** 2, axis=1)
    return np.sqrt(squared_norms)
vectors = np.array([[1, 2], [3, 4], [5, 6]])
lengths = vector_lengths(vectors)
print(lengths)

##Завдання2
def vector_angles(X, Y):

  dot_products = np.sum(X * Y, axis=1)
  norm_X = np.sqrt(np.sum(X**2, axis=1))
  norm_Y = np.sqrt(np.sum(Y**2, axis=1))
  cosines = dot_products / (norm_X * norm_Y)
  cosines = np.clip(cosines, -1, 1)
  angles_rad = np.arccos(cosines)
  angles_deg = np.degrees(angles_rad)
  return angles_deg
X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([[7, 8], [9, 10], [11, 12]])
angles = vector_angles(X, Y)
print(angles)

##Завдання3
def meeting_lines(a1, b1, a2, b2):
  A = np.array([[a1, -1], [a2, -1]])
  b = np.array([-b1, -b2])
  x, y = np.linalg.solve(A, b)
  return x, y
a1, b1 = 2, 1
a2, b2 = -1, 3
x, y = meeting_lines(a1, b1, a2, b2)
print("Точка перетину:", x, y)



from functools import reduce
def matrix_power(a, n):
    if n == 0:
        return np.eye(a.shape[0])
    elif n > 0:
        return reduce(np.matmul, [a] * n)
    else:
        return reduce(np.matmul, [np.linalg.inv(a)] * -n)
A = np.array([[1, 2],
              [3, 4]])
print(matrix_power(A, 4))
