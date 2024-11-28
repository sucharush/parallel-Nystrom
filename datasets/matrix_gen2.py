import numpy as np

# to generate A from datasets
def rbf(x1, x2, c):
    return np.exp(-c * np.linalg.norm(x1 - x2) ** 2)

def generate_A(data, c, n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            A[i, j] = A[j, i] = rbf(data[i], data[j], c)            
    return A