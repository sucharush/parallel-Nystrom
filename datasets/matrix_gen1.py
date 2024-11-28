import bz2
import pandas as pd
import numpy as np 

# -----------------------For the first dataset-------------------------
def poly_decay_matrix(n, R, p):
    """
    Generate a Polynomial Decay matrix.
    
    Args:
        n (int): Size of the matrix (n x n).
        R (int): Number of leading 1s in the diagonal.
        p (float): Decay parameter (p > 0).
    
    Returns:
        np.ndarray: Polynomial Decay matrix.
    """
    diag_values = np.ones(n)
    for i in range(R, n):
        diag_values[i] = (i - R + 2)**(-p)
    return np.diag(diag_values)

def exp_decay_matrix(n, R, q):
    """
    Generate an Exponential Decay matrix.
    
    Args:
        n (int): Size of the matrix (n x n).
        R (int): Number of leading 1s in the diagonal.
        q (float): Decay parameter (q > 0).
    
    Returns:
        np.ndarray: Exponential Decay matrix.
    """
    diag_values = np.ones(n)
    for i in range(R, n):
        diag_values[i] = 10**(-(i - R + 1) * q)
    return np.diag(diag_values)

# Parameters
n = 10^3  # Matrix size
R = np.array([5, 10, 20])  # Number of leading 1s

# Polynomial Decay examples
p_slow, p_med, p_fast = 0.5, 1, 2
poly_decay_slow = poly_decay_matrix(n, R[0], p_slow)
poly_decay_med = poly_decay_matrix(n, R[1], p_med)
poly_decay_fast = poly_decay_matrix(n, R[2], p_fast)

# Exponential Decay examples
q_slow, q_med, q_fast = 0.1, 0.25, 1
exp_decay_slow = exp_decay_matrix(n, R[0], q_slow)
exp_decay_med = exp_decay_matrix(n, R[1], q_med)
exp_decay_fast = exp_decay_matrix(n, R[2], q_fast)

# Save the matrices as .npy files
np.save("poly_decay_slow.npy", poly_decay_slow)
np.save("poly_decay_med.npy", poly_decay_med)
np.save("poly_decay_fast.npy", poly_decay_fast)

np.save("exp_decay_slow.npy", exp_decay_slow)
np.save("exp_decay_med.npy", exp_decay_med)
np.save("exp_decay_fast.npy", exp_decay_fast)

# -----------------------For the second dataset-------------------------
# read mnist dataset
with bz2.open("data/mnist.bz2") as f:
    mnist = pd.read_csv(f)   
data = mnist['Z']

# read and process year prediction dataset
def parse_libsvm_line(line):
    elements = line.strip().split()
    label = elements[0]
    indices = []
    values = []
    for elem in elements[1:]:
        index, value = elem.split(":")
        indices.append(int(index) - 1)  # LIBSVM indices are 1-based, convert to 0-based
        values.append(float(value))
    return label, indices, values

def read_libsvm_bz2(filepath):
    data = []
    labels = []
    with bz2.open(filepath, 'rt') as file:
        for line in file:
            label, indices, values = parse_libsvm_line(line)
            labels.append(label)
            feature_array = np.zeros(90)
            feature_array[indices] = values
            data.append(feature_array)
    return pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(90)]), labels
features_df, _ = read_libsvm_bz2("data/YearPredictionMSD.bz2")

# only the first 10000 columns are saved
np.save( "mnist.npy", data)
np.save("year_prediction.npy",features_df.values[:10000]) 

print("Matrices saved as .npy files.")

