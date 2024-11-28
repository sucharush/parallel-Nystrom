import numpy as np

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

print("Matrices saved as .npy files.")

