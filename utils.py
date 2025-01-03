import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
def svd_factorization(Q, R):
    """
    Compute the truncated SVD of the matrix R and then compute U_k.
    
    Parameters:
    - Q (numpy.ndarray): The Q matrix from QR factorization of Z.
    - R (numpy.ndarray): The R matrix from QR factorization of Z.
    - k (int): Rank for truncation.

    Returns:
    - U_k (numpy.ndarray): The truncated left singular vectors.
    - Sigma_k (numpy.ndarray): The truncated singular values.
    - Vt_k (numpy.ndarray): The truncated right singular vectors transposed.
    - A_k (numpy.ndarray): Rank-k approximation of the matrix A.
    """
    # Step 5: Compute truncated rank-k SVD of R
    U, S, _ = np.linalg.svd(R)
    Sigma = np.diag(S)
    U = Q @ U
    Sigma2 = Sigma @ Sigma
    return U, Sigma2, np.transpose(U)

def truncate_Nys(U, Sigma2, Ut, k):
    A_k = U[:, :k]@Sigma2[:k, :k]@Ut[:k, :]
    return A_k

def relative_nuclear_norm_error(A, A_hat):
    """
    Computes the relative error in nuclear norm between the original matrix A
    and its approximation A_hat.

    Parameters:
    - A (numpy.ndarray): Original matrix.
    - A_hat (numpy.ndarray): Approximated matrix.

    Returns:
    - float: The relative nuclear norm error.
    """
    # Compute the nuclear norm of A
    # U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Compute the residual matrix (A - A_hat)
    residual = A - A_hat
    nuclear_norm_residual = np.linalg.norm(residual, ord="nuc")

    # Compute the relative nuclear norm error
    relative_error = nuclear_norm_residual / A.shape[0]

    return relative_error

def load_mnist(nrows, gamma=1e-4):
    try:
        matrix = np.load('datasets/mnist.npy')[: nrows]
        A = rbf_kernel(matrix, gamma=gamma)
        return A
    except Exception as e:
        print(f"An error occurred while loading or building the kernel: {e}")
        return None
    
def load_year(nrows, gamma=1e-8):
    try:
        matrix = np.load('datasets/year_prediction.npy')[: nrows]
        A = rbf_kernel(matrix, gamma=gamma)
        return A
    except Exception as e:
        print(f"An error occurred while loading or building the kernel: {e}")
        return None