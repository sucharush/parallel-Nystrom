import numpy as np
import time
import scipy.linalg as sla
# from sklearn.metrics.pairwise import rbf_kernel
import torch
from hadamard_transform import hadamard_transform
np.random.seed(4)


class NystromApproximator:
    def __init__(self, A,  nrows=1024, witherr = True):
        # self.path = path
        # self.gamma = gamma
        self.nrows = nrows
        self.witherr = witherr
        self.A = A
        self.bctime = None       

    def random_nystron(self, Omega, type, returnExtra=False):
        """
        Randomized Nystron
        Apply srht directly without forming the omega

        """
        temp = time.time()
        if type == 'gaussian':
            A = self.A
            C = A @ Omega
            B = Omega.T @ C
        else:
            d, P =  Omega
            C = self.apply_srht(self.A, d, P).T
            B = self.apply_srht(C, d, P)         
        self.bctime = time.time() - temp  

        try:
            L = np.linalg.cholesky(B)
            Z = np.linalg.lstsq(L, np.transpose(C))[0]
            Z = np.transpose(Z)
        except np.linalg.LinAlgError:
            print("Cholesky failed, using LDL")
            L, d, perm = sla.ldl(B)
            lu = L @ np.sqrt(np.abs(d))
            L = lu[perm, :]
            Cperm = C[:, perm]
            Z = np.linalg.lstsq(L, np.transpose(Cperm))[0]
            Z = np.transpose(Z)

        Q, R = np.linalg.qr(Z)
        U, S, _ = np.linalg.svd(R)
        Sigma = np.diag(S)
        U = Q @ U
        Sigma2 = Sigma @ Sigma

        if returnExtra:
            S_B = np.linalg.cond(B)
            rank_A = np.linalg.matrix_rank(A)
            return U, Sigma2, np.transpose(U), S_B, rank_A
        else:
            return U, Sigma2, np.transpose(U)

    def generate_sketching_matrix(self, n, l, omega_type):
        if omega_type == "gaussian":
            return np.random.randn(n, l)
        elif omega_type == "srht":
            return self.srht(n, l)
        else:
            raise ValueError(f"Unknown omega_type {omega_type}")
        
    def srht(self, n, l):
        assert (n & (n - 1)) == 0, "Number of rows must be a power of 2"
        d = np.random.choice([1, -1], n)
        # DA = d[:, np.newaxis] * self.A
        P = np.random.choice(range(n), l, replace=False)
        return (d, P)

    def apply_srht(self, mat, d, P):
        n,m = mat.shape
        assert (n & (n - 1)) == 0, "Number of rows must be a power of 2"
        DA = d[:, np.newaxis] * mat
        omega = DA
        omega = np.array(
            [
                hadamard_transform(torch.from_numpy(omega[:, i])).numpy()
                for i in range(m)
            ]
        ).T
        omega = omega[P, :]
        return omega
    

    def truncate_Nys(self, U, Sigma2, Ut, k):
        A_k = U[:, :k] @ Sigma2[:k, :k] @ Ut[:k, :]
        return A_k

    def compute_nuclear_norm(self, matrix):
        return np.linalg.norm(matrix, ord="nuc")

    def relative_nuclear_norm_error(self, A_k):
        residual = self.A - A_k
        nuclear_norm_residual = self.compute_nuclear_norm(residual)
        return nuclear_norm_residual / self.nrows # only for kernel matrix!!

    def run(self, l, omega_type="gaussian", k=None):
        if k == None: 
            k = l
        n = self.nrows

        start_time = time.time()
        omega = self.generate_sketching_matrix(n=n, l=l, omega_type=omega_type)
        omega_time = time.time() - start_time
        print(f"Time to build omega: {omega_time:.2f}")
        
        start_nys = time.time()
        U, Sigma2, Ut = self.random_nystron(omega, omega_type)
        A_k = self.truncate_Nys(U, Sigma2, Ut, k)
        total_time = time.time() - start_time
        key_time = omega_time+self.bctime
        print(f"l={l}, BC Time: {key_time}, Total time: {total_time}")
        if self.witherr:
            nuclear_norm_error = self.relative_nuclear_norm_error(A_k)
            print(f"err: {nuclear_norm_error}")

        return key_time, total_time
    
# usage:
# n, l = 1024, 128
# approximator = NystromApproximator(A)
# key_time, total_time = approximator.run(l, omega_type="gaussian")