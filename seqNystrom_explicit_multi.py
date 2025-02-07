import numpy as np
import time
import scipy.linalg as sla
from sklearn.metrics.pairwise import rbf_kernel
import torch
from hadamard_transform import hadamard_transform
np.random.seed(4)


class NystromApproximator:
    def __init__(self, A, nrows=1024, witherr = False):
        # self.path = path
        # self.gamma = gamma
        self.nrows = nrows
        self.witherr = witherr
        self.A = A
        self.Btime = None
        self.Ctime = None

    def random_nystron(self, Omega, returnExtra=False):
        """
        Randomized Nystron
        Option to return the singular values of B and rank of A

        """
        temp = time.time()
        A = self.A
        n = self.nrows
        C = A @ Omega
        self.Ctime = time.time() - temp
        temp = time.time()
        B = Omega.T @ C
        self.Btime = time.time() - temp
        temp = time.time()

        try:
            L = np.linalg.cholesky(B)
            Z = np.linalg.lstsq(L, np.transpose(C))[0]
            Z = np.transpose(Z)
        except np.linalg.LinAlgError:
            print("Cholesky failed, using LDL")
            temp = time.time()
            L, d, perm = sla.ldl(B)
            lu = L @ np.sqrt(np.abs(d))
            L = lu[perm, :]
            Cperm = C[:, perm]
            # print(f"cholesky:{time.time()-temp}" )
            temp = time.time()
            Z = np.linalg.lstsq(L, np.transpose(Cperm))[0]
            Z = np.transpose(Z)
            # print(f" CL:{time.time()-temp}" )
        temp = time.time()
        Q, R = np.linalg.qr(Z)
        # print(f"QR:{time.time()-temp}" )
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
        d = np.random.choice([1, -1], n)
        D = np.diag(np.sqrt(n / l) * d)
        P = np.random.choice(range(n), l)
        omega = D
        H = sla.hadamard(n)
        omega = H * d[np.newaxis, :]
        omega = omega[P, :]
        return omega.T

    def truncate_Nys(self, U, Sigma2, Ut, k):
        A_k = U[:, :k] @ Sigma2[:k, :k] @ Ut[:k, :]
        return A_k

    def compute_nuclear_norm(self, matrix):
        return np.linalg.norm(matrix, ord="nuc")

    def relative_nuclear_norm_error(self, A_k):
        residual = self.A - A_k
        nuclear_norm_residual = self.compute_nuclear_norm(residual)
        return nuclear_norm_residual / self.nrows

    def run(self, l, omega_type="gaussian", k=None):
        if k is None:
            k = l  
        n = self.nrows
        results = []

        start_time = time.time()
        omega = self.generate_sketching_matrix(n=n, l=l, omega_type=omega_type)
        omega_time = time.time() - start_time
        print(f"Time to build omega: {omega_time:.2f}")
        # print(omega.shape)
        
        start_nys = time.time()
        U, Sigma2, Ut = self.random_nystron(omega)
        A_k = self.truncate_Nys(U, Sigma2, Ut, k)
        nys_time = time.time() - start_nys
        total_time = nys_time + omega_time
        key_time = omega_time + self.Ctime + self.Btime
        print(f"l={l}, type={omega_type}, Omega+C+B time: {key_time}, Approx. time: {nys_time:.4f}, Total time: {total_time:.4f}")
        if self.witherr:
            print(f'error = {self.relative_nuclear_norm_error(A_k=A_k)}')

        return key_time,  total_time
    
# usage:
# n, l = 1024, 128
# approximator = NystromApproximator(A)
# key_time, total_time = approximator.run(l, omega_type="gaussian")