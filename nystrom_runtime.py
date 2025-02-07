
import numpy as np
from mpi4py import MPI
import scipy.linalg as sla
from math import ceil, sqrt
# from sklearn.metrics.pairwise import rbf_kernel
from importlib import reload
import mpitools
reload(mpitools)
from mpitools import generate_local_sketching, computeR, getQexplicitly
from utils import svd_factorization, truncate_Nys, relative_nuclear_norm_error, load_mnist, load_year
import argparse
import sys

# ----------------- ADDED ARG PARSER ----------------- #
def parse_arguments():
    parser = argparse.ArgumentParser(description='Nystrom Runtime (Single Run, repeated 3 times).')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the input .npy dataset')
    parser.add_argument('--l', type=int, required=True,
                        help='Parameter l for Nystrom method')
    parser.add_argument('--sketching_type', type=str, choices=['G', 'S'], required=True,
                        help='Sketching type: G for Gaussian, S for SRHT')
    return parser.parse_args()
# ----------------------------------------------------- #

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
wt = MPI.Wtime()

# ----------------- OVERRIDE FROM ARGS (Minimal Changes) ----------------- #
args = parse_arguments()
dataset_path = args.dataset_path
l = args.l  # Overriding the hard-coded l=128 below
sketching_type = args.sketching_type.upper()
# ------------------------------------------------------------------------ #

n = 8*1024
npr = int(sqrt(size))
n_blocks = n // npr
n_local = n // size

matrix = None
matrix_transpose = None
x = None
y = None
sol = None
C = None
L = None
Z = None
Cravel = None

# ----------------- ACCUMULATORS FOR 3 RUNS ----------------- #
C_times = []
B_times = []
TSQR_times = []
total_times = []

# rel_nuclear_errors = []
# two_norm_errors = []
# qr_errors = []
# ----------------------------------------------------------- #

for run_idx in range(3):
    if rank % npr == 0:
        # REPLACE the fixed path with dataset_path from args
        matrix = np.load(dataset_path).astype("d")
        mat_start = MPI.Wtime()
        mat_time = MPI.Wtime() - mat_start
        arrs = np.split(matrix, n, axis=1)
        raveled = [arr.ravel() for arr in arrs]
        matrix_transpose = matrix.T.ravel()
        x = np.arange(1, n*l + 1, 1, dtype="d").reshape(n, l)
        sol = np.empty((n, l))

    comm_col = comm.Split(color=rank / npr, key=rank % npr)
    comm_row = comm.Split(color=rank % npr, key=rank / npr)

    rank_col = comm_col.Get_rank()
    rank_row = comm_row.Get_rank()

    # ---------------------------- compute A@Omega ----------------------------
    C_start = MPI.Wtime()
    # scatter columns and put them in the correct order
    submatrix = np.empty((n_blocks, n), dtype="d")
    revceiveMat = np.empty((n_blocks * n), dtype="d")
    comm_col.Scatterv(matrix_transpose, revceiveMat, root=0)
    subArrs = np.split(revceiveMat, n_blocks)
    raveled = [arr.ravel(order="F") for arr in subArrs]
    submatrix = np.ravel(raveled, order="F")

    # scatter matrix rows
    blockMatrix = np.empty((n_blocks, n_blocks), dtype="d")
    comm_row.Scatterv(submatrix, blockMatrix, root=0)

    ########################################
    # distribute sketching matrix (Omega) 
    ########################################
    x_block = np.empty((n_blocks, l), dtype="d")
    x_block = generate_local_sketching(comm=comm, comm_row=comm_row, n=n, l=l, type=sketching_type)

    # solve
    local_result = blockMatrix @ x_block
    # sum
    row_result = np.empty((n_blocks, l), dtype="d")
    comm_col.Allreduce(local_result, row_result, op=MPI.SUM)

    if rank_col == 0:
        comm_row.Gather(row_result, sol, root=0)   

    C_time = MPI.Wtime() - C_start    

    # ------------------------- compute Omega^T A Omega -------------------------
    B_start = MPI.Wtime()
    final_result = np.zeros((l, l), dtype="d")
    sym_local = np.zeros((l, l), dtype="d")

    if rank_col == rank_row:
        sym_local = np.transpose(x_block) @ row_result

    comm.Reduce(sym_local, final_result, op=MPI.SUM, root=0)
    B_time = MPI.Wtime() - B_start

    # ---------------------------- compute B = LL^T -----------------------------
    L = np.empty((l, l), dtype="d")
    if rank == 0:
        C = np.empty((n, l), dtype="d")
        Z = np.empty((n, l), dtype="d")
        L = np.zeros((l, l), dtype="d")
        L, d, perm = sla.ldl(final_result)
        lu = L @ np.sqrt(np.abs(d))
        L = lu[perm, :]
        C = sol[:, perm]
        Cravel = C.ravel()

    # ---------------------------- compute Z = C L^-T ---------------------------
    comm.Barrier()
    C_local = np.empty((n_local*l), dtype="d")
    Z_local = np.empty((n_local*l), dtype="d")
    comm.Bcast(L, root=0)
    comm.Scatterv(Cravel, C_local, root=0)
    C_local = C_local.reshape(n_local, l)
    Z_local = np.linalg.solve(L, C_local.T).T.ravel()
    comm.Barrier()
    comm.Gather(Z_local, Z, root=0)

    # if (rank == 0):
    #     L, d, perm = sla.ldl(x.T @ matrix @ x)
    #     lu = L @ np.sqrt(np.abs(d))
    #     L = lu[perm, :]
    #     C = matrix @ x
    #     C = C[:, perm]

    # ------------------------------ compute Z = QR ------------------------------------
    qr_start = MPI.Wtime()

    A = None
    local_size = int(n / size)

    G = None
    Q = None
    R = None

    if rank == 0:
        A = Z.reshape(n, l)
        Q = np.zeros((n, l), dtype="d")
        R = np.zeros((l, l), dtype="d")

    A_local = np.zeros((local_size, l), dtype="d")
    comm.Scatterv(A, A_local, root=0)
    local_Q, local_R = np.linalg.qr(A_local, mode="reduced")
    local_Qs = [local_Q]

    # compute R
    local_Qs, local_R = computeR(comm, local_R, local_Qs)
    # reconstruct Q
    Q = getQexplicitly(local_Qs, comm)

    if rank == 0:
        R = local_R

    qr_time = MPI.Wtime() - qr_start

    # ------------------------------ compute low-rank approx.  ------------------------------------
    if rank == 0:
        U, Sigma2, Ut = svd_factorization(Q, R)
        # total_time is now the sum of the partial times
        total_time = C_time + B_time + qr_time

        # # Compute errors
        # qr_error = np.linalg.norm(Q @ R - Z)
        # two_norm_error = np.linalg.norm(U @ Sigma2 @ Ut - matrix)
        # rel_nuclear_error = relative_nuclear_norm_error(U @ Sigma2 @ Ut, matrix)

        # Append timings and errors to accumulators
        C_times.append(C_time)
        B_times.append(B_time)
        TSQR_times.append(qr_time)
        total_times.append(total_time)

        # rel_nuclear_errors.append(rel_nuclear_error)
        # two_norm_errors.append(two_norm_error)
        # qr_errors.append(qr_error)

    # Synchronize all processes before next run
    comm.Barrier()

# ----------- After 3 runs, compute average ----------- #
if rank == 0:
    avg_C = sum(C_times) / 3
    avg_B = sum(B_times) / 3
    avg_TSQR = sum(TSQR_times) / 3
    avg_total = sum(total_times) / 3

    # avg_rel_nuclear_error = sum(rel_nuclear_errors) / 3
    # avg_two_norm_error = sum(two_norm_errors) / 3
    # avg_qr_error = sum(qr_errors) / 3

    print("------ AVERAGE OVER 3 RUNS ------")
    print(f"Avg Time for computing C: {avg_C}")
    print(f"Avg Time for computing B: {avg_B}")
    print(f"Avg Time for TSQR: {avg_TSQR}")
    print(f"Avg Time for the whole algorithm: {avg_total}")
    # print(f"Avg 2-norm error: {avg_two_norm_error}")
    # print(f"Avg Relative Nuclear Error: {avg_rel_nuclear_error}")
    # print(f"Avg QR Error (||QR - Z||): {avg_qr_error}")
# ------------------------------------------------------ 


