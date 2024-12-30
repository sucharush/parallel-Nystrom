import numpy as np
from mpi4py import MPI
import scipy.linalg as sla
from math import ceil
from sklearn.metrics.pairwise import rbf_kernel
from importlib import reload
import mpitools
reload(mpitools)
from mpitools import generate_local_sketching, computeR, getQexplicitly
from utils import svd_factorization, truncate_Nys, relative_nuclear_norm_error
np.random.seed(4)


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n = 32*size
l = 32
sketching_type = 'G'
npr = int(np.sqrt(size))
# npr = 2
n_blocks = n//npr
n_local = n//size

matrix = None
matrix_transpose = None
x = None
y = None
sol = None
C = None
L = None
Z = None
Cravel = None

if rank % npr == 0:
    matrix = np.random.rand(n**2).reshape(n, n).astype('d')
    matrix = rbf_kernel(matrix, gamma=1e-2)
    # print(matrix)
    arrs = np.split(matrix, n, axis=1)
    raveled = [arr.ravel() for arr in arrs]
    matrix_transpose = matrix.T.ravel()
    # print(matrix_transpose)
    x = np.arange(1, n*l + 1, 1, dtype="d").reshape(n, l)
    # x = np.random.rand(n, l)
    # y = np.dot(matrix, x)
    sol = np.empty((n, l))

comm_col = comm.Split(color=rank / npr, key=rank % npr)
comm_row = comm.Split(color=rank % npr, key=rank / npr)

rank_col = comm_col.Get_rank()
rank_row = comm_row.Get_rank()

# ---------------------------- compute A@Omega ----------------------------
# scatter columns and put them in the correct order
submatrix = np.empty((n_blocks, n), dtype="d")
revceiveMat = np.empty((n_blocks * n), dtype="d")
comm_col.Scatterv(matrix_transpose, revceiveMat, root=0)
subArrs = np.split(revceiveMat, n_blocks)
raveled = [arr.ravel(order="F") for arr in subArrs]
submatrix = np.ravel(raveled, order="F")
# print(f"rank_col:{rank_col},rank_row:{rank_row}, {submatrix}")

# scatter matrix rows
blockMatrix = np.empty((n_blocks, n_blocks), dtype="d")
comm_row.Scatterv(submatrix, blockMatrix, root=0)

########################################
# distribute sketching matrix (Omega) 
########################################
x_block = np.empty((n_blocks, l), dtype="d")
# comm_col.Scatterv(x, x_block, root=0)
x_block = generate_local_sketching(comm=comm, comm_row=comm_row, n=n, l=l, type = sketching_type)
# print(f"rank_col:{rank_col},rank_row:{rank_row}, omega block: {x_block}")

# solve
local_result = blockMatrix @ x_block
# sum
row_result = np.empty((n_blocks, l), dtype="d")
# comm_col.Reduce(local_result, row_result, op=MPI.SUM, root=0)
comm_col.Allreduce(local_result, row_result, op=MPI.SUM)
# print(f"rank_col:{rank_col},rank_row:{rank_row}, row results: {row_result}")
# # gather
if rank_col==0:
    comm_row.Gather(row_result,sol,root=0)   
    
    
# ------------------------- compute Omega^T A Omega -------------------------
final_result = np.zeros((l, l), dtype="d")
sym_local = np.zeros((l, l), dtype="d")
# print(f"rank_col:{rank_col},rank_row:{rank_row}, sym results: {sym_local}")
# match the corresponding parts
if rank_col==rank_row:
    sym_local = np.transpose(x_block) @ row_result
    # print(sym_local)    
comm.Reduce(sym_local, final_result, op = MPI.SUM, root = 0)

# ---------------------------- compute B = LL^T -----------------------------
L = np.empty((l, l), dtype="d")
if rank == 0 :
    # print(f"check point: rank_col:{rank_col},rank_row:{rank_row}")
    C = np.empty((n, l), dtype="d")
    Z = np.empty((n, l), dtype="d")
    L = np.zeros((l, l), dtype="d")
    L, d, perm = sla.ldl(final_result)
    lu = L @ np.sqrt(np.abs(d))
    L = lu[perm, :]
    C = sol[:, perm]
    Cravel = C.ravel()  
    # print(f"L: {L}")

# ---------------------------- compute Z = C L^-T ---------------------------
comm.Barrier()
C_local = np.empty((n_local*l), dtype="d")
Z_local = np.empty((n_local*l), dtype="d")
comm.Bcast(L, root = 0)
comm.Scatterv(Cravel, C_local, root = 0)
C_local = C_local.reshape(n_local,l)
# print(f"rank={rank}, C local: {C_local},")
# print(f"invLt: {np.linalg.inv(L.T)}, \n Z_local: {C_local@np.linalg.inv(L.T)}")
Z_local = np.linalg.solve(L, C_local.T).T.ravel()
# print(f"rank={rank}, Z local: {Z_local},")
comm.Barrier()
comm.Gather(Z_local, Z, root = 0)


if (rank == 0):
    L, d, perm = sla.ldl(x.T@matrix@x)
    lu = L @ np.sqrt(np.abs(d))
    L = lu[perm, :]
    C = matrix@x
    C = C[:, perm]
    # print("OmegaTAomega Solution comparison ", np.linalg.norm(final_result- x.T@matrix@x))
    # print("Aomega Solution comparison: ", np.linalg.norm(sol-matrix@x))
    # print("Z Solution comparison: ", np.linalg.norm(Z.reshape(n, l)-C@np.linalg.inv(L.T)))
    
# ------------------------------ compute Z = QR ------------------------------------


A = None

local_size = int(n/size) # Dividing by rows

# A = None
G = None
Q = None
R = None

if rank == 0:
    A = Z.reshape(n, l)
    Q = np.zeros((n, l), dtype="d")
    R = np.zeros((l, l), dtype="d")
    
A_local = np.zeros((local_size, l), dtype="d")
comm.Scatterv(A, A_local, root=0)
local_Q, local_R = np.linalg.qr(A_local, mode='reduced')
local_Qs = [local_Q]

# compute R
local_Qs, local_R = computeR(comm, local_R, local_Qs)            
# reconstruct Q
Q = getQexplicitly(local_Qs, comm)

if rank == 0:
    R = local_R
    # Q = np.concatenate(Q, axis = 0)
    print("Loss of Orthogonality of Q:\n", np.linalg.norm(np.eye(l) - Q.T@Q))  
    print("Cond(Q):", np.linalg.cond(Q)) 

# ------------------------------ compute low-rank approx.  ------------------------------------


if rank == 0:
    U, Sigma2, Ut  = svd_factorization(Q, R)
    print(f"R:{R.shape}, Q:{Q.shape}, error:{np.linalg.norm(Q@R - Z)}")
    print(f"cond(A): {np.linalg.norm(matrix)}")
    print(f"2-norm error: {np.linalg.norm(U@Sigma2@Ut -matrix)}")
    print(f"rel. nuclear error: {relative_nuclear_norm_error(U@Sigma2@Ut, matrix)}")
    k_vec = 10 * np.arange(1, l//10 + 1, 1).astype(int)
    for k in k_vec:
        truncated_approx = truncate_Nys(U, Sigma2, Ut, k)
        error = relative_nuclear_norm_error(truncated_approx, matrix)
        print(f"k = {k:3d}, relative nuclear error: {error}")
    