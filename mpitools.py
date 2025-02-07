import numpy as np
import torch # This should print the version number of PyTorch
from hadamard_transform import  hadamard_transform
from math import ceil
import scipy.linalg as sla

def generate_local_sketching(comm, comm_row, n, l, type = 'G'):
    size = comm.Get_size()
    rank = comm.Get_rank()
    npr = int(np.sqrt(size))
    n_blocks = n//npr
    x = None 
    D_L = None
    D_R = None
    P = None
    if type == 'G':
        # Generate the matrix on the first row processor of each column
        for i in range(npr):
            if rank % npr == i:
                if rank // npr == 0:
                    np.random.seed(rank)
                    x = np.random.randn(n_blocks, l)
                else:
                    x = np.empty((n_blocks, l), dtype='d')

                comm_row.Bcast(x, root=0)
                
    elif type == 'S':
        if not (n_blocks and ((n_blocks & (n_blocks - 1)) == 0)):
            raise ValueError(f"local size = {n_blocks} is not a power of 2. "
                         f"Ensure that n ({n}) divided by sqrt(size) ({npr}) results in a power of 2.")
        if rank == 0:
            P = np.random.choice(range(n_blocks), l, replace=True)

        else:
            P = np.empty((l, ), dtype=int)
        comm.Bcast(P, root=0)   
        for i in range(npr):
            if rank % npr == i:
                if rank // npr == 0:
                    np.random.seed(rank)
                    # Generate the conponents for SRHT
                    d_l = np.random.choice([-1, 1], l)
                    d_r = np.random.choice([-1, 1], n_blocks)
                else:
                    d_l = np.empty((l,), dtype=int)
                    d_r = np.empty((n_blocks,), dtype=int)
                    
                # Broadcast
                comm_row.Bcast(d_l, root=0) 
                comm_row.Bcast(d_r, root=0) 
                
        # form block SRHT (explicit computation here since pytorch hadamard_transform is too slow)
        H = sla.hadamard(n_blocks)
        x = H * d_r[np.newaxis, :] # vector broadcast, doesn't improve performance
        x = x[P, :] 
        x = d_l[:, np.newaxis] * x 
        x = np.sqrt(n/(l*npr)) * x.T
    return x

def computeR(comm, local_R, local_Qs):
    size = comm.Get_size()
    rank = comm.Get_rank()
    k = int(np.log2(size))

    for i in range(k):
        partner = rank ^ (1 << i)
        active = rank % (1 << (i)) == 0  # Active only if rank is a multiple of 2^(i+1)

        if active:

            if rank > partner :
                # Send local_R to the partner 
                comm.Send(local_R, dest=partner, tag=rank)
                
            elif rank < partner and partner<size:
                R_partner = np.zeros_like(local_R)
                # Receive local_R from the partner 
                comm.Recv(R_partner, source=partner, tag=partner)
                combined_R = np.vstack([local_R, R_partner])
                local_Q, local_R = np.linalg.qr(combined_R, mode='reduced')
                local_Qs.append(local_Q)
    return local_Qs, local_R

def getQexplicitly(local_Qs, comm): 
    rank = comm.Get_rank()
    size = comm.Get_size()
    m = local_Qs[0].shape[0]*size
    n = local_Qs[0].shape[1]
    Q = None
    if rank == 0:
        Q = np.eye(n, n)
        Q = local_Qs[-1]@Q
        local_Qs.pop()
    # Start iterating through the tree backwards
    for k in range(ceil(np.log2(size))-1, -1, -1):
        color = rank%(2**k)
        key = rank//(2**k)
        comm_branch = comm.Split(color = color, key = key)
        rank_branch = comm_branch.Get_rank()
        if(color == 0):
            # We scatter the columns of the Q we have 
            Qrows = np.empty((n,n), dtype = 'd')
            comm_branch.Scatterv(Q, Qrows, root = 0) 
            Qlocal = local_Qs[-1]@Qrows
            local_Qs.pop()
            # Gather
            Q = comm_branch.gather(Qlocal, root = 0) 
            if rank == 0:
                Q = np.concatenate(Q, axis = 0)
                # if k==0:
                #     print("Loss of Orthogonality of Q:\n", np.linalg.norm(np.eye(n) - Q.T@Q))  
                #     print("Cond(Q):", np.linalg.cond(Q)) 
        comm_branch.Free()
    return Q