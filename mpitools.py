import numpy as np
import torch # This should print the version number of PyTorch
from hadamard_transform import  hadamard_transform
from math import ceil

def generate_local_sketching(comm, comm_row, n, l, type = 'G'):
    size = comm.Get_size()
    rank = comm.Get_rank()
    npr = int(np.sqrt(size))
    n_blocks = n//npr
    x = None  # Initialize x to ensure scope outside the loop
    D_L = None
    D_R = None
    P = None
    if type == 'G':
        # Generate the matrix on the first row processor of each column
        for i in range(npr):
            if rank % npr == i:
                if rank // npr == 0:
                    np.random.seed(rank)
                    # Generate the matrix directly into x on the first row processor of this column
                    x = np.random.rand(n_blocks, l)
                else:
                    # Prepare an empty matrix for other processors in the same column
                    x = np.empty((n_blocks, l), dtype='d')

                # Broadcast the matrix from the first row processor to all in this column
                comm_row.Bcast(x, root=0)
                
    elif type == 'S':
        if not (n_blocks and ((n_blocks & (n_blocks - 1)) == 0)):
            raise ValueError(f"local size = {n_blocks} is not a power of 2. "
                         f"Ensure that n ({n}) divided by sqrt(size) ({npr}) results in a power of 2.")
        if rank == 0:
            P = np.random.choice(range(n_blocks), l, replace=True)
            # P = np.array([0, 0]).astype(int)
            # print(P)
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
                    D_L = np.diag(d_l).astype('d')
                    D_R = np.diag(d_r).astype('d')
                    # print(d_l, d_r)
                else:
                    D_L = np.empty((l, l), dtype='d')
                    D_R = np.empty((n_blocks, n_blocks), dtype='d')
                # Broadcast
                comm_row.Bcast(D_L, root=0) 
                comm_row.Bcast(D_R, root=0) 
        # form block SRHT
        x = D_R
        x = np.array([hadamard_transform(torch.from_numpy(x[:, i])).numpy() for i in range(n_blocks)]).T # change it back to column-wise!!!
        x = x[P, :]
        x = np.sqrt(n/(l*npr)) * np.transpose(D_L @ x)
    return x

def computeR(comm, local_R, local_Qs):
    size = comm.Get_size()
    rank = comm.Get_rank()
    k = int(np.log2(size))

    for i in range(k):
        partner = rank ^ (1 << i)
        # print(f'iter {i}, rank {rank}, partner {partner}')
        active = rank % (1 << (i)) == 0  # Active only if rank is a multiple of 2^(i+1)
        # print(f'iter {i}, rank {rank} check 2')
        if active:
            # print(f'iter {i}, rank {rank} check 3')
            if rank > partner :
                # print(f'iter {i}, rank {rank} check 4')
                # Send local_R to the partner 
                comm.Send(local_R, dest=partner, tag=rank)
            elif rank < partner and partner<size:
                R_partner = np.zeros_like(local_R)
                # Receive local_R from the partner 
                comm.Recv(R_partner, source=partner, tag=partner)
                combined_R = np.vstack([local_R, R_partner])
                local_Q, local_R = np.linalg.qr(combined_R, mode='reduced')
                # print(f"rank {rank}, loss of ortho for local Q = {loss}")
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
        # print("k=",k, "Rank: ", rank, " color: ", color, " new rank: ", rank_branch) 
        if(color == 0):
            # We scatter the columns of the Q we have 
            Qrows = np.empty((n,n), dtype = 'd')
            comm_branch.Scatterv(Q, Qrows, root = 0) # Local multiplication
            # print("size of Qrows: ", Qrows.shape)
            Qlocal = local_Qs[-1]@Qrows
            # print("size of Qlocal: ", Qlocal.shape)
            local_Qs.pop()
            # Gather
            Q = comm_branch.gather(Qlocal, root = 0) 
            # loss = check_orthogonality(Q)
            # print(f"Loss of orthogonality after {k} th combining: {loss}")
            # print(f"Debug Q: Type={type(Q)}, Content={Q}")
            if rank == 0:
                Q = np.concatenate(Q, axis = 0)
                if k==0:
                    print("Loss of Orthogonality of Q:\n", np.linalg.norm(np.eye(n) - Q.T@Q))  
                    print("Cond(Q):", np.linalg.cond(Q)) 
        comm_branch.Free()
    return Q