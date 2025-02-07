# Parallel Nyström 

Project for HPC course, done by a group of 2.

**Note**: Only key files are included here.

---

## Files Description

### Core Implementation
- **`seqNystrom_explicit_multi.py`**:  
  Sequential implementation where the **Subsampled Randomized Hadamard Transform (SRHT)** is formed explicitly.

- **`seqNystrom_hadamard.py`**:  
  Sequential implementation where the **SRHT** is performed by applying the Hadamard transform on the matrix.

- **`nystrom_runtime.py`**:  
  Parallel implementation of the Nyström method.

### Utility Files
- **`mpitools.py`**:  
  Contains helper functions for distributing the sketching matrix and performing **Tall-Skinny QR (TSQR)** in a distributed setting.

- **`utils.py`**:  
  General utility functions used across the project.

### Scripts
- **`script_parallel.sh`**:  
  Shell script for running the parallel implementation on a cluster or multi-core machine.

