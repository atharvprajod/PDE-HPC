#include "petsc_utils.h"
#include <petsc/private/vecimpl.h>

void create_petsc_gpu_structures(const MeshPartition& part, 
                               Mat* A, Vec* p, Vec* rhs) {
    MatCreateAIJCUSPARSE(part.cart_comm, 
                        part.nx_local*part.ny_local,
                        part.nx_local*part.ny_local,
                        PETSC_DETERMINE, PETSC_DETERMINE,
                        5, NULL, 5, NULL, A);
    MatSetUp(*A);
    
    VecCreateCUDAWithArrays(MPI_CUDA_AWARE_COMM_WORLD, 1, 
                           part.nx_local*part.ny_local, 
                           part.nx_local*part.ny_local,
                           NULL, p);
    VecSetFromOptions(*p);
    VecDuplicate(*p, rhs);
} 