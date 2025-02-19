#include "mesh_partition.h"
#include <petscksp.h>

void create_petsc_gpu_structures(
    MeshPartition& part,
    Mat* A, Vec* p, Vec* rhs
) {
    MatCreateAIJ(part.cart_comm, 
                part.nx_local*part.ny_local, part.nx_local*part.ny_local,
                PETSC_DETERMINE, PETSC_DETERMINE,
                5, NULL, 5, NULL, A);
    
    MatSetType(*A, MATAIJCUSPARSE);
    
    VecCreateMPICUDA(part.cart_comm, 
                    part.nx_local*part.ny_local, PETSC_DECIDE, p);
    VecDuplicate(*p, rhs);
}

void solve_pressure_poisson(Mat A, Vec p, Vec rhs) {
    KSP ksp;
    KSPCreate(part.cart_comm, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPCG);
    KSPSolve(ksp, rhs, p);
    KSPDestroy(&ksp);
}