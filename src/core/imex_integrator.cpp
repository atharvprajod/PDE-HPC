#include "imex_integrator.h"
#include <petscksp.h>

void imex_step(double* u, double* fluxes, double dt,
              const MeshPartition& part, Mat A, Vec p, Vec rhs,
              cudaStream_t stream) {
    // Explicit flux computation
    cudaEvent_t flux_done;
    cudaEventCreateWithFlags(&flux_done, cudaEventDisableTiming);
    
    // Implicit pressure solve
    KSPSolver ksp;
    KSPCreate(part.cart_comm, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, rhs, p);
    
    // Overlapped device-host transfers
    VecCUDAPlaceArray(p, u);  // Use GPU array directly
    KSPDestroy(&ksp);
    cudaEventDestroy(flux_done);
} 