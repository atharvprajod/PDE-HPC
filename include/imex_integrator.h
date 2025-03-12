/**
 * imex_integrator.h - Implicit-Explicit time integration
 * 
 * This header defines the interface for the IMEX (Implicit-Explicit)
 * time integration scheme, which combines explicit treatment of
 * advection terms with implicit treatment of pressure/diffusion terms.
 * 
 * Parallelization strategy:
 * - Explicit steps use CUDA kernels for high throughput
 * - Implicit steps use PETSc's parallel solvers
 * - Operations are scheduled on CUDA streams for overlap
 */

#pragma once
#include "mesh_partition.h"
#include "cuda_utils.h"
#include <petscksp.h>

/**
 * Perform one step of IMEX time integration
 * 
 * This function:
 * 1. Applies explicit updates for advection terms
 * 2. Solves the implicit pressure system
 * 3. Updates the solution with both contributions
 * 
 * @param U Current solution array on GPU
 * @param F Computed fluxes on GPU
 * @param dt Time step size
 * @param part Mesh partition information
 * @param A PETSc matrix for pressure system
 * @param p PETSc vector for pressure solution
 * @param rhs PETSc vector for right-hand side
 * @param stream CUDA stream for asynchronous execution
 */
void imex_step(
    double* U, double* F, double dt,
    MeshPartition& part, Mat A, Vec p, Vec rhs,
    cudaStream_t stream
); 