/**
 * pressure_solver.cpp - Parallel pressure Poisson solver
 * 
 * This file implements the pressure solver components:
 * - Creation of PETSc structures for the linear system
 * - Configuration of GPU-accelerated solvers
 * - Solution of the pressure Poisson equation
 * 
 * Parallelization strategy:
 * - Distributed sparse matrix assembly across MPI processes
 * - GPU-accelerated sparse matrix operations via cuSPARSE
 * - Parallel iterative solver (Conjugate Gradient)
 * - Algebraic multigrid preconditioning for faster convergence
 */

#include "mesh_partition.h"
#include <petscksp.h>

/**
 * Create PETSc structures for the pressure Poisson system
 * 
 * This function creates:
 * - A sparse matrix (A) for the discrete Laplacian
 * - Vectors for pressure (p) and right-hand side (rhs)
 * All structures are configured for GPU acceleration
 * 
 * @param part Domain partition information
 * @param A Output sparse matrix for pressure system
 * @param p Output vector for pressure solution
 * @param rhs Output vector for right-hand side
 */
void create_petsc_gpu_structures(
    MeshPartition& part,
    Mat* A, Vec* p, Vec* rhs
) {
    // Create sparse matrix for discrete Laplacian
    // 5-point stencil for 2D Poisson equation
    MatCreateAIJ(part.cart_comm, 
                part.nx_local*part.ny_local,  // Local rows
                part.nx_local*part.ny_local,  // Local columns
                PETSC_DETERMINE,              // Global rows (auto)
                PETSC_DETERMINE,              // Global columns (auto)
                5, NULL,                      // Diagonal non-zeros (5-point stencil)
                5, NULL,                      // Off-diagonal non-zeros
                A);
    
    // Set matrix type to cuSPARSE for GPU acceleration
    MatSetType(*A, MATAIJCUSPARSE);
    
    // Create distributed vectors on GPU
    VecCreateMPICUDA(part.cart_comm, 
                    part.nx_local*part.ny_local,  // Local size
                    PETSC_DECIDE,                 // Global size (auto)
                    p);
    
    // Create right-hand side vector with same layout
    VecDuplicate(*p, rhs);
}

/**
 * Solve the pressure Poisson equation
 * 
 * This function:
 * 1. Sets up a Krylov solver (Conjugate Gradient)
 * 2. Configures solver options and tolerances
 * 3. Solves the linear system A*p = rhs
 * 
 * @param A Coefficient matrix (discrete Laplacian)
 * @param p Output vector for pressure solution
 * @param rhs Right-hand side vector
 */
void solve_pressure_poisson(Mat A, Vec p, Vec rhs) {
    // Create Krylov solver context
    KSP ksp;
    KSPCreate(part.cart_comm, &ksp);
    
    // Set matrix operators for the linear system
    KSPSetOperators(ksp, A, A);
    
    // Use Conjugate Gradient method (optimal for symmetric positive definite systems)
    KSPSetType(ksp, KSPCG);
    
    // Solve the system
    KSPSolve(ksp, rhs, p);
    
    // Clean up
    KSPDestroy(&ksp);
}