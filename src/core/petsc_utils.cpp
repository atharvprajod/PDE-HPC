/**
 * petsc_utils.cpp - PETSc integration utilities
 * 
 * This file provides utilities for integrating PETSc with CUDA:
 * - Creation of GPU-enabled PETSc matrices and vectors
 * - Configuration of PETSc solvers for GPU execution
 * - Direct integration with CUDA memory for zero-copy operations
 * 
 * Parallelization strategy:
 * - Leverages PETSc's GPU-enabled solvers for implicit operations
 * - Uses CUDA-aware MPI for direct GPU-GPU transfers when available
 * - Minimizes host-device transfers by keeping data on GPU
 */

#include "petsc_utils.h"
#include <petsc/private/vecimpl.h>

/**
 * Create PETSc structures optimized for GPU execution
 * 
 * This function creates:
 * - A sparse matrix (A) using cuSPARSE for GPU acceleration
 * - Vectors for pressure (p) and right-hand side (rhs) on GPU
 * 
 * @param part Domain partition information
 * @param A Output sparse matrix for pressure system
 * @param p Output vector for pressure solution
 * @param rhs Output vector for right-hand side
 */
void create_petsc_gpu_structures(const MeshPartition& part, 
                               Mat* A, Vec* p, Vec* rhs) {
    // Create sparse matrix using cuSPARSE backend
    // This enables GPU-accelerated sparse matrix operations
    MatCreateAIJCUSPARSE(part.cart_comm, 
                        part.nx_local*part.ny_local,  // Local rows
                        part.nx_local*part.ny_local,  // Local columns
                        PETSC_DETERMINE,              // Global rows (auto)
                        PETSC_DETERMINE,              // Global columns (auto)
                        5, NULL,                      // Diagonal non-zeros (5-point stencil)
                        5, NULL,                      // Off-diagonal non-zeros
                        A);
    MatSetUp(*A);
    
    // Create CUDA-enabled vectors with direct GPU memory access
    // This allows zero-copy operations with CUDA arrays
    VecCreateCUDAWithArrays(MPI_CUDA_AWARE_COMM_WORLD,  // CUDA-aware MPI communicator
                           1,                          // Block size
                           part.nx_local*part.ny_local, // Local size
                           part.nx_local*part.ny_local, // Global size
                           NULL,                       // No initial array
                           p);
    VecSetFromOptions(*p);
    VecDuplicate(*p, rhs);  // Create rhs with same layout as p
} 