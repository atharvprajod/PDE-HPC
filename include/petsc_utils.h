/**
 * petsc_utils.h - PETSc integration utilities
 * 
 * This header provides utilities for integrating PETSc with CUDA,
 * including creation of GPU-enabled matrices and vectors for
 * efficient linear system solving.
 * 
 * Parallelization strategy:
 * - GPU-accelerated sparse matrices using cuSPARSE
 * - CUDA-enabled vectors for direct GPU memory access
 * - Zero-copy operations between PETSc and CUDA
 */

#include <petsc.h>
#include "mesh_partition.h"
#include <cstddef>  // for NULL in C++
#include <petscmat.h>
#include <petscvec.h>

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
    MatCreateAIJCUSPARSE(part.cart_comm, 
                        part.nx_local * part.ny_local,  // Local rows
                        part.nx_local * part.ny_local,  // Local columns
                        PETSC_DECIDE,                   // Global rows (auto)
                        PETSC_DECIDE,                   // Global columns (auto)
                        5, NULL,                        // Diagonal non-zeros (5-point stencil)
                        5, NULL,                        // Off-diagonal non-zeros
                        A);
    MatSetFromOptions(*A);
    MatSetUp(*A);
    
    // Create CUDA-enabled vectors with direct GPU memory access
    VecCreateCUDAWithArrays(part.cart_comm,  // MPI communicator
                           1,               // Block size
                           part.nx_local*part.ny_local,  // Local size
                           part.nx_local*part.ny_local,  // Global size
                           NULL,            // No initial array
                           p);
    VecDuplicate(*p, rhs);  // Create rhs with same layout as p
} 