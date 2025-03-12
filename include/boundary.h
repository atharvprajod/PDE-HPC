/**
 * boundary.h - Ghost cell exchange for domain decomposition
 * 
 * This header defines functions for exchanging ghost cell data between
 * neighboring domains in the distributed simulation. It supports both
 * synchronous and asynchronous communication patterns.
 * 
 * Parallelization strategy:
 * - MPI communication between domain boundaries
 * - CUDA-aware MPI for direct GPU-GPU transfers
 * - Asynchronous exchanges overlapped with computation
 * - Dedicated CUDA streams for communication operations
 */

#pragma once
#include "mesh_partition.h"
#include "cuda_utils.h"
#include <mpi.h>

/**
 * Exchange ghost cells synchronously between neighboring domains
 * 
 * This function performs a complete ghost cell exchange, waiting for
 * all communications to complete before returning.
 * 
 * @param U Distributed array containing solution data
 * @param part Mesh partition information
 */
void exchange_ghost_cells(DistributedArray<double>& U, MeshPartition& part);

/**
 * Exchange ghost cells asynchronously using a dedicated CUDA stream
 * 
 * This function initiates ghost cell exchanges without waiting for completion,
 * allowing computation to overlap with communication for better performance.
 * 
 * @param U Distributed array containing solution data
 * @param part Mesh partition information
 * @param stream CUDA stream for asynchronous operations
 */
void exchange_ghost_cells_async(DistributedArray<double>& U, MeshPartition& part, cudaStream_t stream);

/**
 * Get send buffer for a specific direction
 * 
 * Helper function to access the appropriate send buffer for a boundary direction.
 * 
 * @param data Base pointer to solution data
 * @param dir Direction index (0=North, 1=East, 2=South, 3=West)
 * @param part Mesh partition information
 * @return Pointer to the send buffer for the specified direction
 */
double* get_send_buffer(double* data, int dir, MeshPartition& part);

/**
 * Get receive buffer for a specific direction
 * 
 * Helper function to access the appropriate receive buffer for a boundary direction.
 * 
 * @param data Base pointer to solution data
 * @param dir Direction index (0=North, 1=East, 2=South, 3=West)
 * @param part Mesh partition information
 * @return Pointer to the receive buffer for the specified direction
 */
double* get_recv_buffer(double* data, int dir, MeshPartition& part); 