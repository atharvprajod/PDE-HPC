/**
 * main.cpp - Entry point for the PDE-Sim parallel solver
 * 
 * This file implements the main application flow, coordinating:
 * - MPI initialization and domain decomposition
 * - GPU device setup and stream management
 * - Memory allocation for solution variables
 * - The main simulation time-stepping loop
 * - Overlapped computation and communication
 * 
 * Parallelization strategy:
 * - MPI for distributed memory parallelism across nodes
 * - CUDA for GPU acceleration within each node
 * - Asynchronous execution using multiple CUDA streams
 * - PETSc for parallel linear algebra operations
 */

#include "mesh_partition.h"
#include "cuda_utils.h"
#include "simulation_params.h"
#include "imex_integrator.h"
#include "boundary.h"
#include <petscksp.h>
#include "distributed_array.h"

int main(int argc, char** argv) {
    // Initialize MPI for distributed parallelism
    MPI_Init(&argc, &argv);
    
    // Initialize PETSc with command-line options support
    PetscInitialize(&argc, &argv, NULL, NULL);
    
    // Configure PETSc to use GPU acceleration
    // This enables GPU-accelerated sparse matrix and vector operations
    PetscOptionsSetValue(NULL, "-vec_type", "cuda");         // Use CUDA vectors
    PetscOptionsSetValue(NULL, "-mat_type", "aijcusparse");  // Use cuSPARSE matrices
    PetscOptionsSetValue(NULL, "-pc_type", "gamg");          // Use algebraic multigrid preconditioner

    // Load simulation parameters and decompose domain
    // This distributes the global domain across available MPI processes
    SimulationParams params;
    load_parameters("config.yaml", params);
    MeshPartition part = decompose_domain(MPI_COMM_WORLD, params);
    
    // Set up GPU device and create streams with different priorities
    // - compute_stream: High priority for computational kernels
    // - comm_stream: Lower priority for communication operations
    // This enables effective overlap of computation and communication
    setup_cuda_device(MPI_COMM_WORLD);
    cudaStream_t compute_stream, comm_stream;
    create_stream_with_priority(&compute_stream, 30);  // Higher priority
    create_stream_with_priority(&comm_stream, 0);      // Default priority
    
    // Allocate distributed arrays for solution variables
    // These arrays handle both GPU memory and MPI communication
    DistributedArray<double> u_old(part.nx_local, part.ny_local, 4);
    DistributedArray<double> u_new(part.nx_local, part.ny_local, 4);
    DistributedArray<double> fluxes(part.nx_local, part.ny_local, 4);

    // Create PETSc structures with GPU support for pressure solver
    // These leverage PETSc's GPU-enabled solvers for the implicit step
    Mat A;
    Vec pressure, rhs;
    create_petsc_gpu_structures(part, &A, &pressure, &rhs);

    // Main simulation time loop
    // Advances solution until final time using adaptive time stepping
    double t = 0.0;
    while(t < params.t_final) {
        // Configure CUDA kernel launch parameters
        // Block size chosen for optimal occupancy on modern GPUs
        const int BLOCK_SIZE = 16;
        dim3 blocks(
            (part.nx_local + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (part.ny_local + BLOCK_SIZE - 1) / BLOCK_SIZE
        );
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        
        // Launch flux computation kernel on high-priority compute stream
        // This is the most computationally intensive part of each time step
        compute_fluxes<<<blocks, threads, 0, compute_stream>>>(
            u_old.device_ptr(),
            fluxes.device_ptr(),
            part.nx_local,
            part.ny_local,
            params.dx,
            params.dy
        );
        
        // Asynchronously exchange ghost cells on communication stream
        // This overlaps communication with computation for better performance
        exchange_ghost_cells_async(u_new, part, comm_stream);
        
        // Perform IMEX (Implicit-Explicit) time integration
        // - Explicit for advection terms
        // - Implicit for pressure/diffusion terms
        imex_step(u_old.device_ptr(), fluxes.device_ptr(), 
                params.dt, part, A, pressure, rhs, compute_stream);

        // Synchronize compute stream and compute adaptive time step
        // The global minimum dt ensures stability across all processes
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        params.dt = compute_cfl(u_new.device_ptr(), 
                                part.nx_local, part.ny_local,
                                params.dx, params.dy);
        MPI_Allreduce(MPI_IN_PLACE, &params.dt, 1, 
                     MPI_DOUBLE, MPI_MIN, part.cart_comm);
        
        // Swap solution arrays using pointer swap (zero-copy)
        // More efficient than copying data between arrays
        u_old.swap(u_new);
        t += params.dt;
    }

    // Clean up resources
    finalize_simulation(A, pressure, rhs);
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(comm_stream));
    MPI_Finalize();
    return 0;
}