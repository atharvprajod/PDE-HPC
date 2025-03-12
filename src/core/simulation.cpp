/**
 * simulation.cpp - Core simulation implementation
 * 
 * This file implements the main simulation algorithm, including:
 * - MPI and PETSc initialization
 * - Domain decomposition and memory allocation
 * - The main time-stepping loop with IMEX integration
 * - Boundary exchange and adaptive time stepping
 * 
 * Parallelization strategy:
 * - Domain decomposition using MPI across multiple nodes
 * - GPU acceleration of computational kernels
 * - Implicit-Explicit (IMEX) time integration
 * - Adaptive time stepping with global synchronization
 */

#include <mpi.h>
#include <petscksp.h>
#include "config.h"
#include "cuda_utils.h"
#include "mesh_types.h"

int main(int argc, char** argv) {
    // Initialize MPI for distributed computing across multiple nodes
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get current process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    // Initialize PETSc with GPU support for efficient linear solvers
    // This enables GPU-accelerated sparse matrix operations
    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscOptionsSetValue(NULL, "-vec_type", "cuda");         // Use CUDA vectors
    PetscOptionsSetValue(NULL, "-mat_type", "aijcusparse");  // Use cuSPARSE matrices

    // Load mesh and decompose domain across MPI processes
    // Each process gets a portion of the global domain with ghost cells
    DomainInfo domain;
    decompose_domain(rank, size, &domain);

    // Allocate GPU memory for solution variables and fluxes
    // - U: Conservative variables (density, momentum, energy)
    // - F: Fluxes for each variable
    DeviceArray<double> U(domain.nx_local * domain.ny_local * 4);
    DeviceArray<double> F(domain.nx_local * domain.ny_local * 4);

    // Initialize flow field with initial conditions
    // This CUDA kernel sets up the initial state on the GPU
    initial_condition<<<blocks, threads>>>(U.data, domain);

    // Main time loop - advance solution until final time
    // Uses adaptive time stepping based on CFL condition
    double t = 0.0, dt;
    while (t < SIMULATION_TIME) {
        // 1. Compute fluxes using explicit CUDA kernel
        // This is the most computationally intensive part
        compute_fluxes<<<blocks, threads>>>(U.data, F.data, domain);
        
        // 2. Implicit time integration step (when needed)
        // Solves pressure Poisson equation using PETSc
        if (implicit_step_required(t)) {
            // Create PETSc matrix and vectors for linear system
            Mat A;
            Vec p, rhs;
            
            // Assemble pressure system on GPU
            assemble_pressure_system(&A, &rhs, U.data, domain);
            
            // Solve pressure Poisson equation using PETSc's GPU-enabled solvers
            solve_pressure_poisson(A, rhs, &p);
            
            // Update velocity field based on pressure gradient
            update_velocity<<<blocks, threads>>>(U.data, p, domain);
            
            // Clean up PETSc objects
            MatDestroy(&A);
            VecDestroy(&p);
            VecDestroy(&rhs);
        }

        // 3. Exchange ghost cells between neighboring domains
        // This ensures consistency across domain boundaries
        exchange_ghost_cells(U.data, domain);

        // 4. Compute stable timestep based on CFL condition
        // Local computation followed by global reduction
        dt = compute_cfl(U.data, domain);
        
        // Find minimum dt across all processes for stability
        MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        
        // Advance simulation time
        t += dt;
    }

    // Cleanup resources
    PetscFinalize();
    MPI_Finalize();
    return 0;
}