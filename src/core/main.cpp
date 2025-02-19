#include "mesh_partition.h"
#include "cuda_utils.h"
#include "simulation_params.h"
#include "imex_integrator.h"
#include "boundary.h"
#include <petscksp.h>
#include "distributed_array.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, NULL, NULL);
    
    // Set PETSc GPU options programmatically
    PetscOptionsSetValue(NULL, "-vec_type", "cuda");
    PetscOptionsSetValue(NULL, "-mat_type", "aijcusparse");
    PetscOptionsSetValue(NULL, "-pc_type", "gamg");

    // Load parameters and domain decomposition
    SimulationParams params;
    load_parameters("config.yaml", params);
    MeshPartition part = decompose_domain(MPI_COMM_WORLD, params);
    
    // GPU setup with stream pool
    setup_cuda_device(MPI_COMM_WORLD);
    cudaStream_t compute_stream, comm_stream;
    create_stream_with_priority(&compute_stream);
    create_stream_with_priority(&comm_stream);
    
    // Allocate memory using improved DistributedArray
    DistributedArray<double> u_old(part.nx_local, part.ny_local, 4);
    DistributedArray<double> u_new(part.nx_local, part.ny_local, 4);
    DistributedArray<double> fluxes(part.nx_local, part.ny_local, 4);

    // PETSc structures with GPU support
    Mat A;
    Vec pressure, rhs;
    create_petsc_gpu_structures(part, &A, &pressure, &rhs);

    // Main simulation loop
    double t = 0.0;
    while(t < params.t_final) {
        // Overlap computation and communication
        const int BLOCK_SIZE = 16;
        dim3 blocks(
            (part.nx_local + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (part.ny_local + BLOCK_SIZE - 1) / BLOCK_SIZE
        );
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        compute_fluxes<<<blocks, threads, 0, compute_stream>>>(
            u_old.device_ptr(),
            fluxes.device_ptr(),
            part.nx_local,
            part.ny_local,
            params.dx,
            params.dy
        );
        
        // Async boundary exchange with dedicated stream
        exchange_ghost_cells_async(u_new, part, comm_stream);
        
        // Unified IMEX integration
        imex_step(u_old.device_ptr(), fluxes.device_ptr(), 
                params.dt, part, A, pressure, rhs, compute_stream);

        // Synchronize and compute time step
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        params.dt = compute_cfl(u_new.device_ptr(), 
                                part.nx_local, part.ny_local,
                                params.dx, params.dy);
        MPI_Allreduce(MPI_IN_PLACE, &params.dt, 1, 
                     MPI_DOUBLE, MPI_MIN, part.cart_comm);
        
        // Swap arrays using pointer swap instead of copy
        u_old.swap(u_new);
        t += params.dt;
    }

    // Cleanup
    finalize_simulation(A, pressure, rhs);
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(comm_stream));
    MPI_Finalize();
    return 0;
}