/**
 * mesh_decomposition.cpp - Domain decomposition for parallel execution
 * 
 * This file implements the domain decomposition strategy:
 * - Partitioning the global domain across MPI processes
 * - Creating a 2D Cartesian process grid
 * - Determining process neighbors for communication
 * - Handling uneven divisions of the domain
 * 
 * Parallelization strategy:
 * - 2D Cartesian decomposition for optimal surface-to-volume ratio
 * - Balanced load distribution with remainder handling
 * - Neighbor determination for efficient ghost cell exchange
 * - Support for both periodic and non-periodic boundaries
 */

#include "mesh_partition.h"
#include "simulation_params.h"
#include <mpi.h>

/**
 * Decompose the global domain across MPI processes
 * 
 * This function:
 * 1. Creates a 2D Cartesian process grid
 * 2. Determines local domain size for each process
 * 3. Identifies neighboring processes for communication
 * 4. Handles remainder cells for balanced distribution
 * 
 * @param comm MPI communicator
 * @param params Simulation parameters
 * @return MeshPartition structure with decomposition information
 */
MeshPartition decompose_domain(MPI_Comm comm, const SimulationParams& params) {
    // Get MPI rank and size information
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Initialize mesh partition structure
    MeshPartition part;
    part.nx_global = params.nx;
    part.ny_global = params.ny;

    // Create 2D Cartesian process grid
    // This automatically determines optimal process arrangement
    int dims[2] = {0, 0};  // Let MPI choose optimal dimensions
    MPI_Dims_create(size, 2, dims);
    
    // Create Cartesian communicator (non-periodic boundaries)
    const int periods[2] = {0, 0};
    MPI_Cart_create(comm, 2, dims, periods, 0, &part.cart_comm);

    // Get this process's coordinates in the Cartesian grid
    int coords[2];
    MPI_Cart_coords(part.cart_comm, rank, 2, coords);

    // Calculate base local grid dimensions (without remainder handling)
    part.nx_local = params.nx / dims[1];
    part.ny_local = params.ny / dims[0];
    
    // Handle remainder cells for balanced load distribution
    // Processes with lower ranks get one extra cell when needed
    int rem_x = params.nx % dims[1];
    int rem_y = params.ny % dims[0];
    
    if(coords[1] < rem_x) part.nx_local++;
    if(coords[0] < rem_y) part.ny_local++;

    // Determine neighboring processes in all directions (NESW)
    // -1 indicates no neighbor (boundary of global domain)
    MPI_Cart_shift(part.cart_comm, 0, 1, &part.neighbors[0], &part.neighbors[2]); // North/South
    MPI_Cart_shift(part.cart_comm, 1, 1, &part.neighbors[1], &part.neighbors[3]); // East/West

    return part;
} 