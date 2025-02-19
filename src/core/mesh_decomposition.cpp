#include "mesh_partition.h"
#include "simulation_params.h"
#include <mpi.h>

MeshPartition decompose_domain(MPI_Comm comm, const SimulationParams& params) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MeshPartition part;
    part.nx_global = params.nx;
    part.ny_global = params.ny;

    // Create 2D Cartesian grid
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    const int periods[2] = {0, 0}; // Non-periodic boundaries
    MPI_Comm cart_comm;
    MPI_Cart_create(comm, 2, dims, periods, 0, &part.cart_comm);

    // Get process coordinates
    int coords[2];
    MPI_Cart_coords(part.cart_comm, rank, 2, coords);

    // Calculate local grid dimensions
    part.nx_local = params.nx / dims[1];
    part.ny_local = params.ny / dims[0];
    
    // Handle remainder for uneven divisions
    int rem_x = params.nx % dims[1];
    int rem_y = params.ny % dims[0];
    
    if(coords[1] < rem_x) part.nx_local++;
    if(coords[0] < rem_y) part.ny_local++;

    // Determine neighbors (NESW)
    MPI_Cart_shift(part.cart_comm, 0, 1, &part.neighbors[0], &part.neighbors[2]); // North/South
    MPI_Cart_shift(part.cart_comm, 1, 1, &part.neighbors[1], &part.neighbors[3]); // East/West

    return part;
} 