/**
 * mesh_partition.h - Domain decomposition for parallel execution
 * 
 * This header defines the MeshPartition structure and related functions
 * for domain decomposition across MPI processes. It handles the creation
 * of Cartesian communicators and neighbor identification.
 * 
 * Parallelization strategy:
 * - 2D Cartesian process grid for optimal communication patterns
 * - Neighbor identification for ghost cell exchange
 * - Support for both periodic and non-periodic boundaries
 */

#include <mpi.h>

/**
 * Mesh Partition Structure
 * 
 * This structure holds information about how the global domain is
 * partitioned for a specific MPI process, including:
 * - Local domain dimensions
 * - Cartesian communicator for structured communication
 * - Neighboring process ranks
 * - Ghost layer configuration
 */
struct MeshPartition {
    MPI_Comm cart_comm;        // Cartesian communicator for structured grid
    int nx_local, ny_local;    // Local grid dimensions for this process
    int neighbors[4];          // Neighboring processes (NESW ordering)
    int ghost_layers;          // Number of ghost cell layers
    
    /**
     * Default constructor
     */
    MeshPartition() = default;
    
    /**
     * Constructor with explicit initialization
     * 
     * @param comm MPI communicator
     * @param nx Local x-dimension
     * @param ny Local y-dimension
     * @param neigh Array of neighbor ranks
     */
    MeshPartition(MPI_Comm comm, int nx, int ny, int* neigh) 
        : cart_comm(comm), nx_local(nx), ny_local(ny), ghost_layers(1) {
        for(int i=0; i<4; ++i) neighbors[i] = neigh[i];
    }
};

/**
 * Decompose the global domain across MPI processes
 * 
 * This function creates a 2D Cartesian process grid and determines
 * the local domain size and neighbors for each process.
 * 
 * @param comm MPI communicator
 * @param params Simulation parameters containing global domain size
 * @return MeshPartition structure with decomposition information
 */
MeshPartition decompose_domain(MPI_Comm comm, const class SimulationParams& params);
