/**
 * mesh_types.h - Mesh data structures
 * 
 * This header defines the core data structures for representing
 * the computational mesh, including domain decomposition information.
 * 
 * The DomainInfo structure contains all necessary information about
 * the global and local mesh dimensions, as well as the position of
 * the local domain within the global domain and its neighbors.
 */

#pragma once

/**
 * Domain Information Structure
 * 
 * This structure holds both global and local mesh information for
 * domain decomposition. It tracks the size and position of the local
 * domain within the global domain, as well as neighboring processes.
 */
struct DomainInfo {
    int nx_global, ny_global;   // Global grid dimensions (entire simulation domain)
    int nx_local, ny_local;     // Local grid dimensions for this MPI process
    int offset_x, offset_y;     // Starting indices in the global grid for this local domain
    
    /**
     * Neighboring process ranks in each direction
     * [0]=East, [1]=West, [2]=North, [3]=South
     * Set to MPI_PROC_NULL if there is no neighbor (domain boundary)
     */
    int neighbors[4];
};
