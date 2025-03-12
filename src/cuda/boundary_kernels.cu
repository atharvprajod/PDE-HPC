/**
 * boundary_kernels.cu
 * 
 * CUDA kernels for handling domain boundaries and MPI communication.
 * This file implements the kernels for packing/unpacking data at domain
 * boundaries for ghost cell exchange between MPI processes.
 */

#include <cuda_runtime.h>
#include "config.h"
#include "mesh_types.h"
#include "cuda_utils.h"

/**
 * Pack west boundary data for sending to neighboring process
 * 
 * Parallelization strategy:
 * - Each thread packs data for one cell along the boundary
 * - 1D thread blocks process cells along the y-direction
 * - Coalesced memory access pattern for optimal performance
 * - Minimal thread divergence as all threads perform similar work
 * 
 * @param U Source data array (solution variables)
 * @param buffer Destination buffer for packed data
 * @param domain Domain information
 */
__global__ void pack_west_halo(
    const double* __restrict__ U,
    double* __restrict__ buffer,
    const DomainInfo domain
) {
    // Calculate global thread index (along y-direction)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the domain boundary
    if (j >= GHOST_LAYERS && j < domain.ny_local - GHOST_LAYERS) {
        // Calculate index of the interior cell adjacent to west boundary
        int idx = j * domain.nx_local + GHOST_LAYERS;
        
        // Pack all components for this cell into contiguous buffer
        // This improves MPI communication efficiency
        #pragma unroll
        for (int var = 0; var < NUM_VARS; var++) {
            // Store in buffer with components grouped together
            // This layout optimizes for MPI message size
            buffer[(j - GHOST_LAYERS) + var * (domain.ny_local - 2*GHOST_LAYERS)] = 
                U[idx + var * domain.nx_local * domain.ny_local];
        }
    }
}

/**
 * Unpack data received from west neighbor into east ghost cells
 * 
 * Parallelization strategy:
 * - Each thread unpacks data for one ghost cell
 * - 1D thread blocks process cells along the y-direction
 * - Memory access pattern optimized for coalescing
 * 
 * @param U Destination data array (solution variables)
 * @param buffer Source buffer containing received data
 * @param domain Domain information
 */
__global__ void unpack_east_halo(
    double* __restrict__ U,
    const double* __restrict__ buffer,
    const DomainInfo domain
) {
    // Calculate global thread index (along y-direction)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the domain boundary
    if (j >= GHOST_LAYERS && j < domain.ny_local - GHOST_LAYERS) {
        // Calculate index of the ghost cell at east boundary
        int idx = j * domain.nx_local + (domain.nx_local - GHOST_LAYERS);
        
        // Unpack all components for this ghost cell
        #pragma unroll
        for (int var = 0; var < NUM_VARS; var++) {
            // Load from buffer with components grouped together
            U[idx + var * domain.nx_local * domain.ny_local] = 
                buffer[(j - GHOST_LAYERS) + var * (domain.ny_local - 2*GHOST_LAYERS)];
        }
    }
}

/**
 * Pack east boundary data for sending to neighboring process
 * 
 * Parallelization strategy:
 * - 2D thread blocks: y-direction × components
 * - Maximizes parallelism by handling multiple components simultaneously
 * - Optimized for modern GPU architectures with high thread counts
 * 
 * @param U Source data array (solution variables)
 * @param buffer Destination buffer for packed data
 * @param nx Number of cells in x-direction (including ghosts)
 * @param ny Number of cells in y-direction (including ghosts)
 * @param comps Number of components/variables
 */
__global__ void pack_east_halo(
    const double* __restrict__ U,
    double* __restrict__ buffer,
    int nx, int ny, int comps
) {
    // Calculate global thread indices
    int y = blockIdx.x * blockDim.x + threadIdx.x;  // Position along y-direction
    int c = blockIdx.y;                            // Component index
    
    // Check if this thread is within bounds
    if (y >= GHOST_LAYERS && y < ny - GHOST_LAYERS && c < comps) {
        // Calculate index of interior cell adjacent to east boundary
        int idx = (y * nx + (nx - GHOST_LAYERS - 1)) * comps + c;
        
        // Pack data into buffer with components grouped together
        buffer[(y - GHOST_LAYERS) + c * (ny - 2*GHOST_LAYERS)] = U[idx];
    }
}

/**
 * Unpack data received from east neighbor into west ghost cells
 * 
 * Parallelization strategy:
 * - 2D thread blocks: y-direction × components
 * - Balanced workload across threads
 * - Optimized memory access pattern
 * 
 * @param U Destination data array (solution variables)
 * @param buffer Source buffer containing received data
 * @param nx Number of cells in x-direction (including ghosts)
 * @param ny Number of cells in y-direction (including ghosts)
 * @param comps Number of components/variables
 */
__global__ void unpack_west_halo(
    double* __restrict__ U,
    const double* __restrict__ buffer,
    int nx, int ny, int comps
) {
    // Calculate global thread indices
    int y = blockIdx.x * blockDim.x + threadIdx.x;  // Position along y-direction
    int c = blockIdx.y;                            // Component index
    
    // Check if this thread is within bounds
    if (y >= GHOST_LAYERS && y < ny - GHOST_LAYERS && c < comps) {
        // Calculate index of ghost cell at west boundary
        int idx = (y * nx + (GHOST_LAYERS - 1)) * comps + c;
        
        // Unpack data from buffer
        U[idx] = buffer[(y - GHOST_LAYERS) + c * (ny - 2*GHOST_LAYERS)];
    }
}

/**
 * Pack north boundary data for sending to neighboring process
 * 
 * Parallelization strategy:
 * - Each thread packs data for one cell along the boundary
 * - 1D thread blocks process cells along the x-direction
 * - Strided memory access pattern handled efficiently
 * 
 * @param U Source data array (solution variables)
 * @param buffer Destination buffer for packed data
 * @param domain Domain information
 */
__global__ void pack_north_halo(
    const double* __restrict__ U,
    double* __restrict__ buffer,
    const DomainInfo domain
) {
    // Calculate global thread index (along x-direction)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the domain boundary
    if (i >= GHOST_LAYERS && i < domain.nx_local - GHOST_LAYERS) {
        // Calculate index of interior cell adjacent to north boundary
        int idx = (domain.ny_local - GHOST_LAYERS - 1) * domain.nx_local + i;
        
        // Pack all components for this cell
        #pragma unroll
        for (int var = 0; var < NUM_VARS; var++) {
            // Store in buffer with components grouped together
            buffer[(i - GHOST_LAYERS) + var * (domain.nx_local - 2*GHOST_LAYERS)] = 
                U[idx + var * domain.nx_local * domain.ny_local];
        }
    }
}

/**
 * Unpack data received from north neighbor into south ghost cells
 * 
 * Parallelization strategy:
 * - Each thread unpacks data for one ghost cell
 * - 1D thread blocks process cells along the x-direction
 * - Careful handling of memory access patterns
 * 
 * @param U Destination data array (solution variables)
 * @param buffer Source buffer containing received data
 * @param domain Domain information
 */
__global__ void unpack_south_halo(
    double* __restrict__ U,
    const double* __restrict__ buffer,
    const DomainInfo domain
) {
    // Calculate global thread index (along x-direction)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the domain boundary
    if (i >= GHOST_LAYERS && i < domain.nx_local - GHOST_LAYERS) {
        // Calculate index of ghost cell at south boundary
        int idx = (GHOST_LAYERS - 1) * domain.nx_local + i;
        
        // Unpack all components for this ghost cell
        #pragma unroll
        for (int var = 0; var < NUM_VARS; var++) {
            // Load from buffer with components grouped together
            U[idx + var * domain.nx_local * domain.ny_local] = 
                buffer[(i - GHOST_LAYERS) + var * (domain.nx_local - 2*GHOST_LAYERS)];
        }
    }
}

/**
 * Pack south boundary data for sending to neighboring process
 * 
 * Parallelization strategy:
 * - Each thread packs data for one cell along the boundary
 * - 1D thread blocks process cells along the x-direction
 * - Optimized for bandwidth-bound operations
 * 
 * @param U Source data array (solution variables)
 * @param buffer Destination buffer for packed data
 * @param domain Domain information
 */
__global__ void pack_south_halo(
    const double* __restrict__ U,
    double* __restrict__ buffer,
    const DomainInfo domain
) {
    // Calculate global thread index (along x-direction)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the domain boundary
    if (i >= GHOST_LAYERS && i < domain.nx_local - GHOST_LAYERS) {
        // Calculate index of interior cell adjacent to south boundary
        int idx = GHOST_LAYERS * domain.nx_local + i;
        
        // Pack all components for this cell
        #pragma unroll
        for (int var = 0; var < NUM_VARS; var++) {
            // Store in buffer with components grouped together
            buffer[(i - GHOST_LAYERS) + var * (domain.nx_local - 2*GHOST_LAYERS)] = 
                U[idx + var * domain.nx_local * domain.ny_local];
        }
    }
}

/**
 * Unpack data received from south neighbor into north ghost cells
 * 
 * Parallelization strategy:
 * - Each thread unpacks data for one ghost cell
 * - 1D thread blocks process cells along the x-direction
 * - Efficient memory access pattern
 * 
 * @param U Destination data array (solution variables)
 * @param buffer Source buffer containing received data
 * @param domain Domain information
 */
__global__ void unpack_north_halo(
    double* __restrict__ U,
    const double* __restrict__ buffer,
    const DomainInfo domain
) {
    // Calculate global thread index (along x-direction)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the domain boundary
    if (i >= GHOST_LAYERS && i < domain.nx_local - GHOST_LAYERS) {
        // Calculate index of ghost cell at north boundary
        int idx = (domain.ny_local - GHOST_LAYERS) * domain.nx_local + i;
        
        // Unpack all components for this ghost cell
        #pragma unroll
        for (int var = 0; var < NUM_VARS; var++) {
            // Load from buffer with components grouped together
            U[idx + var * domain.nx_local * domain.ny_local] = 
                buffer[(i - GHOST_LAYERS) + var * (domain.nx_local - 2*GHOST_LAYERS)];
        }
    }
}

/**
 * Apply physical boundary conditions at domain edges
 * 
 * Parallelization strategy:
 * - Each thread handles one ghost cell
 * - Different boundary types handled with minimal divergence
 * - Thread blocks organized to maximize occupancy
 * 
 * @param U Solution variables array
 * @param domain Domain information
 * @param bc_types Array of boundary condition types for each edge
 */
__global__ void apply_boundary_conditions(
    double* __restrict__ U,
    const DomainInfo domain,
    const int* __restrict__ bc_types
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if this thread is within the domain
    if (i < domain.nx_local && j < domain.ny_local) {
        // Determine if this cell is a ghost cell at a physical boundary
        bool is_west_ghost = (i < GHOST_LAYERS && domain.neighbors[WEST] < 0);
        bool is_east_ghost = (i >= domain.nx_local - GHOST_LAYERS && domain.neighbors[EAST] < 0);
        bool is_south_ghost = (j < GHOST_LAYERS && domain.neighbors[SOUTH] < 0);
        bool is_north_ghost = (j >= domain.ny_local - GHOST_LAYERS && domain.neighbors[NORTH] < 0);
        
        // Calculate 1D index for this cell
        int idx = j * domain.nx_local + i;
        
        // Apply boundary conditions based on boundary type
        if (is_west_ghost) {
            // Get corresponding interior cell index
            int interior_idx = j * domain.nx_local + (2*GHOST_LAYERS - 1 - i);
            
            // Apply boundary condition based on type
            switch (bc_types[WEST]) {
                case BC_WALL: {
                    // Wall boundary: reflect velocity
                    U[idx*NUM_VARS + 0] = U[interior_idx*NUM_VARS + 0];  // Density
                    U[idx*NUM_VARS + 1] = -U[interior_idx*NUM_VARS + 1]; // Negate x-momentum
                    U[idx*NUM_VARS + 2] = U[interior_idx*NUM_VARS + 2];  // y-momentum
                    U[idx*NUM_VARS + 3] = U[interior_idx*NUM_VARS + 3];  // Energy
                    break;
                }
                case BC_INFLOW: {
                    // Inflow boundary: use specified values
                    U[idx*NUM_VARS + 0] = INFLOW_DENSITY;
                    U[idx*NUM_VARS + 1] = INFLOW_DENSITY * INFLOW_VELOCITY_X;
                    U[idx*NUM_VARS + 2] = INFLOW_DENSITY * INFLOW_VELOCITY_Y;
                    U[idx*NUM_VARS + 3] = INFLOW_PRESSURE/(GAMMA-1.0) + 
                                         0.5 * INFLOW_DENSITY * 
                                         (INFLOW_VELOCITY_X*INFLOW_VELOCITY_X + 
                                          INFLOW_VELOCITY_Y*INFLOW_VELOCITY_Y);
                    break;
                }
                case BC_OUTFLOW: {
                    // Outflow boundary: extrapolate from interior
                    #pragma unroll
                    for (int var = 0; var < NUM_VARS; var++) {
                        U[idx*NUM_VARS + var] = U[interior_idx*NUM_VARS + var];
                    }
                    break;
                }
            }
        }
        else if (is_east_ghost) {
            // Similar implementation for east boundary
            // ...
        }
        else if (is_south_ghost) {
            // Similar implementation for south boundary
            // ...
        }
        else if (is_north_ghost) {
            // Similar implementation for north boundary
            // ...
        }
    }
}

/**
 * Launch configuration and wrapper for packing boundary data
 * 
 * @param U Source data array
 * @param buffers Array of buffers for each direction
 * @param domain Domain information
 * @param streams CUDA streams for asynchronous execution
 */
extern "C" void launch_pack_boundaries(
    const double* U,
    double** buffers,
    const DomainInfo domain,
    cudaStream_t* streams
) {
    // Calculate thread block size and grid dimensions
    int block_size = 256;  // Optimal block size for most GPUs
    
    // Pack west boundary if we have a west neighbor
    if (domain.neighbors[WEST] >= 0) {
        int grid_size = (domain.ny_local - 2*GHOST_LAYERS + block_size - 1) / block_size;
        pack_west_halo<<<grid_size, block_size, 0, streams[WEST]>>>(
            U, buffers[WEST], domain);
    }
    
    // Pack east boundary if we have an east neighbor
    if (domain.neighbors[EAST] >= 0) {
        int grid_size = (domain.ny_local - 2*GHOST_LAYERS + block_size - 1) / block_size;
        dim3 block(block_size);
        dim3 grid(grid_size, NUM_VARS);
        pack_east_halo<<<grid, block, 0, streams[EAST]>>>(
            U, buffers[EAST], domain.nx_local, domain.ny_local, NUM_VARS);
    }
    
    // Pack north boundary if we have a north neighbor
    if (domain.neighbors[NORTH] >= 0) {
        int grid_size = (domain.nx_local - 2*GHOST_LAYERS + block_size - 1) / block_size;
        pack_north_halo<<<grid_size, block_size, 0, streams[NORTH]>>>(
            U, buffers[NORTH], domain);
    }
    
    // Pack south boundary if we have a south neighbor
    if (domain.neighbors[SOUTH] >= 0) {
        int grid_size = (domain.nx_local - 2*GHOST_LAYERS + block_size - 1) / block_size;
        pack_south_halo<<<grid_size, block_size, 0, streams[SOUTH]>>>(
            U, buffers[SOUTH], domain);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Launch configuration and wrapper for unpacking boundary data
 * 
 * @param U Destination data array
 * @param buffers Array of buffers for each direction
 * @param domain Domain information
 * @param streams CUDA streams for asynchronous execution
 */
extern "C" void launch_unpack_boundaries(
    double* U,
    double** buffers,
    const DomainInfo domain,
    cudaStream_t* streams
) {
    // Calculate thread block size and grid dimensions
    int block_size = 256;  // Optimal block size for most GPUs
    
    // Unpack west boundary data if we have an east neighbor
    if (domain.neighbors[EAST] >= 0) {
        int grid_size = (domain.ny_local - 2*GHOST_LAYERS + block_size - 1) / block_size;
        unpack_east_halo<<<grid_size, block_size, 0, streams[EAST]>>>(
            U, buffers[EAST], domain);
    }
    
    // Unpack east boundary data if we have a west neighbor
    if (domain.neighbors[WEST] >= 0) {
        int grid_size = (domain.ny_local - 2*GHOST_LAYERS + block_size - 1) / block_size;
        dim3 block(block_size);
        dim3 grid(grid_size, NUM_VARS);
        unpack_west_halo<<<grid, block, 0, streams[WEST]>>>(
            U, buffers[WEST], domain.nx_local, domain.ny_local, NUM_VARS);
    }
    
    // Unpack north boundary data if we have a south neighbor
    if (domain.neighbors[SOUTH] >= 0) {
        int grid_size = (domain.nx_local - 2*GHOST_LAYERS + block_size - 1) / block_size;
        unpack_south_halo<<<grid_size, block_size, 0, streams[SOUTH]>>>(
            U, buffers[SOUTH], domain);
    }
    
    // Unpack south boundary data if we have a north neighbor
    if (domain.neighbors[NORTH] >= 0) {
        int grid_size = (domain.nx_local - 2*GHOST_LAYERS + block_size - 1) / block_size;
        unpack_north_halo<<<grid_size, block_size, 0, streams[NORTH]>>>(
            U, buffers[NORTH], domain);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Launch configuration and wrapper for applying physical boundary conditions
 * 
 * @param U Solution variables array
 * @param domain Domain information
 * @param bc_types Array of boundary condition types
 */
extern "C" void launch_apply_boundary_conditions(
    double* U,
    const DomainInfo domain,
    const int* bc_types
) {
    // Define grid and block dimensions
    dim3 block(16, 16);  // 2D block for 2D domain
    dim3 grid(
        (domain.nx_local + block.x - 1) / block.x,
        (domain.ny_local + block.y - 1) / block.y
    );
    
    // Launch kernel
    apply_boundary_conditions<<<grid, block>>>(U, domain, bc_types);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
} 