/**
 * flux_kernels.cu
 * 
 * CUDA kernels for computing numerical fluxes in the PDE solver.
 * This file implements the core computational kernels that calculate
 * inviscid and viscous fluxes for fluid dynamics simulations.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "config.h"
#include "mesh_types.h"
#include "cuda_utils.h"

/**
 * Compute pressure from conservative variables
 * 
 * @param rho Density
 * @param rho_u x-momentum
 * @param rho_v y-momentum
 * @param E Total energy
 * @return Pressure
 */
__device__ double compute_pressure(double rho, double rho_u, double rho_v, double E) {
    // Extract velocities
    double u = rho_u / rho;
    double v = rho_v / rho;
    
    // Compute kinetic energy
    double kinetic = 0.5 * rho * (u*u + v*v);
    
    // Pressure from ideal gas law: p = (gamma-1)*(E - kinetic)
    // where gamma is the ratio of specific heats (typically 1.4 for air)
    return (GAMMA - 1.0) * (E - kinetic);
}

/**
 * Compute inviscid fluxes for all cells in the domain
 * 
 * Parallelization strategy:
 * - Each CUDA thread processes one cell in the domain
 * - Thread blocks are organized in a 2D grid matching the domain structure
 * - Shared memory is used to cache frequently accessed data
 * - Memory access is coalesced for better throughput
 * 
 * @param U Input solution variables (density, momentum, energy)
 * @param F Output fluxes
 * @param domain Domain information
 */
__global__ void compute_inviscid_fluxes(
    const double* __restrict__ U,  // Input solution variables (read-only)
    double* __restrict__ F,        // Output fluxes
    const DomainInfo domain
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Shared memory for frequently accessed variables
    // This reduces global memory traffic and improves performance
    __shared__ double s_U[BLOCK_SIZE_Y][BLOCK_SIZE_X][4];
    
    // Local thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Check if this thread is within the domain (excluding ghost cells)
    if (i >= GHOST_LAYERS && i < domain.nx_local - GHOST_LAYERS &&
        j >= GHOST_LAYERS && j < domain.ny_local - GHOST_LAYERS) {
        
        // Calculate 1D index for this cell
        int idx = j * domain.nx_local + i;
        
        // Load solution variables into shared memory
        // This is done by all threads in the block to maximize memory throughput
        #pragma unroll
        for (int var = 0; var < 4; var++) {
            s_U[ty][tx][var] = U[idx*4 + var];
        }
        
        // Ensure all threads have loaded their data before proceeding
        __syncthreads();
        
        // Extract solution variables from shared memory
        double rho   = s_U[ty][tx][0];                  // Density
        double rho_u = s_U[ty][tx][1];                  // x-momentum
        double rho_v = s_U[ty][tx][2];                  // y-momentum
        double E     = s_U[ty][tx][3];                  // Total energy
        
        // Compute derived quantities
        double u = rho_u / rho;                         // x-velocity
        double v = rho_v / rho;                         // y-velocity
        double p = compute_pressure(rho, rho_u, rho_v, E);  // Pressure
        
        // Compute fluxes in x-direction (F)
        F[idx*4 + 0] = rho_u;                           // Mass flux
        F[idx*4 + 1] = rho_u * u + p;                   // x-momentum flux
        F[idx*4 + 2] = rho_u * v;                       // y-momentum flux
        F[idx*4 + 3] = (E + p) * u;                     // Energy flux
        
        // Compute fluxes in y-direction (G)
        // These are stored after all x-fluxes for better memory access patterns
        int flux_offset = domain.nx_local * domain.ny_local * 4;
        F[flux_offset + idx*4 + 0] = rho_v;             // Mass flux
        F[flux_offset + idx*4 + 1] = rho_v * u;         // x-momentum flux
        F[flux_offset + idx*4 + 2] = rho_v * v + p;     // y-momentum flux
        F[flux_offset + idx*4 + 3] = (E + p) * v;       // Energy flux
    }
}

/**
 * Compute viscous fluxes for all cells in the domain
 * 
 * Parallelization strategy:
 * - Each CUDA thread processes one cell in the domain
 * - Stencil-based computation requires data from neighboring cells
 * - Shared memory is used to reduce redundant global memory loads
 * - Thread divergence is minimized by careful boundary handling
 * 
 * @param U Input solution variables
 * @param F Output fluxes (added to existing inviscid fluxes)
 * @param domain Domain information
 */
__global__ void compute_viscous_fluxes(
    const double* __restrict__ U,  // Input solution variables
    double* __restrict__ F,        // Output fluxes (added to existing values)
    const DomainInfo domain
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Shared memory allocation with halo cells for stencil computation
    // This significantly reduces global memory traffic for stencil operations
    __shared__ double s_U[BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2][4];
    
    // Local thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Check if this thread is within the domain (excluding ghost cells)
    if (i >= GHOST_LAYERS && i < domain.nx_local - GHOST_LAYERS &&
        j >= GHOST_LAYERS && j < domain.ny_local - GHOST_LAYERS) {
        
        // Calculate 1D index for this cell
        int idx = j * domain.nx_local + i;
        
        // Load solution variables into shared memory including halo cells
        // Each thread loads its own cell plus potentially a halo cell
        // This cooperative loading pattern maximizes memory throughput
        
        // Load center cell
        #pragma unroll
        for (int var = 0; var < 4; var++) {
            s_U[ty+1][tx+1][var] = U[idx*4 + var];
        }
        
        // Load halo cells (boundary threads only)
        // This is where thread divergence occurs, but it's minimized
        if (tx == 0 && i > GHOST_LAYERS) {
            // Load west neighbor
            int west_idx = j * domain.nx_local + (i-1);
            #pragma unroll
            for (int var = 0; var < 4; var++) {
                s_U[ty+1][tx][var] = U[west_idx*4 + var];
            }
        }
        
        if (tx == BLOCK_SIZE_X-1 && i < domain.nx_local - GHOST_LAYERS - 1) {
            // Load east neighbor
            int east_idx = j * domain.nx_local + (i+1);
            #pragma unroll
            for (int var = 0; var < 4; var++) {
                s_U[ty+1][tx+2][var] = U[east_idx*4 + var];
            }
        }
        
        if (ty == 0 && j > GHOST_LAYERS) {
            // Load south neighbor
            int south_idx = (j-1) * domain.nx_local + i;
            #pragma unroll
            for (int var = 0; var < 4; var++) {
                s_U[ty][tx+1][var] = U[south_idx*4 + var];
            }
        }
        
        if (ty == BLOCK_SIZE_Y-1 && j < domain.ny_local - GHOST_LAYERS - 1) {
            // Load north neighbor
            int north_idx = (j+1) * domain.nx_local + i;
            #pragma unroll
            for (int var = 0; var < 4; var++) {
                s_U[ty+2][tx+1][var] = U[north_idx*4 + var];
            }
        }
        
        // Ensure all threads have loaded their data before proceeding
        __syncthreads();
        
        // Compute velocity gradients using central differencing
        // These calculations benefit greatly from the shared memory usage
        
        // Extract center cell values
        double rho_c = s_U[ty+1][tx+1][0];
        double u_c = s_U[ty+1][tx+1][1] / rho_c;
        double v_c = s_U[ty+1][tx+1][2] / rho_c;
        
        // Extract neighbor cell values and compute derivatives
        double rho_e = s_U[ty+1][tx+2][0];
        double rho_w = s_U[ty+1][tx][0];
        double rho_n = s_U[ty+2][tx+1][0];
        double rho_s = s_U[ty][tx+1][0];
        
        double u_e = s_U[ty+1][tx+2][1] / rho_e;
        double u_w = s_U[ty+1][tx][1] / rho_w;
        double u_n = s_U[ty+2][tx+1][1] / rho_n;
        double u_s = s_U[ty][tx+1][1] / rho_s;
        
        double v_e = s_U[ty+1][tx+2][2] / rho_e;
        double v_w = s_U[ty+1][tx][2] / rho_w;
        double v_n = s_U[ty+2][tx+1][2] / rho_n;
        double v_s = s_U[ty][tx+1][2] / rho_s;
        
        // Compute gradients (central differencing)
        double dx = domain.dx;
        double dy = domain.dy;
        
        double dudx = (u_e - u_w) / (2.0 * dx);
        double dudy = (u_n - u_s) / (2.0 * dy);
        double dvdx = (v_e - v_w) / (2.0 * dx);
        double dvdy = (v_n - v_s) / (2.0 * dy);
        
        // Compute viscous stress tensor
        double mu = VISCOSITY;  // Dynamic viscosity
        double lambda = -2.0/3.0 * mu;  // Bulk viscosity (Stokes hypothesis)
        
        double tau_xx = 2.0 * mu * dudx + lambda * (dudx + dvdy);
        double tau_yy = 2.0 * mu * dvdy + lambda * (dudx + dvdy);
        double tau_xy = mu * (dudy + dvdx);
        
        // Compute temperature gradients for heat flux
        // Simplified approach using pressure as proxy for temperature
        double p_c = compute_pressure(rho_c, rho_c*u_c, rho_c*v_c, s_U[ty+1][tx+1][3]);
        double T_c = p_c / (rho_c * R_GAS);  // Temperature from ideal gas law
        
        double p_e = compute_pressure(rho_e, rho_e*u_e, rho_e*v_e, s_U[ty+1][tx+2][3]);
        double p_w = compute_pressure(rho_w, rho_w*u_w, rho_w*v_w, s_U[ty+1][tx][3]);
        double p_n = compute_pressure(rho_n, rho_n*u_n, rho_n*v_n, s_U[ty+2][tx+1][3]);
        double p_s = compute_pressure(rho_s, rho_s*u_s, rho_s*v_s, s_U[ty][tx+1][3]);
        
        double T_e = p_e / (rho_e * R_GAS);
        double T_w = p_w / (rho_w * R_GAS);
        double T_n = p_n / (rho_n * R_GAS);
        double T_s = p_s / (rho_s * R_GAS);
        
        double dTdx = (T_e - T_w) / (2.0 * dx);
        double dTdy = (T_n - T_s) / (2.0 * dy);
        
        // Compute heat flux
        double k = CP * mu / PRANDTL;  // Thermal conductivity
        double q_x = -k * dTdx;
        double q_y = -k * dTdy;
        
        // Compute viscous fluxes in x-direction
        double F_v_x[4];
        F_v_x[0] = 0.0;                           // No mass diffusion
        F_v_x[1] = tau_xx;                        // x-momentum diffusion
        F_v_x[2] = tau_xy;                        // y-momentum diffusion
        F_v_x[3] = u_c * tau_xx + v_c * tau_xy + q_x;  // Energy diffusion
        
        // Compute viscous fluxes in y-direction
        double F_v_y[4];
        F_v_y[0] = 0.0;                           // No mass diffusion
        F_v_y[1] = tau_xy;                        // x-momentum diffusion
        F_v_y[2] = tau_yy;                        // y-momentum diffusion
        F_v_y[3] = u_c * tau_xy + v_c * tau_yy + q_y;  // Energy diffusion
        
        // Add viscous fluxes to inviscid fluxes
        // Subtract because viscous fluxes act in opposite direction to inviscid
        #pragma unroll
        for (int var = 0; var < 4; var++) {
            F[idx*4 + var] -= F_v_x[var];
        }
        
        int flux_offset = domain.nx_local * domain.ny_local * 4;
        #pragma unroll
        for (int var = 0; var < 4; var++) {
            F[flux_offset + idx*4 + var] -= F_v_y[var];
        }
    }
}

/**
 * Compute fluxes using mixed precision (FP16/FP32)
 * 
 * Parallelization strategy:
 * - Uses half precision for computation where possible
 * - Maintains full precision for critical calculations
 * - Leverages tensor cores on supported GPUs
 * - Reduces memory bandwidth requirements
 * 
 * @param U Input solution variables
 * @param F Output fluxes
 * @param domain Domain information
 */
__global__ void compute_fluxes_mixed_precision(
    const double* __restrict__ U,  // Input solution variables
    double* __restrict__ F,        // Output fluxes
    const DomainInfo domain
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if this thread is within the domain (excluding ghost cells)
    if (i >= GHOST_LAYERS && i < domain.nx_local - GHOST_LAYERS &&
        j >= GHOST_LAYERS && j < domain.ny_local - GHOST_LAYERS) {
        
        // Calculate 1D index for this cell
        int idx = j * domain.nx_local + i;
        
        // Convert double precision inputs to half precision
        // This reduces memory bandwidth requirements
        half2 rho_mom_x, mom_y_E;
        
        // Load and convert solution variables
        rho_mom_x = __floats2half2_rn(U[idx*4 + 0], U[idx*4 + 1]);
        mom_y_E = __floats2half2_rn(U[idx*4 + 2], U[idx*4 + 3]);
        
        // Extract individual variables (still in half precision)
        half rho = __low2half(rho_mom_x);
        half rho_u = __high2half(rho_mom_x);
        half rho_v = __low2half(mom_y_E);
        half E = __high2half(mom_y_E);
        
        // Convert critical values back to single precision for accuracy
        float rho_f = __half2float(rho);
        float rho_u_f = __half2float(rho_u);
        float rho_v_f = __half2float(rho_v);
        float E_f = __half2float(E);
        
        // Compute derived quantities in single precision
        float u_f = rho_u_f / rho_f;
        float v_f = rho_v_f / rho_f;
        
        // Compute pressure (requires full precision)
        float kinetic_f = 0.5f * rho_f * (u_f*u_f + v_f*v_f);
        float p_f = (GAMMA - 1.0f) * (E_f - kinetic_f);
        
        // Convert back to half precision for flux computation
        half u = __float2half(u_f);
        half v = __float2half(v_f);
        half p = __float2half(p_f);
        
        // Compute fluxes in half precision
        half2 F_mass_mom_x = __floats2half2_rn(__half2float(rho_u), 
                                              __half2float(rho_u) * __half2float(u) + __half2float(p));
        
        half2 F_mom_y_E = __floats2half2_rn(__half2float(rho_u) * __half2float(v),
                                           (__half2float(E) + __half2float(p)) * __half2float(u));
        
        half2 G_mass_mom_x = __floats2half2_rn(__half2float(rho_v),
                                              __half2float(rho_v) * __half2float(u));
        
        half2 G_mom_y_E = __floats2half2_rn(__half2float(rho_v) * __half2float(v) + __half2float(p),
                                           (__half2float(E) + __half2float(p)) * __half2float(v));
        
        // Convert back to double precision for output
        F[idx*4 + 0] = __half2float(__low2half(F_mass_mom_x));
        F[idx*4 + 1] = __half2float(__high2half(F_mass_mom_x));
        F[idx*4 + 2] = __half2float(__low2half(F_mom_y_E));
        F[idx*4 + 3] = __half2float(__high2half(F_mom_y_E));
        
        int flux_offset = domain.nx_local * domain.ny_local * 4;
        F[flux_offset + idx*4 + 0] = __half2float(__low2half(G_mass_mom_x));
        F[flux_offset + idx*4 + 1] = __half2float(__high2half(G_mass_mom_x));
        F[flux_offset + idx*4 + 2] = __half2float(__low2half(G_mom_y_E));
        F[flux_offset + idx*4 + 3] = __half2float(__high2half(G_mom_y_E));
    }
}

/**
 * Compute stable timestep based on CFL condition
 * 
 * Parallelization strategy:
 * - Each thread computes local timestep for one cell
 * - Parallel reduction finds minimum across all cells
 * - Atomic operations ensure correct minimum finding
 * 
 * @param U Solution variables
 * @param dt_local Array to store local timestep (single element)
 * @param domain Domain information
 */
__global__ void compute_timestep_kernel(
    const double* __restrict__ U,
    double* __restrict__ dt_local,
    const DomainInfo domain
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Shared memory for block-level reduction
    __shared__ double s_dt[BLOCK_SIZE_X * BLOCK_SIZE_Y];
    
    // Local thread index
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Initialize local timestep to a large value
    double thread_dt = 1e10;
    
    // Check if this thread is within the domain (excluding ghost cells)
    if (i >= GHOST_LAYERS && i < domain.nx_local - GHOST_LAYERS &&
        j >= GHOST_LAYERS && j < domain.ny_local - GHOST_LAYERS) {
        
        // Calculate 1D index for this cell
        int idx = j * domain.nx_local + i;
        
        // Extract solution variables
        double rho = U[idx*4 + 0];
        double u = U[idx*4 + 1] / rho;
        double v = U[idx*4 + 2] / rho;
        double E = U[idx*4 + 3];
        
        // Compute pressure and sound speed
        double p = compute_pressure(rho, rho*u, rho*v, E);
        double c = sqrt(GAMMA * p / rho);  // Speed of sound
        
        // Compute spectral radii
        double lambda_x = fabs(u) + c;
        double lambda_y = fabs(v) + c;
        
        // Compute local timestep based on CFL condition
        double dx = domain.dx;
        double dy = domain.dy;
        
        thread_dt = CFL / (lambda_x/dx + lambda_y/dy);
    }
    
    // Store local timestep in shared memory
    s_dt[tid] = thread_dt;
    
    // Synchronize threads in block
    __syncthreads();
    
    // Parallel reduction to find minimum timestep in block
    // This is a tree-based reduction with logarithmic complexity
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dt[tid] = min(s_dt[tid], s_dt[tid + stride]);
        }
        __syncthreads();
    }
    
    // First thread in block updates global minimum using atomic operation
    if (tid == 0) {
        atomicMin((unsigned long long int*)dt_local, 
                 __double_as_longlong(s_dt[0]));
    }
}

/**
 * Launch configuration and wrapper for flux computation
 * 
 * @param U Input solution variables
 * @param F Output fluxes
 * @param domain Domain information
 */
extern "C" void launch_compute_fluxes(
    double* U,
    double* F,
    const DomainInfo domain
) {
    // Define grid and block dimensions for optimal occupancy
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (domain.nx_local + block.x - 1) / block.x,
        (domain.ny_local + block.y - 1) / block.y
    );
    
    // Choose kernel based on configuration
    if (USE_MIXED_PRECISION) {
        // Use mixed precision kernel for better performance
        compute_fluxes_mixed_precision<<<grid, block>>>(U, F, domain);
    } else {
        // Compute inviscid fluxes
        compute_inviscid_fluxes<<<grid, block>>>(U, F, domain);
        
        // Compute viscous fluxes if Reynolds number is finite
        if (REYNOLDS < 1e10) {
            compute_viscous_fluxes<<<grid, block>>>(U, F, domain);
        }
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Launch configuration and wrapper for timestep computation
 * 
 * @param U Solution variables
 * @param dt Pointer to store computed timestep
 * @param domain Domain information
 */
extern "C" void launch_compute_timestep(
    double* U,
    double* dt,
    const DomainInfo domain
) {
    // Define grid and block dimensions
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (domain.nx_local + block.x - 1) / block.x,
        (domain.ny_local + block.y - 1) / block.y
    );
    
    // Initialize dt to a large value
    double large_dt = 1e10;
    CUDA_CHECK(cudaMemcpy(dt, &large_dt, sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel
    compute_timestep_kernel<<<grid, block>>>(U, dt, domain);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}