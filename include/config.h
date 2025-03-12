/**
 * config.h - Global configuration parameters
 * 
 * This header defines global constants and structures used throughout
 * the simulation. It includes physical parameters, numerical settings,
 * and domain decomposition structures.
 * 
 * The configuration parameters control:
 * - Simulation dimensionality and resolution
 * - Numerical stability factors
 * - Hardware acceleration options
 * - Physical model constants
 */

#pragma once

/**
 * Simulation Parameters
 * These control the basic numerical properties of the simulation
 */
constexpr int DIM = 2;            // Dimensionality (2D simulation)
constexpr int GHOST_LAYERS = 1;   // Number of ghost cell layers (1 for 2nd order schemes)
constexpr double CFL = 0.4;       // Courant-Friedrichs-Lewy stability factor
constexpr bool USE_GPU = true;    // Enable GPU acceleration
constexpr bool USE_MIXED_PREC = true; // Use mixed precision (FP16/FP32) for better performance

/**
 * Physical Constants
 * These define the physical properties of the simulated system
 */
constexpr double REYNOLDS = 1000.0;  // Reynolds number for viscous flows
constexpr double MACH = 0.1;         // Mach number for compressible flows

/**
 * Domain Decomposition Structure
 * 
 * This structure holds information about how the global domain is
 * partitioned across MPI processes, including local dimensions,
 * global position, and neighboring processes.
 */
struct DomainInfo {
    int nx_global, ny_global;     // Global grid dimensions
    int nx_local, ny_local;       // Local grid dimensions for this process
    int offset_x, offset_y;       // Global offset of this local domain
    int neighbors[4];             // Neighboring processes (NESW order, -1 if none)
};