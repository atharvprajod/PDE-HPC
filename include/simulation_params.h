/**
 * simulation_params.h - Simulation configuration parameters
 * 
 * This header defines the structure for storing simulation parameters
 * loaded from configuration files. These parameters control the
 * simulation behavior, including domain size, time stepping, and
 * physical properties.
 */

/**
 * Simulation Parameters Structure
 * 
 * This structure holds the core parameters that define the simulation,
 * including domain size, grid resolution, time stepping parameters,
 * and physical properties.
 */
struct SimulationParams {
    double t_final;  // Final simulation time
    double cfl;      // CFL number for adaptive time stepping
    double dx, dy;   // Grid spacing in x and y directions
    int nx, ny;      // Number of grid cells in x and y directions
};

/**
 * Load simulation parameters from configuration file
 * 
 * This function reads simulation parameters from a configuration file
 * and populates the SimulationParams structure.
 * 
 * @param filename Path to the configuration file
 * @param params Structure to store the loaded parameters
 */
void load_parameters(const char* filename, SimulationParams& params); 