/**
 * simulation_functions.h - Core simulation functions
 * 
 * This header declares the main simulation functions used throughout
 * the codebase, including time step calculation, resource management,
 * and time integration.
 * 
 * These functions form the backbone of the simulation algorithm,
 * connecting the various components into a cohesive whole.
 */

#ifndef SIMULATION_FUNCTIONS_H
#define SIMULATION_FUNCTIONS_H

#include "mesh_partition.h"
#include "cuda_utils.h"

/**
 * Compute stable time step based on CFL condition
 * 
 * This function calculates the maximum stable time step based on
 * the current solution state and the CFL stability criterion.
 * 
 * @param u Solution array on GPU
 * @param nx Number of cells in x-direction
 * @param ny Number of cells in y-direction
 * @param dx Cell size in x-direction
 * @param dy Cell size in y-direction
 * @return Maximum stable time step
 */
double compute_cfl(const double* u, int nx, int ny, double dx, double dy);

/**
 * Clean up simulation resources
 * 
 * This function properly destroys PETSc objects and frees resources
 * at the end of the simulation.
 * 
 * @param A PETSc matrix to destroy
 * @param p PETSc vector to destroy
 * @param rhs PETSc vector to destroy
 */
void finalize_simulation(Mat& A, Vec& p, Vec& rhs);

/**
 * Create PETSc structures optimized for GPU execution
 * 
 * This function creates the PETSc objects needed for the implicit
 * pressure solve, configured for GPU acceleration.
 * 
 * @param part Domain partition information
 * @param A Output sparse matrix for pressure system
 * @param p Output vector for pressure solution
 * @param rhs Output vector for right-hand side
 */
void create_petsc_gpu_structures(const MeshPartition& part, 
                               Mat* A, Vec* p, Vec* rhs);

/**
 * Perform one step of IMEX time integration
 * 
 * This function advances the solution by one time step using
 * the IMEX (Implicit-Explicit) scheme.
 * 
 * @param u_old Current solution array on GPU
 * @param fluxes Computed fluxes on GPU
 * @param dt Time step size
 * @param part Mesh partition information
 * @param A PETSc matrix for pressure system
 * @param p PETSc vector for pressure solution
 * @param rhs PETSc vector for right-hand side
 * @param stream CUDA stream for asynchronous execution
 */
void imex_step(double* u_old, double* fluxes, double dt, 
              const MeshPartition& part, Mat A, Vec p, Vec rhs,
              cudaStream_t stream);

#endif 