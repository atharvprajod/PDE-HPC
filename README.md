# PDE-Sim: GPU-Accelerated Parallel PDE Solver

A high-performance parallel solver for partial differential equations using MPI, CUDA, and PETSc. Designed for large-scale fluid dynamics simulations with mixed-precision computing and domain decomposition.

## Features

- **Hybrid Parallelization**
  - MPI for distributed memory parallelism
  - CUDA for GPU acceleration
  - Mixed-precision (FP16/FP32) computation for optimal performance
  - Asynchronous boundary exchanges

- **Numerical Methods**
  - IMEX (Implicit-Explicit) time integration
  - Second-order finite volume spatial discretization
  - Pressure Poisson solver using PETSc's GPU-enabled solvers
  - Adaptive timestepping with CFL condition

- **Performance Optimizations**
  - Efficient domain decomposition with minimal communication
  - CUDA-aware MPI for direct GPU-GPU transfers
  - Optimized ghost cell exchange patterns
  - Smart memory management with unified memory when beneficial

## Requirements

- CUDA Toolkit 11.0+
- MPI implementation (OpenMPI/MPICH)
- PETSc 3.15+ with CUDA support
- CMake 3.20+
- C++17 compliant compiler
- Python 3.8+ (for preprocessing tools)

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

## Project Structure

```
├── src/
│   ├── core/
│   │   ├── main.cpp              # Program entry point
│   │   ├── mesh_decomposition.cpp # Domain partitioning
│   │   ├── pressure_solver.cpp   # PETSc-based Poisson solver
│   │   ├── petsc_utils.cpp      # PETSc wrapper utilities
│   │   └── imex_integrator.cpp  # Time integration
│   ├── cuda/
│   │   ├── flux_kernels.cu      # CUDA flux computations
│   │   └── boundary_kernels.cu  # GPU boundary handling
│   └── utils/
│       └── mpi_comm.cpp         # MPI communication wrappers
├── include/
│   ├── config.h                 # Simulation parameters
│   ├── mesh_types.h            # Mesh data structures
│   ├── mesh_partition.h        # Partitioning interfaces
│   ├── distributed_array.h     # Distributed data container
│   ├── cuda_utils.h            # CUDA helper functions
│   └── boundary.h              # Boundary condition handling
└── preprocessing/
    └── decompose_mesh.py       # Mesh preprocessing tool
```

## Implementation Details

### Domain Decomposition

The simulation domain is partitioned using a structured grid approach:

```cpp
// From include/mesh_types.h
struct DomainInfo {
    int nx_global, ny_global;     // Global grid size
    int nx_local, ny_local;       // Local partition size
    int offset_x, offset_y;       // Partition offset
    int neighbors[4];             // NESW neighbors (-1 if none)
};
```

Mesh decomposition is handled by the preprocessing tool:
```python
# preprocessing/decompose_mesh.py
# Usage: python decompose_mesh.py <nx> <ny> <num_procs>
```

### Distributed Arrays

Data distribution is managed through a templated container:

```cpp
// From include/distributed_array.h
template<typename T>
class DeviceArray {
    T* data;
    size_t size;
    // ... memory management and MPI communication methods
};
```

### Time Integration

The IMEX scheme combines explicit flux calculations with implicit pressure solves:

```cpp
// From src/core/imex_integrator.cpp
while (t < SIMULATION_TIME) {
    // Explicit flux computation (GPU)
    compute_fluxes<<<blocks, threads>>>(U.data, F.data, domain);
    
    // Implicit pressure solve (GPU-enabled PETSc)
    if (implicit_step_required(t)) {
        solve_pressure_poisson(A, rhs, &p);
        update_velocity<<<...>>>(U.data, p, domain);
    }

    // Boundary exchange
    exchange_ghost_cells(U.data, domain);
    
    // Adaptive timestepping
    dt = compute_cfl(U.data, domain);
    MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
}
```

### Boundary Conditions

Boundary handling is implemented using CUDA kernels:

```cpp
// From src/cuda/boundary_kernels.cu
__global__ void apply_boundary_conditions(
    double* U,
    const DomainInfo domain,
    const BoundaryType* types
);
```

### Configuration

Simulation parameters can be modified in `include/config.h`:

```cpp
// Simulation Parameters
constexpr int DIM = 2;            // 2D simulation
constexpr int GHOST_LAYERS = 1;   // 1-cell overlap for 2nd order
constexpr double CFL = 0.4;       // Stability factor
constexpr bool USE_GPU = true;    // GPU acceleration
constexpr bool USE_MIXED_PREC = true; // FP16/FP32 mixed precision

// Physical Constants
constexpr double REYNOLDS = 1000.0;
constexpr double MACH = 0.1;
```

## Usage

1. Prepare the mesh decomposition:
```bash
python preprocessing/decompose_mesh.py 1024 1024 4
```

2. Run the simulation:
```bash
mpirun -np 4 ./pde-sim input.cfg
```

3. Monitor performance:
```bash
nvidia-smi dmon -s pucvt
```

## Performance Optimization Tips

1. **GPU Memory Management**
   - Use pinned memory for host-device transfers
   - Leverage unified memory for large domains
   - Minimize host-device synchronization

2. **MPI Communication**
   - Enable CUDA-aware MPI when available
   - Overlap computation with communication
   - Use non-blocking communications

3. **Load Balancing**
   - Adjust domain decomposition based on GPU capabilities
   - Consider dynamic load balancing for adaptive meshes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

Your Name - Initial work

## Acknowledgments

- PETSc development team for their excellent solver library
- CUDA development team for GPU computing support
- The open-source CFD community for inspiration and algorithms
```
