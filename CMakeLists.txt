cmake_minimum_required(VERSION 3.20)
project(pde-sim LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PETSc REQUIRED)

add_executable(pde-sim
    src/core/main.cpp
    src/core/mesh_decomposition.cpp
    src/core/pressure_solver.cpp
    src/cuda/flux_kernels.cu
    src/cuda/boundary_kernels.cu
    src/utils/mpi_comm.cpp
)

target_include_directories(pde-sim PRIVATE include)
target_link_libraries(pde-sim PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    PETSc::PETSc
)

set_target_properties(pde-sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)

include_directories(
    ${MPI_INCLUDE_PATH}
    ${PETSC_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
) 