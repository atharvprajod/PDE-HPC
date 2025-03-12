/**
 * cuda_utils.h - CUDA helper functions and memory management
 * 
 * This header provides utilities for CUDA operations, including:
 * - Error checking macros
 * - GPU memory management through a wrapper class
 * - Device selection and initialization
 * - CUDA stream creation and management
 * 
 * Parallelization strategy:
 * - Unified memory for simplified memory management
 * - Multiple CUDA streams for concurrent execution
 * - Device selection based on MPI rank
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <mpi.h>

/**
 * CUDA error checking macro
 * 
 * This macro checks for CUDA errors and prints diagnostic information
 * when errors occur, making debugging easier.
 * 
 * @param err CUDA function call or error code to check
 */
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if(err_ != cudaSuccess) \
            fprintf(stderr, "CUDA error %d: %s\n", err_, cudaGetErrorString(err_)); \
    } while(0)

/**
 * GPU memory management class
 * 
 * This template class provides a simple RAII wrapper around CUDA memory
 * allocation and deallocation, using unified memory for simplified access.
 * 
 * @tparam T Data type to be stored in GPU memory
 */
template<typename T>
class DeviceArray {
public:
    T* data;      // Pointer to GPU memory
    size_t size;  // Number of elements
    
    /**
     * Constructor allocates unified memory accessible from both CPU and GPU
     * 
     * @param n Number of elements to allocate
     */
    DeviceArray(size_t n) : size(n) {
        CUDA_CHECK(cudaMallocManaged(&data, n*sizeof(T)));
    }
    
    /**
     * Destructor automatically frees GPU memory
     */
    ~DeviceArray() { CUDA_CHECK(cudaFree(data)); }
};

/**
 * Set up CUDA device based on MPI rank
 * 
 * This function selects a GPU device for each MPI process based on its rank,
 * distributing processes across available GPUs in a round-robin fashion.
 * 
 * @param comm MPI communicator
 */
inline void setup_cuda_device(MPI_Comm comm) {
    int dev_count, rank;
    cudaGetDeviceCount(&dev_count);
    MPI_Comm_rank(comm, &rank);
    cudaSetDevice(rank % dev_count);  // Round-robin assignment
}

/**
 * Create a CUDA stream with specified priority
 * 
 * This function creates a non-blocking CUDA stream that can be used
 * for asynchronous operations.
 * 
 * @param stream Pointer to store the created stream
 */
inline void create_stream_with_priority(cudaStream_t* stream) {
    cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, 0);
}

#endif