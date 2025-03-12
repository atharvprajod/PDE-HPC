/**
 * distributed_array.h - GPU-accelerated distributed array
 * 
 * This header defines a template class for managing distributed arrays
 * that reside primarily on GPUs. It handles memory allocation, deallocation,
 * and provides methods for accessing and manipulating the array data.
 * 
 * Parallelization strategy:
 * - GPU memory allocation for computational efficiency
 * - Support for MPI communication between distributed arrays
 * - Zero-copy pointer swapping for efficient time stepping
 */

/**
 * Distributed Array Template Class
 * 
 * This class manages a multi-dimensional array distributed across
 * MPI processes and stored on GPU memory. It provides methods for
 * accessing and manipulating the array data.
 * 
 * @tparam T Data type stored in the array
 */
template <typename T>
class DistributedArray {
    T* d_ptr;           // Device (GPU) memory pointer
    size_t nx, ny, comps;  // Dimensions and components
    
public:
    /**
     * Constructor allocates GPU memory for the array
     * 
     * @param x Number of cells in x-direction
     * @param y Number of cells in y-direction
     * @param c Number of components per cell
     */
    DistributedArray(size_t x, size_t y, size_t c) : nx(x), ny(y), comps(c) {
        cudaMalloc(&d_ptr, x*y*c*sizeof(T));
    }
    
    /**
     * Destructor frees GPU memory
     */
    ~DistributedArray() { cudaFree(d_ptr); }
    
    /**
     * Get pointer to device memory
     * 
     * @return Pointer to the array data on GPU
     */
    T* device_ptr() const { return d_ptr; }
    
    /**
     * Swap contents with another distributed array
     * 
     * This is a zero-copy operation that simply exchanges pointers,
     * making it very efficient for time-stepping algorithms.
     * 
     * @param other Another distributed array to swap with
     */
    void swap(DistributedArray& other) {
        std::swap(d_ptr, other.d_ptr);
        std::swap(nx, other.nx);
        std::swap(ny, other.ny);
        std::swap(comps, other.comps);
    }
}; 