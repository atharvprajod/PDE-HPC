void exchange_ghost_cells_async(CudaArray& u, MeshPartition& part) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for(int dir=0; dir<part.num_neighbors; dir++) {
        // Post receives
        MPI_Irecv(part.recv_buffers[dir], ..., 
                 part.neighbors[dir], tag, 
                 MPI_COMM_WORLD, &requests[dir]);
        
        // Pack and send
        pack_halo_data<<<..., stream>>>(u.dev_ptr, part.send_buffers[dir]);
        MPI_Isend(part.send_buffers[dir], ..., 
                 part.neighbors[dir], tag,
                 MPI_COMM_WORLD, &send_req[dir]);
    }
    
    // Wait and unpack
    MPI_Waitall(...);
    unpack_halo_data<<<..., stream>>>(u.dev_ptr, part.recv_buffers);
    cudaStreamDestroy(stream);
}