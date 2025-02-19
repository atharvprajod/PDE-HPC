import h5py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with h5py.File(f"output_{rank:04d}.h5", "r") as f:
    rho = f["density"][:]
    local_mass = np.sum(rho)

total_mass = comm.allreduce(local_mass, op=MPI.SUM)

if rank == 0:
    print(f"Total system mass: {total_mass:.6e}")
    assert np.isclose(total_mass, INITIAL_MASS, rtol=1e-4), "Mass not conserved!"