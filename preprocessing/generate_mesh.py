#!/usr/bin/env python3
import argparse
import numpy as np
import h5py

def main():
    parser = argparse.ArgumentParser(
        description="Generate a uniform 2D mesh for the PDE simulation."
    )
    parser.add_argument("-nx", type=int, default=100, help="Number of grid points in x-direction")
    parser.add_argument("-ny", type=int, default=100, help="Number of grid points in y-direction")
    parser.add_argument("-o", "--output", type=str, default="mesh.h5", help="Output HDF5 mesh file")
    args = parser.parse_args()

    nx, ny = args.nx, args.ny

    # Create coordinate arrays (cell centers in [0,1])
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")  # shape (nx, ny)

    # Write the mesh into an HDF5 file.
    with h5py.File(args.output, "w") as f:
        # Store the grid sizes as file attributes
        f.attrs["nx"] = nx
        f.attrs["ny"] = ny
        # Also store the coordinate arrays for later postprocessing/visualization.
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        print(f"Mesh generated: {nx} x {ny} grid, saved to '{args.output}'")

if __name__ == "__main__":
    main()
