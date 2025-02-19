#!/bin/bash
PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux-cuda \
./configure --with-cuda=1 --download-fblaslapack=1 \
--with-precision=double --with-cc=mpicc --with-cxx=mpicxx 