PETSC_DIR = /opt/petsc
CUDA_HOME = /usr/local/cuda
MPI_HOME = /usr/lib/x86_64-linux-gnu/openmpi

CFLAGS = -O3 -march=native -std=c++17
CUFLAGS = -arch=sm_80 -Xcompiler="-fopenmp -O3"
INCLUDES = -Iinclude -I$(PETSC_DIR)/include -I$(CUDA_HOME)/include

LIBS = -L$(PETSC_DIR)/lib -lpetsc \
       -L$(CUDA_HOME)/lib64 -lcudart -lcusparse \
       -L$(MPI_HOME)/lib -lmpi -lmpi_cxx

SRC = $(wildcard src/*/*.cpp) $(wildcard src/*/*.cu)
OBJ = $(patsubst %.cu,%.o,$(patsubst %.cpp,%.o,$(SRC)))

sim: $(OBJ)
    $(CUDA_HOME)/bin/nvcc $(CUFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
    $(MPI_HOME)/bin/mpic++ $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
    $(CUDA_HOME)/bin/nvcc $(CUFLAGS) $(INCLUDES) -dc $< -o $@

clean:
    find src -name "*.o" -delete
    rm -f sim