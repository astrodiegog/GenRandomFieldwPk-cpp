#include <random>

#include <fftw3-mpi.h>
#include "hdf5.h"


extern void set_random_field_oneD(int global_seed, int Nx, ptrdiff_t alloc_local, fftw_complex *xi_arr_local);

