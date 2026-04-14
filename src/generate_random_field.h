#include <random>

#include <fftw3-mpi.h>
#include "hdf5.h"

#include "params.h"

extern int procID;

extern void set_random_field(int global_seed, PS_Params *ps_params, ptrdiff_t alloc_local, fftw_complex *xi_arr_local);

