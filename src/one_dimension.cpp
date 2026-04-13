#include "one_dimension.h"
#include "generate_random_field.h"

extern void run_one_dimension(int global_seed, hid_t grp_1D_id, int Nx, double Lbox)
{
	// Declare array of dimensions for datasets
    hsize_t dims1D_c[1];
    int Rank = 1;

	// Declare fftw info
	ptrdiff_t N0;
    fftw_plan plan_FFT_c2c, plan_iFFT_c2c;
    ptrdiff_t alloc_local_FFT, local_ni_FFT, local_i_start_FFT, local_no_FFT, local_o_start_FFT;
    ptrdiff_t alloc_local_iFFT, local_ni_iFFT, local_i_start_iFFT, local_no_iFFT, local_o_start_iFFT;

	//Declare all FFT-related arrays
    double *Pk_arr_local;
    fftw_complex *xi_arr_local, *xi_k_arr_c2c_local, *delta_k_c2c_local, *delta_x_c2c_local;
    hid_t dataspace1D_id_local_in_c_FFT, dataspace1D_id_local_out_c_FFT;
    hid_t dataspace1D_id_local_in_c_iFFT, dataspace1D_id_local_out_c_iFFT;


	// Grab the amount of data allocated by local_size routines
    N0 = Nx;

    alloc_local_FFT = fftw_mpi_local_size_1d(N0, MPI_COMM_WORLD,
                                    FFTW_FORWARD, FFTW_ESTIMATE,
                                    &local_ni_FFT, &local_i_start_FFT,
                                    &local_no_FFT, &local_o_start_FFT);

    alloc_local_iFFT = fftw_mpi_local_size_1d(N0, MPI_COMM_WORLD,
                                    FFTW_BACKWARD, FFTW_ESTIMATE,
                                    &local_ni_iFFT, &local_i_start_iFFT,
                                    &local_no_iFFT, &local_o_start_iFFT);


	// Create in/out dataspaces for FFT/iFFT
	dims1D_c[0] = local_ni_FFT;
    dataspace1D_id_local_in_c_FFT = H5Screate_simple(Rank, dims1D_c, NULL);
    dims1D_c[0] = local_no_FFT;
    dataspace1D_id_local_out_c_FFT = H5Screate_simple(Rank, dims1D_c, NULL);

    dims1D_c[0] = local_ni_iFFT;
    dataspace1D_id_local_in_c_iFFT = H5Screate_simple(Rank, dims1D_c, NULL);
    dims1D_c[0] = local_no_iFFT;
    dataspace1D_id_local_out_c_iFFT = H5Screate_simple(Rank, dims1D_c, NULL);


	// Allocate memory
    Pk_arr_local = (double *) fftw_malloc(sizeof(double) * alloc_local_FFT);
    xi_arr_local = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local_FFT);
    xi_k_arr_c2c_local = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local_FFT);
    delta_k_c2c_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local_FFT);
    delta_x_c2c_local = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local_iFFT);

	// Create plans
    plan_FFT_c2c = fftw_mpi_plan_dft_1d(N0, xi_arr_local, xi_k_arr_c2c_local, MPI_COMM_WORLD,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
    plan_iFFT_c2c = fftw_mpi_plan_dft_1d(N0, delta_k_c2c_local, delta_x_c2c_local, MPI_COMM_WORLD,
                                         FFTW_BACKWARD, FFTW_ESTIMATE);

	// Create xi - random field
	set_random_field_oneD(global_seed, Nx, alloc_local_FFT, &xi_arr_local[0]);
	printf("ayo\n");

	// Write xi - random field
	Write_FFTWarr_1Dgroup(grp_1D_id, "xi_arr_local_FFT", dataspace1D_id_local_in_c_FFT, &xi_arr_local[0], local_ni_FFT);

}
