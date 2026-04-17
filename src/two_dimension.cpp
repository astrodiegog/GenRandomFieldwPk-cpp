#include "two_dimension.h"
#include "params.h"
#include "generate_random_field.h"

extern void run_two_dimension(int global_seed, hid_t grp_2D_id, PS_Params *ps_params)
{
	// Declare array of dimensions for datasets
    hsize_t dims2D_r[2], dims2D_r_input[2];
    int Rank = 2;

	// Power spectra values
	double Lbox = ps_params->Lbox;
	double dx, dy, kmag, variance;

	int i;

	// Declare fftw info
	ptrdiff_t N0, N1, N1_r2c;
    fftw_plan plan_FFT_r2c, plan_iFFT_c2r, plan_FFT_calc_r2c;
	ptrdiff_t alloc_local_r, local_n_r, local_n0_start_r;

	//Declare all FFT-related arrays
    double *Pk_input_local, *Tk2_input_local, *Pk_calc_local;
	double *kx_local, *ky_local;
	double *xi_local;
    fftw_complex *xi_k_r2c_local, *delta_k_r2c_local, *delta_k_calc_r2c_local;
	double *delta_x_c2r_local;
    hid_t dataspace2D_id_local_r, dataspace2D_id_local_r_input;


	// Grab the amount of data allocated by local_size routines
    N0 = ps_params->Ng;
	N1 = ps_params->Ng;
	N1_r2c = N1/2 + 1;

	alloc_local_r = fftw_mpi_local_size_2d(N0, N1_r2c, MPI_COMM_WORLD,
										   &local_n_r, &local_n0_start_r);


	// Create in/out dataspaces for FFT/iFFT
	dims2D_r[0] = local_n_r;
	dims2D_r[1] = N1_r2c;
	dataspace2D_id_local_r = H5Screate_simple(Rank, dims2D_r, NULL);

	dims2D_r_input[0] = local_n_r;
	dims2D_r_input[1] = 2 * N1_r2c;
	dataspace2D_id_local_r_input = H5Screate_simple(Rank, dims2D_r_input, NULL);
	

	// Allocate memory
    Pk_input_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);
	Tk2_input_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

    xi_local = (double *) fftw_malloc(sizeof(double) * 2 * alloc_local_r);

	kx_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);
	ky_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

    xi_k_r2c_local = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local_r);
    delta_k_r2c_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local_r);

    delta_x_c2r_local = (double *) fftw_malloc(sizeof(fftw_complex) * 2 * alloc_local_r);

	delta_k_calc_r2c_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local_r);
	Pk_calc_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

	// Create plans
	plan_FFT_r2c = fftw_mpi_plan_dft_r2c_2d(N0, N1, xi_local, xi_k_r2c_local, 
											MPI_COMM_WORLD, FFTW_ESTIMATE);

	plan_iFFT_c2r = fftw_mpi_plan_dft_c2r_2d(N0, N1, xi_k_r2c_local, delta_x_c2r_local, 
											 MPI_COMM_WORLD, FFTW_ESTIMATE);

	plan_FFT_calc_r2c = fftw_mpi_plan_dft_r2c_2d(N0, N1, delta_x_c2r_local, delta_k_calc_r2c_local,
												 MPI_COMM_WORLD, FFTW_ESTIMATE);
	

	/*
	// Fill in k, P(k), T^2(k) info
	dx = ps_params->Lbox / ps_params->Ng;
	double dx_sample = dx / (2. * M_PI);
	double l_kmag, l_Pk;
	double l_ks = log10(ps_params->ks);
	double l_As = log10(ps_params->As);
    for (i = 0; i < local_no_FFT; i++) {
        // Assigning kmodes assumes even number of local_ni 
        if ( (int) (i + local_i_start_FFT) > (int) ((N0 / 2) - 1) ) {
            // Negative frequencies 
            kx_local[i] = -( N0 - (i + local_o_start_FFT)) / (dx_sample * ps_params->Ng);
        }
        else {
            // Positive frequencies
            kx_local[i] = (i + local_o_start_FFT) / (dx_sample * ps_params->Ng);
        }

		kmag = sqrt(kx_local[i] * kx_local[i]);
		
		if (kmag == 0) {
			Pk_input_local[i] = 1.e-16;
		}
		else {
			l_kmag = log10(kmag);
			l_Pk = l_As + (ps_params->ns) * (l_kmag - l_ks);
			Pk_input_local[i] = pow(10, l_Pk);
		}
		
		
		//printf("--- Rank %d : P(k=%.4e)=%.4e \n", procID, kmag, Pk_input_local[i]);
		Tk2_input_local[i] = pow((2. * M_PI / Lbox), ps_params->ndims) * Pk_input_local[i];
		//printf("--- Rank %d : T^2(k=%.4e)=%.4e \n", procID, kmag, Tk2_input_local[i]);
    }

	// Write k , P(k) array
	Write_HDF5_dataset(grp_1D_id, "kx_local", dataspace1D_id_local_in_c_FFT, &kx_local[0]);
	Write_HDF5_dataset(grp_1D_id, "Pk_input_local", dataspace1D_id_local_in_c_FFT, &Pk_input_local[0]);

	// Step 1 : Create xi - random field
	set_random_field(global_seed, ps_params, alloc_local_FFT, &xi_local[0]);

	// Write xi - random field
	Write_FFTWarr_1Dgroup(grp_1D_id, "xi_local", dataspace1D_id_local_in_c_FFT, &xi_local[0], local_ni_FFT);

	// Step 2 : Take FFT of xi --> populate xi_k & normalize
	fftw_execute(plan_FFT_c2c);
	variance = pow(ps_params->Ng, ps_params->ndims);
	printf("--- Rank %d : Normalizing xi(k) with variance %f \n", procID, variance);
	for (i = 0; i < local_no_FFT; i++) {
        xi_k_c2c_local[i][0] = xi_k_c2c_local[i][0] / variance;
        xi_k_c2c_local[i][1] = xi_k_c2c_local[i][1] / variance;
    }

	//Write_FFTWarr_1Dgroup(grp_1D_id, "xi_k_local", dataspace1D_id_local_out_c_FFT, &xi_k_c2c_local[0], local_no_FFT);

	// Step 3 : Apply Transfer Function
	for (i = 0; i < local_no_FFT; i++) {
		delta_k_c2c_local[i][0] = xi_k_c2c_local[i][0] * sqrt(Tk2_input_local[i]);
		delta_k_c2c_local[i][1] = xi_k_c2c_local[i][1] * sqrt(Tk2_input_local[i]);
	}

	// Write delta_k - power spectrum applied to noise in k-space
	//Write_FFTWarr_1Dgroup(grp_1D_id, "deltak_local", dataspace1D_id_local_out_c_iFFT, &delta_k_c2c_local[0], local_no_FFT);

	// Step 4 : Take iFFT of delta_k --> evaluate delta(m), scale by (1/N)
	fftw_execute(plan_iFFT_c2c);
	for (i = 0; i < local_ni_FFT; i++) {
        delta_x_c2c_local[i][0] = delta_x_c2c_local[i][0] / ps_params->Ng;
        delta_x_c2c_local[i][1] = delta_x_c2c_local[i][1] / ps_params->Ng;
    }
	
	// Write delta_x - power spectrum applied to noise
	Write_FFTWarr_1Dgroup(grp_1D_id, "deltax_local", dataspace1D_id_local_in_c_iFFT, &delta_x_c2c_local[0], local_ni_FFT);

	// Reconstruct P(k) from delta_x
	fftw_execute(plan_FFT_calc_c2c);
	double Tk2_calc;
	for (i = 0; i< local_no_FFT; i++) {
		Tk2_calc = delta_k_calc_c2c_local[i][0] * delta_k_calc_c2c_local[i][0] + delta_k_calc_c2c_local[i][1] * delta_k_calc_c2c_local[i][1];
		Pk_calc_local[i] = Tk2_calc / pow((2. * M_PI / ps_params->Lbox), ps_params->ndims);
	}

	// Write P(k)
	Write_HDF5_dataset(grp_1D_id, "Pk_calc_local", dataspace1D_id_local_out_c_FFT, &Pk_calc_local[0]);
	*/

}


