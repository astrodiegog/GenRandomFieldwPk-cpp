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
	double dx, dy, kx2, ky2, kmag, variance;

	int i, i_global, j, indx, nkbins;

	// Declare fftw info
	ptrdiff_t N0, N1, N1_r2c, N1_r_buff;
    fftw_plan plan_FFT_r2c, plan_iFFT_c2r, plan_FFT_calc_r2c;
	ptrdiff_t alloc_local_r, local_n_r, local_n0_start_r;

	//Declare all FFT-related arrays
    double *Pk_input_local, *Tk2_input_local, *Pk_calc_local;
	double *kx_local, *ky_local;
	double *xi_local;
    fftw_complex *xi_k_r2c_local, *delta_k_r2c_local, *delta_k_calc_r2c_local;
	double *delta_x_c2r_local;
    hid_t dataspace2D_id_local_r, dataspace2D_id_local_r_input;

	// Declare binning arrays
	long int *ikbin_local;
	long int *counts_global;
	double *Pk_bin_global;
	double *k_bin_global;

	// Grab the amount of data allocated by local_size routines
    N0 = ps_params->Ng;
	N1 = ps_params->Ng;
	N1_r2c = N1/2 + 1;
	nkbins = ceil( sqrt( (N0/2) * (N0/2) + (N1/2) * (N1/2) );

	printf("--- Rank %d : N0 %ld N1 %ld N1_r2c %ld \n", procID, N0, N1, N1_r2c);
	alloc_local_r = fftw_mpi_local_size_2d(N0, N1_r2c, MPI_COMM_WORLD,
										   &local_n_r, &local_n0_start_r);

	N1_r_buff = (2 * alloc_local_r / local_n_r) - N1;
	printf("--- Rank %d : responsible for (%ld,%ld) section with local allocation of %ld = (%ld, %ld) real numbers or %ld = (%ld, %ld) complex \n", 
					procID, 
					local_n_r, N1, 2 * alloc_local_r, local_n_r, N1_r_buff + N1, 
					alloc_local_r, local_n_r, (alloc_local_r / local_n_r));

	// Create in/out dataspaces for FFT/iFFT
	dims2D_r[0] = local_n_r;
	dims2D_r[1] = N1_r2c;
	dataspace2D_id_local_r = H5Screate_simple(Rank, dims2D_r, NULL);
	
	dims2D_r_input[0] = local_n_r;
	dims2D_r_input[1] = 2 * N1_r2c;
	dataspace2D_id_local_r_input = H5Screate_simple(Rank, dims2D_r_input, NULL);
	
	// Create dataspace for binned P(k)
	dims_binned[0] = counts_gl;
	dataspace_id_binned = H5Screate_simple(Rank, dims_binned, NULL);

	// Allocate memory
    Pk_input_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);
	Tk2_input_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

    xi_local = (double *) fftw_malloc(sizeof(double) * 2 * alloc_local_r);

	kx_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);
	ky_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

    xi_k_r2c_local = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local_r);
    delta_k_r2c_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local_r);

    delta_x_c2r_local = (double *) fftw_malloc(sizeof(double) * 2 * alloc_local_r);

	delta_k_calc_r2c_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local_r);
	Pk_calc_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

	ikbin_local = (long int *) fftw_malloc(sizeof(long int) * alloc_local_r);
	counts_global = (long int *) fftw_malloc(sizeof(long int) * alloc_local_r);
	Pk_bin_global = (double *) fftw_malloc(sizeof(double) * alloc_local_r);
	k_bin_global = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

	// Create plans
	plan_FFT_r2c = fftw_mpi_plan_dft_r2c_2d(N0, N1, xi_local, xi_k_r2c_local, 
											MPI_COMM_WORLD, FFTW_ESTIMATE);

	plan_iFFT_c2r = fftw_mpi_plan_dft_c2r_2d(N0, N1, delta_k_r2c_local, delta_x_c2r_local, 
											 MPI_COMM_WORLD, FFTW_ESTIMATE);

	plan_FFT_calc_r2c = fftw_mpi_plan_dft_r2c_2d(N0, N1, delta_x_c2r_local, delta_k_calc_r2c_local,
												 MPI_COMM_WORLD, FFTW_ESTIMATE);
	

	// Fill in k, P(k), T^2(k) info
	dx = ps_params->Lbox / ps_params->Ng;
	dy = ps_params->Lbox / ps_params->Ng;
	double dx_sample = dx / (2. * M_PI);
	double dy_sample = dy / (2. * M_PI);
	double l_kmag, l_Pk;
	double l_ks = log10(ps_params->ks);
	double l_As = log10(ps_params->As);

	for (i = 0; i < local_n_r; i++) {
		for (j = 0; j < N1_r2c; j++) {
			indx = j + i * N1_r2c;
			
			// Assigning kmodes assumes even number of local_n_c
			if ( (int) (i + local_n0_start_r) > (int) ( (N0/2) - 1) ) {
				// Negative freqs
				i_global = -( N0 - (i + local_n0_start_r));
			}
			else {
				// Positive feqs
				i_global = (i + local_n0_start_r);
			}

			kx_local[indx] = i_global / (dx_sample * N0);

			// Positive freqs
			ky_local[indx] = j / (dy_sample * N1);

			ikbin_local[indx] = floor(sqrt( i_global*i_global + j*j ));

			kx2 = kx_local[indx] * kx_local[indx];
			ky2 = ky_local[indx] * ky_local[indx];
			kmag = sqrt(kx2 + ky2);
			if (kmag == 0) {
				// Guard against logging zero
				Pk_input_local[indx] = 1.e-16;
			}
			else {
				l_kmag = log10(kmag);
				l_Pk = l_As + (ps_params->ns) * (l_kmag - l_ks);
				Pk_input_local[indx] = pow(10, l_Pk);
			}

			Tk2_input_local[indx] = pow((2. * M_PI / Lbox), ps_params->ndims) * Pk_input_local[indx];
			//printf("--- Rank %d : T^2(k=%.4e)=%.4e \n", procID, kmag, Tk2_input_local[indx]);			
		}
	}



	// Write k , P(k) array
	Write_HDF5_longint_dataset(grp_2D_id, "ikbin_local", dataspace2D_id_local_r, &ikbin_local[0]);
	Write_HDF5_dataset(grp_2D_id, "kx_local", dataspace2D_id_local_r, &kx_local[0]);
	Write_HDF5_dataset(grp_2D_id, "ky_local", dataspace2D_id_local_r, &ky_local[0]);
	Write_HDF5_dataset(grp_2D_id, "Pk_input_local", dataspace2D_id_local_r, &Pk_input_local[0]);

	// Step 1 : Create xi - random field
	printf("--- Rank %d : Requesting %ld random numbers \n", procID, local_n_r * N1_r2c);
	set_real2D_random_field(global_seed, ps_params, local_n_r, N1, &xi_local[0]);

	// Write xi - random field
	Write_HDF5_dataset(grp_2D_id, "xi_local", dataspace2D_id_local_r_input, &xi_local[0]);

	// Step 2 : Take FFT of xi --> populate xi_k & normalize
	fftw_execute(plan_FFT_r2c);
	variance = pow(ps_params->Ng, ps_params->ndims);
	printf("--- Rank %d : Normalizing xi(k) with variance %f \n", procID, variance);
	for (i = 0; i < local_n_r; i++) {
		for (j = 0; j < N1_r2c; j++) {
			indx = j + i *  N1_r2c;
			xi_k_r2c_local[indx][0] = xi_k_r2c_local[indx][0] / variance;
			xi_k_r2c_local[indx][1] = xi_k_r2c_local[indx][1] / variance;
		}
    }

	Write_FFTWarr_2Dgroup(grp_2D_id, "xi_k_local", dataspace2D_id_local_r, &xi_k_r2c_local[0], local_n_r, N1_r2c);

	// Step 3 : Apply Transfer Function
	for (i = 0; i < local_n_r; i++) {
		for (j = 0; j < N1_r2c; j++) {
			indx = j + i *  N1_r2c;
			delta_k_r2c_local[indx][0] = xi_k_r2c_local[indx][0] * sqrt(Tk2_input_local[indx]);
			delta_k_r2c_local[indx][1] = xi_k_r2c_local[indx][1] * sqrt(Tk2_input_local[indx]);
		}
	}

	// Write delta_k - power spectrum applied to noise in k-space
	Write_FFTWarr_2Dgroup(grp_2D_id, "deltak_local", dataspace2D_id_local_r, &delta_k_r2c_local[0], local_n_r, N1_r2c);

	
	// Step 4 : Take iFFT of delta_k --> evaluate delta(m), scale by (1/N)
	fftw_execute(plan_iFFT_c2r);
	for (i = 0; i < local_n_r; i++) {
 		for (j = 0; j < N1; j++) {
			indx = j + i * 2*N1_r2c;
			delta_x_c2r_local[indx] = delta_x_c2r_local[indx] / (N0 * N1);
		}
	}
	
	// Write delta_x - power spectrum applied to noise
	Write_HDF5_dataset(grp_2D_id, "deltax_local", dataspace2D_id_local_r_input, &delta_x_c2r_local[0]);

	// Reconstruct P(k) from delta_x
	fftw_execute(plan_FFT_calc_r2c);
	double Tk2_calc;
	for (i = 0; i < local_n_r; i++) {
        for (j = 0; j < N1_r2c; j++) {
            indx = j + i *  N1_r2c;
			Tk2_calc = delta_k_calc_r2c_local[indx][0] * delta_k_calc_r2c_local[indx][0] + delta_k_calc_r2c_local[indx][1] * delta_k_calc_r2c_local[indx][1];
      		Pk_calc_local[indx] = Tk2_calc / pow((2. * M_PI / ps_params->Lbox), ps_params->ndims);      
        }
    }

	// Write P(k)
	Write_HDF5_dataset(grp_2D_id, "Pk_calc_local", dataspace2D_id_local_r, &Pk_calc_local[0]);

	// Bin P(k) by fundamental mode (ikbins_local)
	//counts_global \ Pk_bin_global \ k_bin_global
	
	for (i = 0; i < local_n_r; i++) {
        for (j = 0; j < N1_r2c; j++) {
            indx = j + i *  N1_r2c;

			counts[ ikbin_local[indx] ] += 1;
        }
    }

}


