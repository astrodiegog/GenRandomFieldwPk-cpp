#include "two_dimension.h"
#include "params.h"
#include "generate_random_field.h"


extern void run_two_dimension(int global_seed, hid_t grp_2D_id, PS_Params *ps_params)
{
	// Declare array of dimensions for datasets
    int Rank = 2;
    int Rank_binned = 1;
	hsize_t dims2D_r[Rank], dims2D_r_input[Rank], dims_binned[Rank_binned];

	// Declare dataspaced for datasets
	hid_t dataspace2D_id_local_r, dataspace2D_id_local_r_input, dataspace_id_binned;

	// Power spectra values
	double dx, dy, kx2, ky2, variance;
	double dx_sample, dy_sample;
    double l_kmag, l_Pk;
    double l_ks, l_As;

	int i, i_global, j, indx, nkbins, indx_bin;

	// Declare fftw info
	ptrdiff_t N0, N1, N1_r2c, N1_r_buff;
    fftw_plan plan_FFT_r2c, plan_iFFT_c2r, plan_FFT_calc_r2c;
	ptrdiff_t alloc_local_r, local_n_r, local_n0_start_r;

	//Declare all FFT-related arrays
    double *Pk_input_local, *Tk2_input_local, *Pk_calc_local;
	double *kx_local, *ky_local, *kmag_local;
	double *xi_local;
    fftw_complex *xi_k_r2c_local, *delta_k_r2c_local, *delta_k_calc_r2c_local;
	double *delta_x_c2r_local;

	// Declare binning arrays
	long int *ikbin_local;
	long int *counts_local, *counts_global;
	double *Pk_bin_local_sum, *Pk_bin_local_avg, *Pk_bin_global;
	double *k_bin_local_sum, *k_bin_local_avg, *k_bin_global;

	// Grab the amount of data allocated by local_size routines
    N0 = ps_params->Ng;
	N1 = ps_params->Ng;
	N1_r2c = N1/2 + 1;
	nkbins = ceil( sqrt( (N0/2) * (N0/2) + (N1/2) * (N1/2) ) );

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
	dims_binned[0] = nkbins;
	dataspace_id_binned = H5Screate_simple(Rank_binned, dims_binned, NULL);

	// Allocate memory
    Pk_input_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);
	Tk2_input_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

    xi_local = (double *) fftw_malloc(sizeof(double) * 2 * alloc_local_r);

	kx_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);
	ky_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);
	kmag_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

    xi_k_r2c_local = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local_r);
    delta_k_r2c_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local_r);

    delta_x_c2r_local = (double *) fftw_malloc(sizeof(double) * 2 * alloc_local_r);

	delta_k_calc_r2c_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local_r);
	Pk_calc_local = (double *) fftw_malloc(sizeof(double) * alloc_local_r);

	ikbin_local = (long int *) fftw_malloc(sizeof(long int) * alloc_local_r);
	counts_local = (long int *) fftw_malloc(sizeof(long int) * nkbins);
	counts_global = (long int *) fftw_malloc(sizeof(long int) * nkbins);
	Pk_bin_local_sum = (double *) fftw_malloc(sizeof(double) * nkbins);
	Pk_bin_local_avg = (double *) fftw_malloc(sizeof(double) * nkbins);
	Pk_bin_global = (double *) fftw_malloc(sizeof(double) * nkbins);
	k_bin_local_sum = (double *) fftw_malloc(sizeof(double) * nkbins);
	k_bin_local_avg = (double *) fftw_malloc(sizeof(double) * nkbins);
	k_bin_global = (double *) fftw_malloc(sizeof(double) * nkbins);

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
	dx_sample = dx / (2. * M_PI);
	dy_sample = dy / (2. * M_PI);
	l_ks = log10(ps_params->ks);
	l_As = log10(ps_params->As);

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

			//ikbin_local[indx] = floor(sqrt( i_global*i_global + j*j ));

			kx2 = kx_local[indx] * kx_local[indx];
			ky2 = ky_local[indx] * ky_local[indx];
			kmag_local[indx] = sqrt(kx2 + ky2);
			if (kmag_local[indx] == 0) {
				// Guard against logging zero
				Pk_input_local[indx] = 1.e-16;
			}
			else {
				l_kmag = log10(kmag_local[indx]);
				l_Pk = l_As + (ps_params->ns) * (l_kmag - l_ks);
				Pk_input_local[indx] = pow(10, l_Pk);
			}

			Tk2_input_local[indx] = pow((2. * M_PI / ps_params->Lbox), ps_params->ndims) * Pk_input_local[indx];
			//printf("--- Rank %d : T^2(k=%.4e)=%.4e \n", procID, kmag, Tk2_input_local[indx]);			
		}
	}



	// Write k , P(k) array
	Write_HDF5_dataset(grp_2D_id, "kx_local", dataspace2D_id_local_r, &kx_local[0]);
	Write_HDF5_dataset(grp_2D_id, "ky_local", dataspace2D_id_local_r, &ky_local[0]);
	Write_HDF5_dataset(grp_2D_id, "kmag_local", dataspace2D_id_local_r, &kmag_local[0]);
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
	printf("--- Rank %d : binning P(k) into %d bins of deltak = kfund = %.4e \n", procID, nkbins, (2. * M_PI) / ps_params->Lbox);
	for (i = 0; i < local_n_r; i++) {
        for (j = 0; j < N1_r2c; j++) {
            indx = j + i *  N1_r2c;

			// Assigning kmodes assumes even number of local_n_c
			if ( (int) (i + local_n0_start_r) > (int) ( (N0/2) - 1) ) {
                // Negative freqs
				i_global = -( N0 - (i + local_n0_start_r));
            }
            else {
                // Positive feqs
				i_global = (i + local_n0_start_r);
            }

			ikbin_local[indx] = floor(sqrt( i_global*i_global + j*j ));

			indx_bin = ikbin_local[indx]; // index within bins

			counts_local[ indx_bin ] += 1;
			k_bin_local_sum[ indx_bin ] += kmag_local[indx];
			Pk_bin_local_sum[ indx_bin ] += Pk_calc_local[indx];
        }
    }

	// Define averaged binned k, P(k) values in this process
	for (i = 0; i < nkbins; i++) {
		if (counts_local[i] == 0) { // Avoid dividing by zero
			k_bin_local_avg[i] = k_bin_local_sum[i];
            Pk_bin_local_avg[i] = Pk_bin_local_sum[i] ;
		}
		else {
			k_bin_local_avg[i] = k_bin_local_sum[i] / counts_local[i];
			Pk_bin_local_avg[i] = Pk_bin_local_sum[i] / counts_local[i];
		}
	}


	// Write local bin info
	Write_HDF5_longint_dataset(grp_2D_id, "ikbin_local", dataspace2D_id_local_r, &ikbin_local[0]);
	Write_HDF5_longint_dataset(grp_2D_id, "counts_local", dataspace_id_binned, &counts_local[0]);
	Write_HDF5_dataset(grp_2D_id, "Pk_bin_local", dataspace_id_binned, &Pk_bin_local_avg[0]);
	Write_HDF5_dataset(grp_2D_id, "k_bin_local", dataspace_id_binned, &k_bin_local_avg[0]);

	// Reduce global counts, Reduce k&P(k), Normalize by count
	printf("--- Rank %d : MPI-AllReducing %d bins \n", procID, nkbins);
	MPI_Allreduce(&counts_local[0], &counts_global[0], nkbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&k_bin_local_sum[0], &k_bin_global[0], nkbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	 MPI_Allreduce(&Pk_bin_local_sum[0], &Pk_bin_global[0], nkbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	for (i = 0; i < nkbins; i++) {
        if (counts_global[i] != 0) { // Avoid dividing by zero
            k_bin_global[i] /= counts_global[i];
            Pk_bin_global[i] /= counts_global[i];
        }
    }

	Write_HDF5_longint_dataset(grp_2D_id, "counts_global", dataspace_id_binned, &counts_global[0]);
	Write_HDF5_dataset(grp_2D_id, "k_bin_global", dataspace_id_binned, &k_bin_global[0]);
	Write_HDF5_dataset(grp_2D_id, "Pk_bin_global", dataspace_id_binned, &Pk_bin_global[0]);
}


