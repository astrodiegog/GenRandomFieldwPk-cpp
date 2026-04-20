#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <time.h>

#include <mpi.h>
#include "hdf5.h"
#include "params.h"
#include "one_dimension.h"
#include "two_dimension.h"
#include "three_dimension.h"

int procID;

int main(int argc, char **argv)
{
	// Program info
	int nprocs;
	MPI_Status status_mpi;

	char *param_file;
    struct PS_Params ps_params;

	// Call MPI routines & set nprocs and nprocID
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);


	// Declare info for HDF5 file
	std::string FileName_prefix = "FFTWFun_out.h5.";
	std::string FileName_appendix, FileName;
	const char *FileName_C;

    hid_t file_id;
    hid_t grp_1D_id, grp_2D_id, grp_3D_id;
	herr_t status;

	std::uint_fast32_t global_seed = 123456;
	printf("waddup !\n");

#ifdef HOWDY
    printf(" --- HOWDY ! --- \n");
#endif //HOWDY

    if (argc < 2)
    {
        fprintf(stderr, "No param file ! boooo \n");
        return 0;
    }
    else
    {
        param_file = argv[1];
    }


    // Create & populate ps_params
    Parse_Params(param_file, &ps_params);
	printf("--- Rank %d : Creating a %d-D gaussian random field with %d cells along each dimension with length %.4f Mpc/h \n", 
				procID, ps_params.ndims, ps_params.Ng, ps_params.Lbox);

	// Make sure we're gucci, aka make sure ks is in range
	double kFund = 2. * M_PI / ps_params.Lbox;
	double kNyq = kFund * ps_params.Ng / 2.;
	double kmax = sqrt(ps_params.ndims) * kNyq;
	if ( (ps_params.ks < kFund) || (kNyq < ps_params.ks) ) {
		fprintf(stderr, "--- Rank %d : k-mode ks = %.4e defining value of As is outside range between (kFund,kNyq) = (%.4e , %.4e) \n ", procID, ps_params.ks, kFund, kNyq);
		return 0;
	}

	int mags;
	if ( fabs(ps_params.ns) > 32. / (log10(kmax / kFund)) ) {
		mags = (int) fabs(ps_params.ns) * (log10(kmax / kFund));
		fprintf(stderr, "--- Rank %d : spanning >%d orders of magnitude, expect round-off error \n ", procID, mags);
		return 0;
	}

	printf("--- Rank %d : As = %.4e, ks = %.4e, ns = %.4e \n", procID, ps_params.As, ps_params.ks, ps_params.ns);

	// Create file
	FileName_appendix = std::to_string(procID);
    FileName = FileName_prefix + FileName_appendix;
	FileName_C = FileName.c_str();
	printf("--- Rank %d : Creating file %s --- \n", procID, FileName_C);
    file_id = H5Fcreate(FileName_C, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	printf("--- Rank %d : Creating a %d-D gaussian random field with %d cells along each dimension with length %.4f Mpc/h \n",
                procID, ps_params.ndims, ps_params.Ng, ps_params.Lbox);
	if (ps_params.ndims == 1){
		// Create group for 1D
		grp_1D_id = H5Gcreate(file_id, "/OneDimension", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        run_one_dimension(global_seed, grp_1D_id, &ps_params);
		status = H5Gclose(grp_1D_id);
    }
	else if (ps_params.ndims == 2) {
		// Create group for 2D
		grp_2D_id = H5Gcreate(file_id, "/TwoDimension", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		run_two_dimension(global_seed, grp_2D_id, &ps_params);
		status = H5Gclose(grp_2D_id);
	}
	else if (ps_params.ndims == 3) {
		// Create group for 3D
		grp_3D_id = H5Gcreate(file_id, "/TwoDimension", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		run_three_dimension(global_seed, grp_3D_id, &ps_params);
		status = H5Gclose(grp_3D_id);
	}

	// Close HDF5 info
    status = H5Fclose(file_id);

	MPI_Finalize();

	return 0;
}

