#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <time.h>

#include <mpi.h>
#include "hdf5.h"
#include "params.h"
#include "one_dimension.h"

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
    hid_t grp_1D_id;
    hid_t dataspace1D_id_c;
    hid_t attrs1D_id;
	herr_t status;

	// Declare array of dimensions
    hsize_t dims1D_c[1];
    hsize_t attrs1D[1];
    int Rank = 1;

	std::uint_fast32_t global_seed = 123456 + 654321;
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


	// Create file
	FileName_appendix = std::to_string(procID);
    FileName = FileName_prefix + FileName_appendix;
	FileName_C = FileName.c_str();
	printf("--- Rank %d : Creating file %s --- \n", procID, FileName_C);
    file_id = H5Fcreate(FileName_C, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);


	// Create group for 1D
    grp_1D_id = H5Gcreate(file_id, "/OneDimension", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	printf("--- Rank %d : Creating a %d-D gaussian random field with %d cells along each dimension with length %.4f Mpc/h \n",
                procID, ps_params.ndims, ps_params.Ng, ps_params.Lbox);

	if (ps_params.ndims == 1){
        run_one_dimension(global_seed, grp_1D_id, ps_params.Ng, ps_params.Lbox);
    }


	// Close HDF5 info
    status = H5Gclose(grp_1D_id);
    status = H5Fclose(file_id);

	MPI_Finalize();

	return 0;
}

