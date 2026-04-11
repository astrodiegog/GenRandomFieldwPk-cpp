#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <time.h>

#include <mpi.h>
#include "hdf5.h"
#include "params.h"


int main(int argc, char **argv)
{
	/* Program info */
	int nprocs, procID;
	MPI_Status status_mpi;
	
	char *param_file;
    struct PS_Params ps_params;

	/* Call MPI routines & set nprocs and nprocID*/
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);

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


    /* Create & populate ps_params*/
    Parse_Params(param_file, &ps_params);

	printf("--- Rank %d : Creating a %d-D gaussian random field with %d cells along each dimension with length %.4f Mpc/h \n", 
				procID, ps_params.ndims, ps_params.Ng, ps_params.Lbox);


	MPI_Finalize();

	return 0;
}

