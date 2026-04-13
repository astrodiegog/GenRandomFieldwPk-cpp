#include "generate_random_field.h"

#define CHUNKSIZE 64

extern void set_random_field_oneD(int global_seed, int Nx, ptrdiff_t alloc_local, fftw_complex *xi_arr_local)
{
	int i;

	/* 
    MPI_Status status_mpi;
    MPI_Comm world, rand_receivers;
    MPI_Group world_group, rand_receivers_group;
    int rand_generator;
    int ranks[1];
	*/

	std::mt19937 eng(global_seed);
    float mean = 0.;
    float stddev = Nx;
    std::normal_distribution<> normal_dist(mean, stddev);

	for (i = 0; i < alloc_local ; i++)
	{
 		xi_arr_local[i][0] = normal_dist(eng);
		xi_arr_local[i][1] = 0.;
	}

	/*
	// set procID0 as rand generator
    rand_generator = 0;

	if (procID == rand_generator){
        for (i = 0; i < Nx_local ; i++)
        {
            x_arr_local[i] = normal_dist(eng);
        }
    }

	// populate world group
    MPI_Comm_group(world, &world_group);
    ranks[0] = rand_generator;

	// create group without random generator procID0
    MPI_Group_excl(world_group, 1, ranks, &random_receivers_group);

	//create comm between random receivers group
    MPI_Comm_create(world, random_receivers_group, &rand_receivers);
    MPI_Group_free(&random_receivers_group);
	*/
}
