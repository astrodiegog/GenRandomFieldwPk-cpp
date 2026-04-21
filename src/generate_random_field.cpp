#include "generate_random_field.h"

#define REQUEST 1
#define REPLY 2
#define CHUNKSIZE 128

extern void set_random_field(PS_Params *ps_params,  ptrdiff_t alloc_local, fftw_complex *xi_arr_local)
{
	int i;

    MPI_Status status_mpi;
    MPI_Comm world, rand_receivers;
    MPI_Group world_group, rand_receivers_group;
	int rand_receiver_id;
    int rand_generator;
    int ranks[1];
	int num_nrand_request;
	double rand_nums[CHUNKSIZE];

	world = MPI_COMM_WORLD;

	rand_generator = 0;
	std::mt19937 eng(ps_params->seed);
	float mean = 0.;
	float variance = pow(ps_params->Ng, ps_params->ndims);
	float stddev = sqrt(variance);
	std::normal_distribution<> normal_dist(mean, stddev);
	if (procID == rand_generator){
		for (i = 0; i < alloc_local ; i++)
		{
			xi_arr_local[i][0] = normal_dist(eng);
			xi_arr_local[i][1] = 0.;
		}
		printf("--- Rank %d : done generating random numbers \n", procID);
	}


	// populate world group
    MPI_Comm_group(world, &world_group);
    ranks[0] = rand_generator;

	// create group without random generator procID0
    MPI_Group_excl(world_group, 1, ranks, &rand_receivers_group);

	//create comm between random receivers group
    MPI_Comm_create(world, rand_receivers_group, &rand_receivers);
    MPI_Group_free(&rand_receivers_group);


	if (procID == rand_generator) {
		num_nrand_request = CHUNKSIZE;

		while (num_nrand_request > 0) {
			// grab the number of random numbers requested
			MPI_Recv(&num_nrand_request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, world, &status_mpi);

			//printf("--- Rank %d : received request for %d random numbers from %d \n", procID, num_nrand_request, status_mpi.MPI_SOURCE);
			// populate rands if requested
			if (num_nrand_request > 0) {
				for (i = 0; i < num_nrand_request; i++){
					rand_nums[i] = normal_dist(eng);
				}
				MPI_Send(&rand_nums, num_nrand_request, MPI_DOUBLE, status_mpi.MPI_SOURCE, REPLY, world);
				//printf("--- Rank %d : sending reply of %d random numbers to %d \n", procID, num_nrand_request, status_mpi.MPI_SOURCE);
			}
		}
	}
	else {
		// send request
		if (CHUNKSIZE < alloc_local) {
			num_nrand_request = CHUNKSIZE;
		}
		else {
			num_nrand_request = alloc_local;
		}
		MPI_Send(&num_nrand_request, 1, MPI_INT, rand_generator, REQUEST, world);
		MPI_Comm_rank(rand_receivers, &rand_receiver_id);

		int rand_nums_generated = 0;
		int indx;
		while (rand_nums_generated < alloc_local) {
			// receive random array from random_generator via world communicator
			MPI_Recv(&rand_nums, num_nrand_request, MPI_DOUBLE, rand_generator, REPLY, world, &status_mpi);

			for (i = 0; i < num_nrand_request; i++) {
				indx = rand_nums_generated + i;
				xi_arr_local[indx][0] = rand_nums[i];
				xi_arr_local[indx][1] = 0.;
			}
		
			rand_nums_generated += num_nrand_request;
			if (rand_nums_generated + CHUNKSIZE < alloc_local ) {
				num_nrand_request = CHUNKSIZE;
			}
			else {
				num_nrand_request = alloc_local - rand_nums_generated;
			}
			
			if (num_nrand_request > 0) {
				MPI_Send(&num_nrand_request, 1, MPI_INT, rand_generator, REQUEST, world);
			}

		}
		printf("--- Rank %d  : done requesting random numbers from Rank %d \n", procID, rand_generator);

		MPI_Barrier(rand_receivers);

		// have one receiver process tell rand generator to kick itself outside loop
		if (rand_receiver_id == 0) {
			num_nrand_request = 0;
			MPI_Send(&num_nrand_request, 1, MPI_INT, rand_generator, REQUEST, world);
			//printf("--- Rank %d : sending request for %d random numbers \n", procID, num_nrand_request);
		}

		MPI_Barrier(rand_receivers);
		MPI_Comm_free(&rand_receivers);

	}

	MPI_Barrier(world);
}


extern void set_real2D_random_field(PS_Params *ps_params,  ptrdiff_t local_n0, ptrdiff_t local_n1, double *xi_arr_local)
{
    int i, j, r, indx;
	int alloc_local;

    MPI_Status status_mpi;
    MPI_Comm world, rand_receivers;
    MPI_Group world_group, rand_receivers_group;
    int rand_receiver_id;
    int rand_generator;
    int ranks[1];
    int num_nrand_request;
    double rand_nums[CHUNKSIZE];

    world = MPI_COMM_WORLD;

    rand_generator = 0;
    std::mt19937 eng(ps_params->seed);
	float mean = 0.;
    float variance = pow(ps_params->Ng, ps_params->ndims);
    float stddev = sqrt(variance);
    std::normal_distribution<> normal_dist(mean, stddev);
    if (procID == rand_generator){
        for (i = 0; i < local_n0 ; i++) {
			for (j = 0; j < local_n1; j++) {
				indx = j + i * 2*(local_n1/2 + 1);
			 	xi_arr_local[indx] = normal_dist(eng);
			}
        }
        printf("--- Rank %d : done generating random numbers \n", procID);
    }

    // populate world group
    MPI_Comm_group(world, &world_group);
    ranks[0] = rand_generator;

    // create group without random generator procID0
	MPI_Group_excl(world_group, 1, ranks, &rand_receivers_group);

    //create comm between random receivers group
	MPI_Comm_create(world, rand_receivers_group, &rand_receivers);
    MPI_Group_free(&rand_receivers_group);


    if (procID == rand_generator) {
        num_nrand_request = CHUNKSIZE;

        while (num_nrand_request > 0) {
			// grab the number of random numbers requested
			MPI_Recv(&num_nrand_request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, world, &status_mpi);

			// populate rands if requested
			if (num_nrand_request > 0) {
                for (r = 0; r < num_nrand_request; r++){
                    rand_nums[r] = normal_dist(eng);
                }
                MPI_Send(&rand_nums, num_nrand_request, MPI_DOUBLE, status_mpi.MPI_SOURCE, REPLY, world);
			}
		}
    }
    else {
		alloc_local = local_n0 * local_n1;
        // send request
		if (CHUNKSIZE < alloc_local) {
            num_nrand_request = CHUNKSIZE;
        }
        else {
            num_nrand_request = alloc_local;
        }
        MPI_Send(&num_nrand_request, 1, MPI_INT, rand_generator, REQUEST, world);
        MPI_Comm_rank(rand_receivers, &rand_receiver_id);

        int rand_nums_generated = 0;
		i = 0;
		j = 0;
        int indx = 0;
        while (rand_nums_generated < alloc_local) {
            // receive random array from random_generator via world communicator

			MPI_Recv(&rand_nums, num_nrand_request, MPI_DOUBLE, rand_generator, REPLY, world, &status_mpi);

            for (r = 0; r < num_nrand_request; r++) {
                indx = j + i * 2*(local_n1/2 + 1);
				//printf("--- Rank %d : generate random numbers at (i,j)=(%d,%d) = %d -- %f\n", procID, i, j, indx, rand_nums[r]);
                xi_arr_local[indx] = rand_nums[r];

				j += 1;
				if (j >= local_n1) {
					j = 0;
					i += 1;
				}
            }

            rand_nums_generated += num_nrand_request;
            if (rand_nums_generated + CHUNKSIZE < alloc_local ) {
                num_nrand_request = CHUNKSIZE;
            }
            else {
                num_nrand_request = alloc_local - rand_nums_generated;
            }

            if (num_nrand_request > 0) {
                MPI_Send(&num_nrand_request, 1, MPI_INT, rand_generator, REQUEST, world);
            }

        }
        printf("--- Rank %d  : done requesting random numbers from Rank %d \n", procID, rand_generator);

        MPI_Barrier(rand_receivers);

        // have one receiver process tell rand generator to kick itself outside loop
		if (rand_receiver_id == 0) {
            num_nrand_request = 0;
            MPI_Send(&num_nrand_request, 1, MPI_INT, rand_generator, REQUEST, world);

		}

        MPI_Barrier(rand_receivers);
        MPI_Comm_free(&rand_receivers);

    }

    MPI_Barrier(world);
}



extern void set_real3D_random_field(PS_Params *ps_params,  ptrdiff_t local_n0, ptrdiff_t local_n1, ptrdiff_t  local_n2, double *xi_arr_local)
{
    int i, j, k, r, indx;
    int alloc_local;

    MPI_Status status_mpi;
    MPI_Comm world, rand_receivers;
    MPI_Group world_group, rand_receivers_group;
    int rand_receiver_id;
    int rand_generator;
    int ranks[1];
    int num_nrand_request;
    double rand_nums[CHUNKSIZE];

    world = MPI_COMM_WORLD;

    rand_generator = 0;
    std::mt19937 eng(ps_params->seed);
	float mean = 0.;
    float variance = pow(ps_params->Ng, ps_params->ndims);
    float stddev = sqrt(variance);
    std::normal_distribution<> normal_dist(mean, stddev);
    if (procID == rand_generator){
        for (i = 0; i < local_n0 ; i++) {
            for (j = 0; j < local_n1; j++) {
				for (k = 0; k < local_n2; k++) {
					indx = (j + i * local_n1) * (2*(local_n2/2 + 1)) + k;

	 				xi_arr_local[indx] = normal_dist(eng);
				}
            }
        }
        printf("--- Rank %d : done generating random numbers \n", procID);
    }

    // populate world group
    MPI_Comm_group(world, &world_group);
	ranks[0] = rand_generator;


	// create group without random generator procID0
	MPI_Group_excl(world_group, 1, ranks, &rand_receivers_group);

	//create comm between random receivers group
	MPI_Comm_create(world, rand_receivers_group, &rand_receivers);
	MPI_Group_free(&rand_receivers_group);


	if (procID == rand_generator) {
		num_nrand_request = CHUNKSIZE;
	
		while (num_nrand_request > 0) {
			// grab the number of random numbers requested
			MPI_Recv(&num_nrand_request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, world, &status_mpi);

			// populate rands if requested
			if (num_nrand_request > 0) {
                for (r = 0; r < num_nrand_request; r++){
                    rand_nums[r] = normal_dist(eng);
                }
                MPI_Send(&rand_nums, num_nrand_request, MPI_DOUBLE, status_mpi.MPI_SOURCE, REPLY, world);
            }
        }
    }
    else {
        alloc_local = local_n0 * local_n1 * local_n2;

		// send request
		if (CHUNKSIZE < alloc_local) {
            num_nrand_request = CHUNKSIZE;
        }
        else {
            num_nrand_request = alloc_local;
        }
        MPI_Send(&num_nrand_request, 1, MPI_INT, rand_generator, REQUEST, world);
        MPI_Comm_rank(rand_receivers, &rand_receiver_id);

        int rand_nums_generated = 0;
        i = 0;
        j = 0;
		k = 0;
        int indx = 0;
		while (rand_nums_generated < alloc_local) {
            // receive random array from random_generator via world communicator
			MPI_Recv(&rand_nums, num_nrand_request, MPI_DOUBLE, rand_generator, REPLY, world, &status_mpi);

            for (r = 0; r < num_nrand_request; r++) {
				indx = (j + i * local_n1) * (2*(local_n2/2 + 1)) + k;
				xi_arr_local[indx] = rand_nums[r];

				k += 1;
				if (k >= local_n2) { // completed k-loop
					k = 0;
					j += 1;
					if (j >= local_n1) { // completed j-loop
						j = 0;
						i += 1;
					}
				}
            }

			rand_nums_generated += num_nrand_request;
            if (rand_nums_generated + CHUNKSIZE < alloc_local ) {
                num_nrand_request = CHUNKSIZE;
            }
            else {
                num_nrand_request = alloc_local - rand_nums_generated;
            }
            
            if (num_nrand_request > 0) {
                MPI_Send(&num_nrand_request, 1, MPI_INT, rand_generator, REQUEST, world);
            }
        
        }
        printf("--- Rank %d  : done requesting random numbers from Rank %d \n", procID, rand_generator);

		MPI_Barrier(rand_receivers);

        // have one receiver process tell rand generator to kick itself outside loop
		if (rand_receiver_id == 0) {
            num_nrand_request = 0;
            MPI_Send(&num_nrand_request, 1, MPI_INT, rand_generator, REQUEST, world);

        }

        MPI_Barrier(rand_receivers);
        MPI_Comm_free(&rand_receivers);

    }

    MPI_Barrier(world);
}



