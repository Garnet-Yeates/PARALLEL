#ifndef MPI_HELPERS_H
#define MPI_HELPERS_H

#include <mpi.h>
#include <stdio.h>

void MPI_Setup(int* argc, char*** argv, int* numRanks, int* rank, int* len, int print) {
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Get_processor_name(hostname, len);
    if (print) {
        printf("Number of tasks = %d, My rank = %d, Running on %s\n", *numRanks, *rank, hostname);
    }
}

#endif  // MPI_HELPERS_H

