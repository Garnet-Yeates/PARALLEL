#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include "mpi_helpers.h"

int isPrime(int number);
int withLoadBalancing(int argc, char** argv, int N, int chunkSize);
int noLoadBalancing(int argc, char** argv, int N);

int main(int argc, char** argv) {

    if (argc < 2) {
        printf("Error: there must be at least 1 argument\n");
        return 1;
    }
    else if (argc > 3) {
        printf("Error: there can not be more than 2 arguments\n");
        return 1;
    }

    int N = atoi(argv[1]);

    // The user supplying a second argument implies that balancing is enabled
    if (argc == 3) {
        int chunkSize = atoi(argv[2]);
        return withLoadBalancing(argc, argv, N, chunkSize);
    }

    return noLoadBalancing(argc, argv, N);
}

int withLoadBalancing(int argc, char** argv, int N, int chunkSize) {

    // Set up MPI
    int numRanks, myRank, len, master = 0, tag = 0;
    MPI_Setup(&argc, &argv, &numRanks, &myRank, &len, 0);

    if (numRanks < 2) {
        printf("numRanks must be at least 2\n");
        exit(1);
    }

    int totalPrimes = 0;

    double startTime = MPI_Wtime();

    if (myRank == master) {

        int currStart = 0;
        int currEnd = currStart + chunkSize - 1; // End is inclusive btw, not exclusive remember

        printf("N = %d\n", N);
        printf("Num Ranks: %d\n", numRanks);
        printf("Running with load balancing enabled\n");
        printf("Chunk Size = %d\n", chunkSize);

        while (currStart <= N) {

            // Send a small chunk to each worker
            for (int worker = 1; worker < numRanks; worker++, currStart += chunkSize, currEnd += chunkSize) {

                int sendingStart = currStart, sendingEnd = currEnd;

                // No more work, send the 'do nothing' signal
                if (currStart > N) {
                    sendingStart = -1;
                    sendingEnd = -1;
                }
                else if (currEnd > N) {
                    sendingEnd = N;
                }

                MPI_Send(&sendingStart, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
                MPI_Send(&sendingEnd, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            }

            // Wait for recv results from everyone
            for (int worker = 1; worker < numRanks; worker++) {

                int primesInChunk;
                MPI_Recv(&primesInChunk, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                totalPrimes += primesInChunk;
            }
        }

        // Send the termination signal to all nodes now that we are outside of the while here
        for (int worker = 1; worker < numRanks; worker++) {
            int sending = -2;
            MPI_Send(&sending, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            MPI_Send(&sending, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
        }
    }
    else {
        while (1) {
            // Receive the range from the master
            int myStart, myEnd;
            MPI_Recv(&myStart, 1, MPI_INT, master, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&myEnd, 1, MPI_INT, master, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Immediately break if we get the termination signal (no need to send anything)
            if (myStart == -2)
                break;

            // Compute the number of primes in this chunk
            int primesInChunk = 0;
            if (myStart != -1) { // If myStart is -1 it basically means do nothing this rotation
                for (int num = myStart; num <= myEnd; num++) {
                    if (isPrime(num))
                        primesInChunk++;
                }
            }

            // Send the number of primes in this chunk
            MPI_Send(&primesInChunk, 1, MPI_INT, master, tag, MPI_COMM_WORLD);
        }
    }

    if (myRank == master) {
        printf("The total number of primes up to %d is %d\n", N, totalPrimes);
        printf("Time elapsed (with balance): %f\n", MPI_Wtime() - startTime);
    }
    else {
      //  printf("Rank %d made it to end\n", myRank);
    }

    MPI_Finalize();

    return 0;
}

int noLoadBalancing(int argc, char** argv, int N) {

    // Set up MPI
    int numRanks, myRank, len, master = 0, tag = 0;
    MPI_Setup(&argc, &argv, &numRanks, &myRank, &len, 0);

    if (numRanks < 2) {
        printf("numRanks must be at least 2\n");
        exit(1);
    }

    int totalPrimes = 0;

    double startTime = MPI_Wtime();

    if (myRank == master) {

        int numWorkers = numRanks - 1; // Master node is not a worker

        int chunkSize = ceil((float)N / (float)numWorkers); // Integer division   
        int currStart = 0;
        int currEnd = chunkSize - 1;

        printf("N = %d\n", N);
        printf("Num Ranks: %d\n", numRanks);
        printf("Running with load balancing disabled\n");
        printf("One-time chunk size ~ %d\n", chunkSize);

        // Send work to each worker node
        for (int worker = 1; worker <= numWorkers; worker++, currStart += chunkSize, currEnd += chunkSize) {

            int sendingEnd = currEnd;
            if (worker == numWorkers) {
                sendingEnd = N;
            }

            MPI_Send(&currStart, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            MPI_Send(&sendingEnd, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
        }

        // Receive the number of primes in chunk calculated by each worker and increment totalPrimes
        for (int worker = 1; worker <= numWorkers; worker++) {
            int primesInChunk;
            MPI_Recv(&primesInChunk, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            totalPrimes += primesInChunk;
        }

    } else {

        // Receive the range from the master
        int myStart, myEnd;
        MPI_Recv(&myStart, 1, MPI_INT, master, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&myEnd, 1, MPI_INT, master, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute the number of primes in this chunk
        int primesInChunk = 0;
        for (int num = myStart; num <= myEnd; num++) {
            if (isPrime(num)) {
                primesInChunk++;
            }
        }

        // Send the number of primes in this chunk to the master
        MPI_Send(&primesInChunk, 1, MPI_INT, master, tag, MPI_COMM_WORLD);
    }

    if (myRank == master) {
        printf("The total number of primes up to %d is %d\n", N, totalPrimes);
        printf("Time elapsed (no balance): %f\n", MPI_Wtime() - startTime);
    }
    else {
     //   printf("Rank %d made it to end\n", myRank);
    }

    MPI_Finalize();

    return 0;
}

int isPrime(int number) {
    if (number <= 1)
        return 0;
    for (int i = 2; i <= sqrt(number); i++) {
        if (number % i == 0)
            return 0;
    }
    return 1;
}
