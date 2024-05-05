#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define INTS_PER_GB 1024 * 1024 * 1024 / sizeof(int)

int main(int argc, char *argv[]) {
    int numRanks, rank, len, rc;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);
    printf("Number of tasks = %d, My rank = %d, Running on %s\n", numRanks, rank, hostname);

    // Tag 0 for all MPI calls since we don't need to differentiate messages
    int tag = 0;

    MPI_Status stat;
    int buffSize = INTS_PER_GB;
    int* buff = (int*)malloc(buffSize * sizeof(int));

    /*
     * N node ping-pong
     * R0: send/receive R1, then send/receive R2, then send/receive R3, ..., send/receive Rn
     * R1...Rn: receive/send R0
     * Number of send/receive pairs is equivalent to [(numRanks - 1) * 2], so lots of transferring..
     * I expect this to take a bit longer than the ring, not because it has a bottleneck but rather because the amount of
     * data being transferred is higher for this one (6 send/receives with 4 nodes)
     */
    if (rank == 0) {

        // Only fill the buffer on rank 0
        for (int i = 0; i < buffSize; i++) {
            buff[i] = i;
        }

        int gbTransferred = (numRanks - 1) * 2;
        double startTime = MPI_Wtime();

        // Send and receive to all other ranks, starting with rank 1
        for (int i = 1; i < numRanks; i++) {
            int sendingTo = i;
            int receivingFrom = i;
            MPI_Send(buff, buffSize, MPI_INT, sendingTo, tag, MPI_COMM_WORLD);
            MPI_Recv(buff, buffSize, MPI_INT, receivingFrom, tag, MPI_COMM_WORLD, &stat);
        }

        double endTime = MPI_Wtime();
        double timeElapsed = endTime - startTime;
        double bandWidth = gbTransferred / timeElapsed;
        printf("PING PONG: Time elapsed for %d ranks, sending %d gigabytes of data: %f. Bandwidth: %f\n", numRanks, gbTransferred,  timeElapsed, bandWidth);
    }
    else {
        MPI_Recv(buff, buffSize, MPI_INT, 0, tag, MPI_COMM_WORLD, &stat);
        MPI_Send(buff, buffSize, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    // Reuse the same buffer so we don't waste extra time re-allocating a gigabyte

    /*
     * N node ring
     * R0: send to R1, then receive from R(numRanks - 1)
     * R1...Rn: receive from [myRank - 1], send to [(myRank + 1) % numRanks]
     * Number of send/receive pairs is equivalent to numRanks, so it should be faster than ping-pong
     */
    if (rank == 0) {

        int gbTransferred = numRanks;
        double startTime = MPI_Wtime();

        MPI_Send(buff, buffSize, MPI_INT, 1, tag, MPI_COMM_WORLD);
        MPI_Recv(buff, buffSize, MPI_INT, numRanks - 1, tag, MPI_COMM_WORLD, &stat);

        double endTime = MPI_Wtime();
        double timeElapsed = endTime - startTime;
        double bandWidth = gbTransferred / timeElapsed;
        printf("RING: Time elapsed for %d ranks, sending %d gigabytes of data: %f. Bandwidth: %f\n", numRanks, gbTransferred,  timeElapsed, bandWidth);
    }
    else {
        MPI_Recv(buff, buffSize, MPI_INT, rank - 1, tag, MPI_COMM_WORLD, &stat);
        MPI_Send(buff, buffSize, MPI_INT, (rank + 1) % numRanks, tag, MPI_COMM_WORLD);
    }

    free(buff);

    MPI_Finalize();

    return 0;
}

