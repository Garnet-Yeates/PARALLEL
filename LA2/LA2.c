#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DOUBLES_PER_GB 1024 * 1024 * 1024 / sizeof(double)

int dot(int* vector1, int* vector2, int dimension);
int divideVector(int numRanks, int dimension, int* displacements, int* sendCounts);

int main(int argc, char** argv) {
    int rank, numRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int source = 0;
    int dimension;
    int* vector1 = NULL;
    int* vector2 = NULL;

    // Only rank 0 knows the value of dimension
    // Simulating user input/file IO
    if (rank == source) {
        dimension = DOUBLES_PER_GB * 10;
        printf("Num Ranks: %d\n", numRanks);
        printf("Doubles per GB: %lu\n", DOUBLES_PER_GB);
        printf("Dimension: %d\n", dimension);
    }
    MPI_Bcast(&dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Now, every rank has the correct value for dimension

    // Rank 0 is the only one that has
    // vector1 and vector2 and fills them with values
    if (rank == source) {
        vector1 = (int*) malloc(dimension * sizeof(int));
        vector2 = (int*) malloc(dimension * sizeof(int));
        for (int i = 0; i < dimension; i++) {
            vector1[i] = vector2[i] = 2;
        }
    }

    // Perform the splitting for Scatterv as well as the scatter itself

    int displacements[numRanks];
    int sendCounts[numRanks];
    divideVector(numRanks, dimension, displacements, sendCounts);

    int localReceiveCount = sendCounts[rank];
    int* localVector1 = (int*) malloc(localReceiveCount * sizeof(int));
    int* localVector2 = (int*) malloc(localReceiveCount * sizeof(int));

    MPI_Scatterv(vector1, sendCounts, displacements, MPI_INT, localVector1, localReceiveCount, MPI_INT, source, MPI_COMM_WORLD);
    MPI_Scatterv(vector2, sendCounts, displacements, MPI_INT, localVector2, localReceiveCount, MPI_INT, source, MPI_COMM_WORLD);

    // Time and perform the dot product
    double myStartTime = MPI_Wtime();
    int myResult = dot(localVector1, localVector2, localReceiveCount);
    double myTimeElapsed = MPI_Wtime() - myStartTime;

    // Combine all the individual results together using a sum
    int result = 0;
    MPI_Allreduce(&myResult, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Combine individual dot product times together to get an average, as well as max (max is important for accurate time
    // calculation that includes the bottleneck. We are only as fast as our slowest worker since all workers need to finish)
    double totalTimeElapsed = 0, avgTimeElapsed = 0;
    double maxTimeElapsed = 0;
    MPI_Allreduce(&myTimeElapsed, &totalTimeElapsed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&myTimeElapsed, &maxTimeElapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    avgTimeElapsed = totalTimeElapsed / numRanks; // Calculate average time

    if (rank == source) {
        printf("Result: %d\n", result);
        printf("Average dot product time per rank: %f\n", avgTimeElapsed);
        printf("Max dot product time per rank: %f\n", maxTimeElapsed);
    }

    // printf("Rank: %d, Num Elements: %d, My Time: %f\n", rank, localReceiveCount, myTimeElapsed);

    MPI_Finalize();

    free(localVector1);
    free(localVector2);

    if (rank == source) {
        free(vector1);
        free(vector2);
    }
    
    return 0;
}

int dot(int* vector1, int* vector2, int dimension) {
    int result = 0;
    for (int i = 0; i < dimension; i++) {
        result += vector1[i] * vector2[i];
    }
    return result;
}

int divideVector(int numRanks, int dimension, int* displacements, int* sendCounts) {
    int flooredEachCount = dimension / numRanks;

    int currentDisplacement = 0;
    for (int rank = 0; rank < numRanks; rank++, currentDisplacement += flooredEachCount) {
        displacements[rank] = currentDisplacement;
        sendCounts[rank] = flooredEachCount;
        if (rank == numRanks - 1) {
            sendCounts[rank] = dimension - currentDisplacement; // If not divded evenly, the last rank gets the most work
        }
    }
}
