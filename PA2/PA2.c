#include <stdio.h>
#include <stdlib.h>
#include "mvp-student.h"
#include <mpi.h>

#define DOUBLES_PER_GB 1024 * 1024 * 1024 / sizeof(double)
#define ASSUMED_RAM_CAPACITY_GB 16

double getMatValue(double* mat, int r, int c, int rows, int cols) {
    int index = r * cols + c;
    int size = rows*cols;
    if (index >= size) {
        printf("Out of bounds");
        exit(1);
    }
    return mat[index];
}

// m (num columns) is required to figure out where it actually is but num rows (n) isn't
void setMatValue(double* mat, int r, int c, int rows, int cols, double value) {
    int index = r * cols + c;
    int size = rows*cols;
    if (index >= size) {
        printf("Out of bounds");
        exit(1);
    }
    mat[index] = value;
}

void printMatrix(double* mat, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%f ", getMatValue(mat, i, j, r, c));
        }
        printf("\n");
    }
}

void assignMatrix(double* mat, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int diff = i - j;
            int value = 0;
            switch (diff) {
                case 0:
                    value = 2;
                    break;
                case 1:
                case -1:
                    value = 1;
                    break;
                default:
                    value = 0;
                    break;
            }
            setMatValue(mat, i, j, n, m, value);
        }
    }
}

// Make sure I call free(result) after....
// I actually don't think I need to make any changes to this for it to work in parallel (I hope)
double* mvp(double* mat, double* vec, int rows, int cols) {
    double* result = (double*) malloc(rows * sizeof(double));
    for (int row = 0; row < rows; row++) {
        double sum = 0;
        for (int col = 0; col < cols; col++) {
            sum += getMatValue(mat, row, col, rows, cols) * vec[col];
        }
        result[row] = sum;
    }
    return result;
}

int main(int argc, char** argv) {
    
    int numRanks, rank, len, rc;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);
    printf("Number of tasks = %d, My rank = %d, Running on %s\n", numRanks, rank, hostname);

    // So to find a number that is divisible by 1....10 (all numbers between too) I actually asked ChatGPT
    // and it told me to use the LCM (least common multiple), which starts at 2520. So for the matrix size,
    // I will start at 2520 and we will scale it up from there if I need it to be bigger
    int scale = 12;
    int LCM = 2520; // This is LCM for 1...10
    int numRows = LCM * scale;
    int numCols = numRows;
    // TODO in the future LCM should account for 1..12 not 1..10

    if (numRows % numRanks != 0) {
        printf("The number of rows must be divisible by the number of ranks\n");
        exit(1);
    }

    double* matrix = NULL;
    int rowsPerRank = numRows / numRanks; // We ensure this is evenly divided above
    int sendCount = rowsPerRank * numCols;
    int source = 0;

    int totalAdditions = numRows;
    int totalMultiplications = numRows * numCols;
    int totalOps = totalAdditions + totalMultiplications;

    int additionsPerRank = rowsPerRank;
    int multsPerRank = rowsPerRank * numCols;
    int perRankOps = additionsPerRank + multsPerRank;

    // Vector is defined for all ranks since all ranks will work on the full vector
    double* vector = (double*) malloc(numCols * sizeof(double));
    for (int i = 0; i < numCols; i++) {
        vector[i] = 1;
    } 

    // Num rows must be divisble by the number of nodes. This is because we are not doing this in a way where
    // the matrix size scales with the number of nodes. The matrix size is going to be constant (very large)
    // and the nodes are going to get a piece of it. If we have a 10x10 matrix (10 rows), we need to make sure
    // it is divisible so that each rank can get 1 or more FULL rows (in this case: 1, 2, 4, 5, or 10 ranks)
    
    double startTime = -1;
    // Matrix is only defined for our source. The matrix will be very large so we only want to do this where we need to
    if (rank == source) {
        unsigned long maxDoubles = (DOUBLES_PER_GB) * (ASSUMED_RAM_CAPACITY_GB);
        printf("Assumed max amount of doubles we can fit per gb: %lu\n", DOUBLES_PER_GB);
        printf("Assumed max amount of doubles we can fit for 16gb ram: %lu\n", maxDoubles);
        printf("Total Operations: %d\n", totalOps);
        printf("Operations per node: %d\n", perRankOps);
        printf("Total Nodes: %d\n", numRanks);
        printf("Our matrix size is: %d x %d. So the number of doubles we are putting in RAM is %d\n", numRows, numCols, numRows*numCols);
        matrix = (double*) malloc((numRows*numCols) * sizeof(double));
        printf("Matrix being initialized on source (rank %d): OMITTED\n", rank);
        assignMatrix(matrix, numRows, numCols);
  //    printMatrix(matrix, numRows, numCols); Omit this because its super big and wastes time
        printf("This matrix will be multiplied by the following vector: OMITTED\n");
  //    printMatrix(vector, 1, numCols); Omit this as well, only really needed for debugging
        startTime = MPI_Wtime();
    }

    // Allocate subMatrix on each node. MPI_Scatter will fill this up with the data scattered from source
    double* subMatrix = (double*) malloc(sendCount * sizeof(double));

    // Perform, and measure time of the scatter

    double scatterStart = -1;
    if (rank == source) {
        scatterStart = MPI_Wtime();
    }

    // Scatter matrix so that all ranks get a 'subMatrix'
    MPI_Scatter(matrix, sendCount, MPI_DOUBLE, subMatrix, sendCount, MPI_DOUBLE, source, MPI_COMM_WORLD);

    if (rank == source) {
        double scatterElapsed = MPI_Wtime() - scatterStart;
        printf("FOR MATRIX SIZE %d x %d, (%d RANKS), SCATTER TIME IS: %f\n", numRows, numCols, numRanks, scatterElapsed);
    }

    // Now down here, all of our ranks will have their respective 'subMatrix' to work with (including rank 0)
    // The number of elements in the result that are computed on each rank is equal to rowsPerRank
    // If we have 10 rows, and 5 ranks, rowsPerRank is 2. This means each rank computes 2 elements of the result vector

    double* subResult = mvp(subMatrix, vector, rowsPerRank, numCols);
    
    // result is only allocated on the source (rank 0)
    double* result = NULL;
    if (rank == source) {
        result = (double*) malloc(numRows * sizeof(double));
    }

    // Perform, and measure time of the gather

    double gatherStart = -1;
    if (rank == source) {
        gatherStart = MPI_Wtime();
    }

    int destination = source; // The source who sent the matrix to us will be the destination that gets the result back
    MPI_Gather(subResult, rowsPerRank, MPI_DOUBLE, result, rowsPerRank, MPI_DOUBLE, destination, MPI_COMM_WORLD);
   
    if (rank == source) {
        double gatherElapsed = MPI_Wtime() - gatherStart;
        printf("FOR MATRIX SIZE %d x %d, (%d RANKS), GATHER TIME IS: %f\n", numRows, numCols, numRanks, gatherElapsed);
    }

    // Free our subResult
    free(subResult); 
    subResult = NULL;

    // Free our subMatrix
    free(subMatrix); 
    subMatrix = NULL;

    if (rank == source) {
        double totalElapsed = MPI_Wtime() - startTime;
        printf("TIME TO SOLUTION FOR %d RANKS %f\n", numRanks, totalElapsed);

        printf("Result has been gathered back to source (rank %d): OMITTED\n", rank);
    //  printMatrix(result, 1, numRows);
        free(matrix);
        free(result);
    }

    MPI_Finalize();

    return 0;
}