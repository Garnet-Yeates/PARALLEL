#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

extern int* imageToMat(char* name, int* dims);
extern void matToImage(char* name, int* mat, int* dims);

int getMatValue(int* mat, int r, int c, int rows, int cols);
float getMatValueFloat(float* mat, int r, int c, int rows, int cols);
void setMatValue(int* mat, int r, int c, int rows, int cols, int value);
float* generateBoxBlurMatrix(int n);
void distributeRows(int totalRows, int totalColumns, int numRanks, int* myRowCount, int* rowDisplacements, int* displacements, int* sendCounts);
int applyKernelToPixel(int kernelSize, float* kernelMatrix, int* imageMatrix, int row, int col, int rows, int cols);
void MPI_Setup(int* argc, char*** argv, int* numRanks, int* rank, int* len, int print);

int main(int argc, char** argv) {

    // Set up MPI
    double absoluteStartTime = MPI_Wtime();

    int numRanks, rank, len;
    MPI_Setup(&argc, &argv, &numRanks, &rank, &len, 0);

    // Our compute node is rank 0
    int compute = 0;

    // Set up the box blur kernel and print it on the compute node
    int kernelSize = 25;
    float* kernelMatrix = generateBoxBlurMatrix(kernelSize);    
    if (rank == compute && 0) {
        printf("The following box blur matrix will be applied to the image\n");
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) 
                printf("%f ", kernelMatrix[i * kernelSize + j]);
            printf("\n");
        }
    }

    // Broadcast dimensions as well as the image matrix from compute rank to all ranks

    // Only the compute rank will call imageToMat. The compute will then broadcast the image matrix to all other ranks
    int* matrix = NULL; 
    int dims[2];
    if (rank == compute) {
        matrix = imageToMat("chrome and ryusui.jpg", dims);
    }

    // Broadcast dimensions to all ranks
    MPI_Bcast(dims, 2, MPI_INT, compute, MPI_COMM_WORLD);

    int height = dims[0];
    int width = dims[1];

    // The matrix was automatically allocated/filled on the compute rank via imageToMat; the other ranks must dynamically allocate it themselves
    if (rank != compute) {
        matrix = (int*) malloc(height * width * sizeof(int));
    }

    // Broadcast image to all ranks, using the previously sent width/height to calculate buffer size
    MPI_Bcast(matrix, height * width, MPI_INT, compute, MPI_COMM_WORLD);

    // Perform division of work-load between n nodes. Each node will get a certain number of rows to work on

    int howManyRoadsFor[numRanks]; // How many rows am I working on?
    int sendCountsFor[numRanks]; // How many pixels am I working on? 
    int rowDisplacementsFor[numRanks]; // What row index within the image matrix am I starting with?
    int displacementsFor[numRanks]; // What exact index within the image buffer does my starting row coincide with?
    
    distributeRows(height, width, numRanks, howManyRoadsFor, rowDisplacementsFor, displacementsFor, sendCountsFor);

    int myRowCount = howManyRoadsFor[rank];
    int myPixelCount = sendCountsFor[rank]; 
    int myFirstRow = rowDisplacementsFor[rank];

    double computationStartTime = MPI_Wtime();

    // Perform image processing

    int* myResult = (int*) malloc(myPixelCount * sizeof(int));
    for (int row = myFirstRow, i = 0; row < myFirstRow + myRowCount; row++) {
        for (int col = 0; col < width; col++, i++) {
            if (i >= myPixelCount) {
                printf("Out of bounds"); 
                exit(1);
            }
            myResult[i] = applyKernelToPixel(kernelSize, kernelMatrix, matrix, row, col, height, width);
        }
    }

    // Right now computation time is only measure for rank0 so it isn't too useful
    if (rank == compute) {
        printf("Computation time elapsed for %d ranks: %f\n", numRanks, MPI_Wtime() - computationStartTime);
    }

    // recvBuff will only be defined on the compute rank
    int* recvBuff = NULL;
    if (rank == compute) {
        recvBuff = (int*) malloc(height * width * sizeof(int));
    }

    // Gather results into recvBuff
    MPI_Gatherv(myResult, myPixelCount, MPI_INT, recvBuff, sendCountsFor, displacementsFor, MPI_INT, compute, MPI_COMM_WORLD);

    // Save recvBuff to image
    if (rank == compute) {
        matToImage("processedImage.jpg", recvBuff, dims);
    }

    free(matrix);
    free(myResult);

    if (rank == compute) {
        free(recvBuff);
    }

    free(kernelMatrix);

    // End timer before gather to stop non-calculation overhead from affecting our times.
    if (rank == compute) {
        printf("Total time elapsed for %d ranks: %f\n", numRanks, MPI_Wtime() - absoluteStartTime);
    }

    MPI_Finalize();

    return 0;
}

void distributeRows(int totalRows, int totalColumns, int numRanks, int* myRowCount, int* rowDisplacements, int* displacements, int* sendCounts) {
    
    int baseRowsPerRank = totalRows / numRanks;
    int remainder = totalRows % numRanks;

    // Calculate the number of rows per rank
    for (int i = 0; i < numRanks; i++) {
        myRowCount[i] = baseRowsPerRank;
    }

    // Distribute the remainder among the rows
    for (int i = 0; i < remainder; i++) {
        myRowCount[i]++;
    }

    // Calculate displacements and sendCounts
    displacements[0] = 0;
    sendCounts[0] = myRowCount[0] * totalColumns;
    rowDisplacements[0] = 0;

    for (int i = 1; i < numRanks; i++) {
        displacements[i] = displacements[i - 1] + sendCounts[i - 1];
        sendCounts[i] = myRowCount[i] * totalColumns;
        rowDisplacements[i] = rowDisplacements[i - 1] + myRowCount[i - 1];
    }
}

// Implied that kernelSize is odd. It should never be even
int applyKernelToPixel(int kernelSize, float* kernelMatrix, int* imageMatrix, int row, int col, int rows, int cols) {
    
    int matrixOffset = (kernelSize - 1) / 2;

    int startRow = row - matrixOffset;
    int startCol = col - matrixOffset;

    float sum = 0;
    for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
        for (int kernelCol = 0; kernelCol < kernelSize; kernelCol++) {

            int imageRow = kernelRow + startRow;
            int imageCol = kernelCol + startCol;

            float kernelValue = getMatValueFloat(kernelMatrix, kernelRow, kernelCol, kernelSize, kernelSize);

            int imageValue;
            if (imageRow < 0 || imageRow >= rows || imageCol < 0 || imageCol >= cols) {
                imageValue = 0;
            }
            else {
                imageValue = getMatValue(imageMatrix, kernelRow + startRow, kernelCol + startCol, rows, cols);
            }

            sum += kernelValue * imageValue;
        }
    }

    if (sum > 255) {
        sum = 255;
    }

    return (int) sum;
}

float* generateBoxBlurMatrix(int n) {
    float* matrix = (float*)malloc(n*n * sizeof(float));
    float blur_value = 1.0 / ((float)n*n);
    for (int i = 0; i < n*n; i++) {
        matrix[i] = blur_value;
    }
    return matrix;
}

int getMatValue(int* mat, int r, int c, int rows, int cols) {
    int index = r * cols + c;
    int size = rows*cols;
    if (index >= size) {
        printf("Out of bounds");
        exit(1);
    }
    return mat[index];
}

float getMatValueFloat(float* mat, int r, int c, int rows, int cols) {
    int index = r * cols + c;
    int size = rows*cols;
    if (index >= size) {
        printf("Out of bounds");
        exit(1);
    }
    return mat[index];
}

// cols is required to figure out where it actually is but num rows isn't
void setMatValue(int* mat, int r, int c, int rows, int cols, int value) {
    int index = r * cols + c;
    int size = rows*cols;
    if (index >= size) {
        printf("Out of bounds");
        exit(1);
    }
    mat[index] = value;
}

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