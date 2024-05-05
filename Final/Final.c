#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#include <mpi.h>

#include "mpi_helpers.h"

extern int* imageToMat(char* name, int* dims);
extern void matToImage(char* name, int* mat, int* dims);

int mandelbrotIterations(double real, double imag, int maxIterations) {
    double zReal = 0.0;
    double zImag = 0.0;
    int iterations = 0;

    while (iterations < maxIterations && (zReal * zReal + zImag * zImag) < 4.0) {
        double newReal = zReal * zReal - zImag * zImag + real;
        double newImag = 2 * zReal * zImag + imag;
        zReal = newReal;
        zImag = newImag;
        iterations++;
    }

    return iterations;
}

int main(int argc, char** argv) {

    if (argc != 6) {
        printf("Error: there must be 4 arguments\n");
        return 1;
    }

    // Max number of threads that can be used based on the system and runtime
    int maxThreads = omp_get_max_threads();

    // Image width
    int W = atoi(argv[1]);

    // Image height
    int H = atoi(argv[2]);

    // MAX ITERATIONS
    int I = atoi(argv[3]);

    // Chunk Size
    int C = atoi(argv[4]);

    // Can't be more than maxThreads
    int T = atoi(argv[5]);
    if (T > maxThreads) {
        printf("The supplied number of threads (%d) can not be more than %d", T, maxThreads);
        return 1;
    }

    // INIT MPI

    int numRanks, myRank, len, master = 0, tag = 0;
    MPI_Setup(&argc, &argv, &numRanks, &myRank, &len, 0);

    // BEGIN WORK

    int* mat; // H x W. Each MPI worker will perform 'tasks', processing multiple rows per task

    int totalChunks = ceil(H / (float) C);

    int WORK_SIGNAL = -1;
    int TERMINATE_SIGNAL = -2;

    double startTime = MPI_Wtime();

    if (myRank == 0) {

        mat = (int*) malloc(W * H * sizeof(int));

        printf("Running the program with \nWidth=%d\nHeight=%d\nThreads=%d\nRanks=%d\nMaxIterations=%d\nRows Per Task=%d\nTotal Tasks=%d\n", W, H, T, numRanks, I, C, totalChunks);

        int currRow = 0;
        int currEndRow = currRow + C;
        
        for (int worker = 1; worker < numRanks; worker++) {
            MPI_Send(&WORK_SIGNAL, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            MPI_Send(&currRow, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            MPI_Send(&currEndRow, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
            currRow += C;
            currEndRow += C;
        }

        int numWorkers = numRanks - 1;
        bool terminatedWorkers[numWorkers];
        for (int i = 0; i < numWorkers; i++) {
            terminatedWorkers[i] = false;
        }

        while (true) {

            int worker, signal, processedStartRow, processedEndRow;
            MPI_Recv(&worker, 1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&signal, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Process result from this worker
            if (signal == WORK_SIGNAL) {

                MPI_Recv(&processedStartRow, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&processedEndRow, 1, MPI_INT, worker, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int rowsProcessed = processedEndRow - processedStartRow;
                int workerResultSize = rowsProcessed * W;

                int* workerResult = (int*) malloc(sizeof(int) * workerResultSize);
                MPI_Recv(workerResult, workerResultSize, MPI_INT, worker, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // printf("Got rows %d to %d from rank %d\n", processedStartRow, processedEndRow, worker);

                for (int i = 0, row = processedStartRow; row < processedEndRow; row++) {
                    for (int col = 0; col < W; col++, i++) {
                        mat[row * W + col] = workerResult[i];
                    }
                }

                // Send another MPI Task (work signal), or send terminate signal if there is no more work
                // If we send the terminate signal, the worker will send the signal back to us to confirm they are done
                if (currRow < H) {
                    MPI_Send(&WORK_SIGNAL, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
                    MPI_Send(&currRow, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
                    MPI_Send(&currEndRow, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
                    currRow += C;
                    currEndRow += C;
                    if (currEndRow > H) {
                        currEndRow = H; // currEndRow is exclusive btw 
                    }
                }
                else {
                    MPI_Send(&TERMINATE_SIGNAL, 1, MPI_INT, worker, tag, MPI_COMM_WORLD);
                }
            }

            // Get confirmation from workers that they received their termination signal and broke from their worker loops
            // This is so we make sure that we have gotten results from all workers before exiting the master while loop
            else if (signal == TERMINATE_SIGNAL) {

                terminatedWorkers[worker - 1] = true;

                bool allTerminated = true;
                for (int i = 0; i < numWorkers; i++) {
                    if (!terminatedWorkers[i]) {
                        allTerminated = false;
                        break;
                    }
                }
                if (allTerminated) {
                    break;
                }
            }

            // Error handling
            else {
                printf("ERROR: Master node received invalid signal\n");
                exit(0);
            }
        }
    } 
    else {
        double realMin = -2.0;
        double realMax = 1.0;
        double imagMin = -1;
        double imagMax = 1;

        int tasksCompleted = 0;

        // Total time spent working for this node
        double totalWorkTime = 0;

        // Total time each thread on this node spent working
        double threadTimes[T];
        for (int i = 0; i < T; i++) {
            threadTimes[i] = 0;
        }

        while (true) {

            int signal;
            MPI_Recv(&signal, 1, MPI_INT, master, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (signal == WORK_SIGNAL) {

                double workStartTime = MPI_Wtime();

                int startRow, endRow;
                MPI_Recv(&startRow, 1, MPI_INT, master, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&endRow, 1, MPI_INT, master, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int rowsProcessing = endRow - startRow;
                int bufferSize = rowsProcessing * W;
                int* buffer = (int*) malloc(sizeof(int) * bufferSize);

                int workloads[T]; // How many rows will each thread do
                int starts[T]; // What row index does each thread start with (built dynamically using workloads[T])

                // Set workloads as evenly as possible, using division then remainder distribution
                int baseWorkload = rowsProcessing / T, remainingWorkload = rowsProcessing % T;
                for (int i = 0; i < T; i++) 
                    workloads[i] = baseWorkload;
                for (int i = 0; i < remainingWorkload; i++) 
                    workloads[i]++;

                // Calculate displacements from workloads
                starts[0] = startRow;
                for (int i = 1; i < T; i++) {
                    starts[i] = starts[i - 1] + workloads[i - 1];
                }

                #pragma omp parallel num_threads(T)
                {
                    double threadStart = omp_get_wtime();
                    int threadId = omp_get_thread_num();

                    int workload = workloads[threadId];
                    int threadStartRow = starts[threadId];
                    int threadEndRow = threadStartRow + workload;

                    // Perform computation based on workload distribution
                    for (int row = threadStartRow; row < threadEndRow; row++) {
                        for (int col = 0; col < W; col++) {
                            double real = realMin + (realMax - realMin) * col / (W - 1);
                            double imag = imagMin + (imagMax - imagMin) * row / (H - 1);
                            int iterations = mandelbrotIterations(real, imag, I);
                            buffer[(row - startRow) * W + col] = iterations;
                        }
                    }

                    threadTimes[threadId] = threadTimes[threadId] + omp_get_wtime() - threadStart;
                }

                MPI_Send(&myRank, 1, MPI_INT, master, tag, MPI_COMM_WORLD);
                MPI_Send(&WORK_SIGNAL, 1, MPI_INT, master, tag, MPI_COMM_WORLD);
                MPI_Send(&startRow, 1, MPI_INT, master, tag, MPI_COMM_WORLD);
                MPI_Send(&endRow, 1, MPI_INT, master, tag, MPI_COMM_WORLD);
                MPI_Send(buffer, bufferSize, MPI_INT, master, tag, MPI_COMM_WORLD);

                totalWorkTime += MPI_Wtime() - workStartTime;
                tasksCompleted++;
            } 
            else if (signal == TERMINATE_SIGNAL) {
                MPI_Send(&myRank, 1, MPI_INT, master, tag, MPI_COMM_WORLD);
                MPI_Send(&TERMINATE_SIGNAL, 1, MPI_INT, master, tag, MPI_COMM_WORLD);
                break;
            }
            else {
                printf("ERROR: Worker node received invalid signal\n");
                exit(0);
            }
        }

        printf("Rank %d has finished. Total work time for this rank is: %f\n", myRank, totalWorkTime);
        printf("Total tasks completed for rank %d: %d\n", myRank, tasksCompleted);
        printf("Thread times for rank %d:\n", myRank);
        for (int i = 0; i < T; i++) {
            printf("  %f\n", threadTimes[i]);
        }

    }

    if (myRank == 0) {

        int* image = malloc(H * W * sizeof(int));
        for (int i = 0; i < H*W; i++) {
            double lerp = mat[i] / (double) I; // iterations/maxIterations
            image[i] = 255*(1 - lerp);
        }

        printf("Total time = %f\n", MPI_Wtime() - startTime);

        int dims[2] = { H, W };
        matToImage("image.jpg", image, dims);

        free(mat);
    }

    MPI_Finalize();

    return 0;
}