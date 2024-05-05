#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv) {

	int rank, numRanks;

	MPI_Init(&argc, &argv);
	MPI_COMM_rank(MPI_COMM_WORLD, &rank);
	MPI_COMM_size(MPI_COMM_WORLD, &numranks);

	int n = 4;
	int* matrix = NULL;

	if (rank == 0) {
		matrix = (int*) malloc(n*n*sizeof(int));
		for (int i = 0; i < n*n; i++) {
			matrix[i] = i + 1;
		}
	}

	// EVERY rank shall malloc here - even the sender. Because the sender has the whole matrix but they are
	// still 'sending' one of the rows to themselves. So we malloc the row and it gets filled by MPI_Satter
	int* row = (int*) malloc(n*sizeof(int));

	// MPI_Scatter(*sendbuf, sendcount, sendType, dataReceiving, typeReceiving, root, ranks)
	// Sendcount is how much of the array each one gets
	MPI_Scatter(matrix, n, MPI_INT, row, n, MPI_INT, 0, MPI_COMM_WORLD);

	printf("rank: %d, row: [%d %d %d %d]", rank, row[0], row[1], row[2], row[3]);

	// Free the row for everyone, but only free the matrix for the root (0 in this case)

	free(row);
	if (rank == 0)
		free(matrix);

	MPI_Finalize();

}
