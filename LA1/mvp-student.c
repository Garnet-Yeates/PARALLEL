#include <stdio.h>
#include <stdlib.h>
#include "mvp-student.h"

double getMatValue(double* mat, int r, int c, int rows, int cols) {
    int index = r * cols + c;
    int size = rows*cols;
    if (index >= size) {
        printf("Out of bounds");
        exit(1);
    }
    return mat[index];
}

// cols is required to figure out where it actually is but num rows isn't
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

int main() {
    
    int s = 5;
    double* matrix = (double*) malloc((s*s) * sizeof(double));
    
    double* vector = (double*) malloc(s * sizeof(double));
    for (int i = 0; i < s; i++) {
        vector[i] = 1;
    } 
    
    assignMatrix(matrix, s, s);
    printMatrix(matrix, s, s);
    
    double* result = mvp(matrix, vector, s, s);
    printf("Result:\n");
    for (int i = 0; i < s; i++) {
        printf("%f\n", result[i]);
    }

    return 0;
}
