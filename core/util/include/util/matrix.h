#pragma once
#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    unsigned int r, c;
    double** matrix;
} matrix_t;

matrix_t matrix_constructor(int r, int c, double** matrix);
matrix_t matrix_multiply(matrix_t m1, matrix_t m2);
matrix_t matrix_add(matrix_t m1, matrix_t m2);
matrix_t matrix_sub(matrix_t m1, matrix_t m2);

// extends matrix with copies of itself
matrix_t matrix_column_extend(matrix_t m, unsigned int col_factor);
matrix_t matrix_row_extend(matrix_t m, unsigned int row_factor);

matrix_t matrix_transpose(matrix_t m);

#endif