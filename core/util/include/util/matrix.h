#pragma once
#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    unsigned int r, c;
    double** matrix;
} matrix_t;

matrix_t matrix_constructor(int r, int c, double** matrix);

/**
 * Frees the allocated 2d double array of the matrix
 */
void matrix_free(matrix_t* m);
void matrix_multiply(matrix_t *m1, matrix_t *m2, 
                     matrix_t *result);
void matrix_add(matrix_t *m1, matrix_t *m2,
                matrix_t *result);
void matrix_add_row(matrix_t *m1, unsigned int r1, matrix_t *m2, unsigned int r2,
                    matrix_t *result, unsigned int r_result);
void matrix_sub(matrix_t *m1, matrix_t *m2,
                matrix_t *result);

// extends matrix with copies of itself
void matrix_column_extend(matrix_t *m, unsigned int col_factor, 
                          matrix_t *result);
void matrix_row_extend(matrix_t *m, unsigned int row_factor, 
                       matrix_t *result);

void matrix_transpose(matrix_t *m, 
                      matrix_t *result);


#endif