#pragma once
#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

// todo GOAL, since matrix_t is super small, dont bother returning as pointer, just return a copy

typedef struct Matrix {
    unsigned int r, c;
    double** matrix;

    bool transposed; // todo
} matrix_t;

matrix_t* matrix_allocator(int r, int c);
matrix_t* matrix_constructor(int r, int c, double** matrix);

/**
 * Frees the allocated 2d double array of the matrix
 */
void matrix_free(matrix_t* m);
matrix_t* matrix_copy(matrix_t* src);
void matrix_memcpy(matrix_t *dst, matrix_t *src);

void matrix_multiply(matrix_t *m1, matrix_t *m2, 
                     matrix_t *result);

void matrix_multiply_scalar(matrix_t *m1, double scalar,
                            matrix_t *result);

void matrix_elementwise_multiply(matrix_t *m1, matrix_t *m2,
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

bool matrix_equal(matrix_t *m1, matrix_t *m2);

void matrix_for_each_operator(matrix_t *m, double (*op)(double), 
                              matrix_t *result);

void matrix_print(matrix_t *m);

void matrix_set_values_to_fit(matrix_t *m, double* elements, unsigned int num_elements);

#endif