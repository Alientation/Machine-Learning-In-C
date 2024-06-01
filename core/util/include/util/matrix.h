#pragma once
#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>


// FIGURE OUT HOW TO OPTIMIZE
// INSTEAD OF 2D ARRAY (IE ARRAY OF POINTERS TO ARRAYS)
// USE 1D ARRAY TO MODEL THE 2D ARRAY
// TODO
// https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

typedef struct MyMatrix {
    unsigned int r, c;
    float** matrix;

    bool transposed; // todo
} mymatrix_t;

mymatrix_t matrix_allocator(int r, int c);
mymatrix_t matrix_constructor(int r, int c, float** matrix);

/**
 * Frees the allocated 2d float array of the matrix
 */
void matrix_free(mymatrix_t m);
mymatrix_t matrix_copy(mymatrix_t src);
void matrix_memcpy(mymatrix_t dst, mymatrix_t src);

void matrix_multiply(mymatrix_t m1, mymatrix_t m2, 
                     mymatrix_t result);

void matrix_multiply_scalar(mymatrix_t m1, float scalar,
                            mymatrix_t result);

void matrix_elementwise_multiply(mymatrix_t m1, mymatrix_t m2,
                                 mymatrix_t result);

void matrix_add(mymatrix_t m1, mymatrix_t m2,
                mymatrix_t result);
void matrix_add_row(mymatrix_t m1, unsigned int r1, mymatrix_t m2, unsigned int r2,
                    mymatrix_t result, unsigned int r_result);
void matrix_sub(mymatrix_t m1, mymatrix_t m2,
                mymatrix_t result);

// extends matrix with copies of itself
void matrix_column_extend(mymatrix_t m, unsigned int col_factor, 
                          mymatrix_t result);
void matrix_row_extend(mymatrix_t m, unsigned int row_factor, 
                       mymatrix_t result);

void matrix_transpose(mymatrix_t m, 
                      mymatrix_t result);

bool matrix_equal(mymatrix_t m1, mymatrix_t m2);

void matrix_for_each_operator(mymatrix_t m, float (*op)(float), 
                              mymatrix_t result);

void matrix_print(mymatrix_t m);

void matrix_set_values_to_fit(mymatrix_t m, float* elements, unsigned int num_elements);

#endif // MATRIX_H