#pragma once
#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>
#include <stdarg.h>

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


void free_matrix_list(mymatrix_t *matrix_list, int size);

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

// todo maybe create a generic n-dimensional matrix (using macros ??, is it even worth the effort tho)

#ifndef MAX_DIMS
#define MAX_DIMS 4
#endif

typedef struct NMatrix {
    int n_elements;
    int n_dims;
    int dims[MAX_DIMS];
    // unsigned int strides[MAX_DIMS];
    float *matrix;
} nmatrix_t;

void free_nmatrix_list(int size, nmatrix_t *list);

nmatrix_t nmatrix_allocator(int n_dims, ...);
nmatrix_t nmatrix_constructor(int n_elements, float *matrix, int n_dims, ...);
void nmatrix_reshape(nmatrix_t *m, int n_dims, ...);
void nmatrix_shape_contract(nmatrix_t *m, int dim_i);
void nmatrix_shape_extend(nmatrix_t *m, int dim_i, int dim);
void nmatrix_shape_change(nmatrix_t *m, int dim_i, int new_dim);
bool check_nmatrix_shape(nmatrix_t *m, int n_dims, ...);

void nmatrix_free(nmatrix_t *m);
nmatrix_t nmatrix_copy(nmatrix_t *src);
void nmatrix_memcpy(nmatrix_t *dst, nmatrix_t *src);
void nmatrix_memset(nmatrix_t *m, float val);

void nmatrix_multiply(nmatrix_t *m1, nmatrix_t *m2, 
                      nmatrix_t *result);
// todo int nmatrix_multiply_size(nmatrix_t *m1, nmatrix_t *m2);

void nmatrix_multiply_scalar(nmatrix_t *m, float scalar,
                             nmatrix_t *result);

void nmatrix_elementwise_multiply(nmatrix_t *m1, nmatrix_t *m2,
                                  nmatrix_t *result);

void nmatrix_add(nmatrix_t *m1, nmatrix_t *m2,
                 nmatrix_t *result);
void nmatrix_sub(nmatrix_t *m1, nmatrix_t *m2,
                 nmatrix_t *result);

void nmatrix_transpose(nmatrix_t *m, 
                       nmatrix_t *result);
// void nmatrix_transpose_axis(nmatrix_t *m,
                            // nmatrix_t *result, ...);

bool nmatrix_equal(nmatrix_t *m1, nmatrix_t *m2);

void nmatrix_for_each_operator(nmatrix_t *m, float (*op)(float), 
                               nmatrix_t *result);

void nmatrix_print(nmatrix_t *m);

void nmatrix_set_values_to_fit(nmatrix_t *m, int num_elements, float* elements);

#endif // MATRIX_H