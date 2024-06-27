#pragma once
#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>
#include <stdarg.h>

#ifndef MAX_DIMS
#define MAX_DIMS 4
#endif

typedef struct NShape {
    int n_dims;
    int dims[MAX_DIMS];
} nshape_t;

typedef struct NMatrix {
    int n_elements;
    int n_dims;
    int dims[MAX_DIMS];
    float *matrix;
} nmatrix_t;

nshape_t nshape_constructor(int n_dims, ...);

void free_nmatrix_list(int size, nmatrix_t *list);

nmatrix_t nmatrix_allocator(nshape_t shape);
nmatrix_t nmatrix_constructor(int n_elements, float *matrix, nshape_t shape);
void nmatrix_reshape(nmatrix_t *m, nshape_t shape);
void nmatrix_shape_contract(nmatrix_t *m, int dim_i);
void nmatrix_shape_extend(nmatrix_t *m, int dim_i, int dim);
void nmatrix_shape_change(nmatrix_t *m, int dim_i, int new_dim);
bool check_nmatrix_shape(nmatrix_t *m, nshape_t shape);

void nmatrix_free(nmatrix_t *m);
nmatrix_t nmatrix_copy(nmatrix_t *src);
void nmatrix_memcpy(nmatrix_t *dst, nmatrix_t *src);
void nmatrix_memset(nmatrix_t *m, float val);

void nmatrix_convolve(nmatrix_t *m1, nmatrix_t *m2,
                      nmatrix_t *result);
void nmatrix_maxpool(nmatrix_t *m1, nshape_t shape,
                     nmatrix_t *result);

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