#include <util/matrix.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>

nshape_t nshape_constructor(int n_dims, ...) {
    nshape_t shape = {.n_dims = n_dims};
    va_list ptr;
    va_start(ptr, n_dims);
    for (int i = 0; i < n_dims; i++) {
        shape.dims[i] = va_arg(ptr, int);
        assert(shape.dims[i] > 0);
    }
    va_end(ptr);

    for (int i = n_dims; i < MAX_DIMS; i++) {
        shape.dims[i] = 0;
    }

    return shape;
}

void free_nmatrix_list(int size, nmatrix_t *list) {
    for (int i = 0; i < size; i++) {
        nmatrix_free(&list[i]);
    }
    free(list);
}

nmatrix_t nmatrix_allocator(nshape_t shape) {
    assert(shape.n_dims > 0);
    assert(shape.n_dims <= MAX_DIMS);

    nmatrix_t m = {.n_dims = shape.n_dims};
    m.n_elements = 1;
    for (int i = 0; i < shape.n_dims; i++) {
        m.n_elements *= shape.dims[i];
        m.dims[i] = shape.dims[i];

        assert(shape.dims[i] > 0);
    }

    m.matrix = malloc(sizeof(float) * m.n_elements);
    for (int i = 0; i < m.n_elements; i++) {
        m.matrix[i] = 0;
    }

    return m;
}

// should technically make a copy of matrix to be safe
nmatrix_t nmatrix_constructor(int n_elements, float *matrix, nshape_t shape) {
    assert(n_elements > 0);
    assert(shape.n_dims > 0);
    assert(shape.n_dims <= MAX_DIMS);

    nmatrix_t m = {.n_dims = shape.n_dims};
    m.n_elements = n_elements;
    m.matrix = matrix;
    
    int check_n_elements = 1;
    for (int i = 0; i < shape.n_dims; i++) {
        check_n_elements *= shape.dims[i];
        m.dims[i] = shape.dims[i];

        assert(shape.dims[i] > 0);
    }

    assert(check_n_elements == n_elements);
    return m;
}

nmatrix_t nmatrix_constructor_array(int n_elements, float *matrix, int n_dims, int* dims) {
    assert(n_elements > 0);
    assert(n_dims > 0);
    assert(n_dims <= MAX_DIMS);

    nmatrix_t m;
    m.n_elements = n_elements;
    m.n_dims = n_dims;
    m.matrix = matrix;

    int check_n_elements = 1;
    for (int i = 0; i < n_dims; i++) {
        m.dims[i] = dims[i];
        check_n_elements *= dims[i];

        assert(m.dims[i] > 0);
    } 

    assert(check_n_elements == n_elements);
    return m;
}

// https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
void nmatrix_reshape(nmatrix_t *m, nshape_t shape) {
    m->n_dims = shape.n_dims;
    for (int i = 0; i < shape.n_dims; i++) {
        m->dims[i] = shape.dims[i];
    }

    int check_n_elements = 1;
    for (int i = 0; i < shape.n_dims; i++) {
        check_n_elements *= m->dims[i];

        assert(m->dims[i] > 0);
    }

    assert(check_n_elements == m->n_elements);
}

void nmatrix_shape_contract(nmatrix_t *m, int dim_i) {
    assert(dim_i < m->n_dims && dim_i >= 0);

    m->n_elements /= m->dims[dim_i];
    m->n_dims--;
    m->dims[m->n_dims] = 0;
    for (int i = dim_i; i < m->n_dims - 1; i++) {
        m->dims[i] = m->dims[i+1];
    }

    free(m->matrix);
    m->matrix = malloc(sizeof(float) * m->n_elements);
}

void nmatrix_shape_extend(nmatrix_t *m, int dim_i, int dim) {
    assert(dim_i <= m->n_dims && dim_i >= 0);
    assert(dim > 0);
    
    m->n_dims++;
    for (int i = m->n_dims - 1; i > dim_i; i++) {
        m->dims[i] = m->dims[i-1];
    }
    m->dims[dim_i] = dim;
    m->n_elements *= dim_i;

    free(m->matrix);
    m->matrix = malloc(sizeof(float) * m->n_elements);
}

void nmatrix_shape_change(nmatrix_t *m, int dim_i, int new_dim) {
    assert(dim_i < m->n_dims && dim_i >= 0);
    assert(new_dim > 0);

    m->n_elements /= m->dims[dim_i];
    m->dims[dim_i] = new_dim;
    m->n_elements *= new_dim;
    
    free(m->matrix);
    m->matrix = malloc(sizeof(float) * m->n_elements);
    memset(m->matrix, 0, sizeof(float) * m->n_elements);
}

bool check_nmatrix_shape(nmatrix_t *m, nshape_t shape) {
    if (m->n_dims != shape.n_dims) {
        return false;
    }

    for (int i = 0; i < shape.n_dims; i++) {
        if (m->dims[i] != shape.dims[i]) {
            return false;
        }
    }
    return true;
}

void nmatrix_free(nmatrix_t *m) {
    if (m->matrix == NULL) {
        return;
    }

    free(m->matrix);
}

nmatrix_t nmatrix_copy(nmatrix_t *src) {
    nmatrix_t copy;
    copy.n_elements = src->n_elements;
    copy.n_dims = src->n_dims;
    for (int i = 0; i < copy.n_dims; i++) {
        copy.dims[i] = src->dims[i];
    }
    copy.matrix = malloc(sizeof(float) * src->n_elements);
    memcpy(copy.matrix, src->matrix, sizeof(float) * src->n_elements);
    return copy;
}

// simply just copies the matrix data, will not reshape the matrix
void nmatrix_memcpy(nmatrix_t *dst, nmatrix_t *src) {
    assert(dst->n_elements == src->n_elements); // don't need to check dims
    memcpy(dst->matrix, src->matrix, sizeof(float) * src->n_elements);
}

void nmatrix_memset(nmatrix_t *m, float val) {
    for (int i = 0; i < m->n_elements; i++) {
        m->matrix[i] = val;
    }
}


void nmatrix_convolve(nmatrix_t *m1, nmatrix_t *m2,
                      nmatrix_t *result) {
    assert(m1->n_dims == m2->n_dims);
    assert(m2->n_dims == result->n_dims);
}

void nmatrix_maxpool(nmatrix_t *m1, nshape_t shape,
                     nmatrix_t *result) {
    assert(m1->n_dims == shape.n_dims);
    assert(m1->n_dims == result->n_dims);
}

void matrix_2d_multiply(int r1, int c1, float *m1, int r2, int c2, float *m2,
                        float *dst) {
    assert(c1 == r2);

    for (int r = 0; r < r1; r++) {
        int o1 = r * c1;
        int o2 = r * c2;
        for (int c = 0; c < c2; c++) {
            float dot = 0;
            for (int i = 0; i < c1; i++) {
                dot += m1[o1 + i] * m2[i * c2 + c];
            }
            dst[o2 + c] = dot;
        }
    }
}

// like numpy's matmul https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
// todo parallelize with omp library https://medium.com/tech-vision/parallel-matrix-multiplication-c-parallel-processing-5e3aadb36f27
void nmatrix_multiply(nmatrix_t *m1, nmatrix_t *m2,
                      nmatrix_t *result) {
    assert(m1->n_dims >= 2);
    assert(m2->n_dims >= 2);
    assert(m1->n_dims == m2->n_dims);
    assert(m1->dims[m1->n_dims-1] == m2->dims[m2->n_dims-2]);

    if (m1->n_dims == 2 && m2->n_dims == 2) {
        matrix_2d_multiply(m1->dims[0], m1->dims[1], m1->matrix, m2->dims[0], m2->dims[1], m2->matrix, result->matrix);
        return;
    }

    // all the dimensions other than the last 2 must be identical
    result->n_dims = m1->n_dims;
    int n_inner_matmul = 1;
    int check_n_elements = 1;
    for (int i = 0; i < m1->n_dims - 2; i++) {
        assert(m1->dims[i] == m2->dims[i]);
        result->dims[i] = m1->dims[i];
        n_inner_matmul *= m1->dims[i];
        check_n_elements *= m1->dims[i];
    }

    result->dims[m1->n_dims-2] = m1->dims[m1->n_dims-2];
    result->dims[m1->n_dims-1] = m2->dims[m2->n_dims-1];

    int r1 = m1->dims[m1->n_dims-2];
    int c1 = m1->dims[m1->n_dims-1];
    int r2 = m2->dims[m2->n_dims-2];
    int c2 = m2->dims[m2->n_dims-1];

    check_n_elements *= r1 * c2;
    assert(check_n_elements == result->n_elements);

    int offset_1 = r1 * c1;
    int offset_2 = r2 * c2;
    int offset_3 = r1 * c2;
    for (int i = 0; i < n_inner_matmul; i++) {
        float *src1 = m1->matrix + offset_1*i;
        float *src2 = m2->matrix + offset_2*i;
        float *dst = result->matrix + offset_3*i;
        memset(dst, 0, sizeof(float) * r1 * c2);

        for (int r = 0; r < r1; r++) {
            int dst_offset = r*c2;
            int src1_offset = r*c1;
            for (int i = 0; i < c1; i++) {
                int src2_offset = i*c2;
                float val1 = src1[src1_offset + i];
                for (int c = 0; c < c2; c++) {
                    dst[dst_offset + c] += val1 * src2[src2_offset + c];
                }
            }
        }
    }
}

void nmatrix_multiply_scalar(nmatrix_t *m, float scalar,
                             nmatrix_t *result) {
    assert(m->n_dims == result->n_dims);
    assert(m->n_elements == result->n_elements);
    for (int i = 0; i < result->n_dims; i++) {
        assert(m->dims[i] == result->dims[i]);
    }

    for (int i = 0; i < m->n_elements; i++) {
        result->matrix[i] = m->matrix[i] * scalar;
    }
}

void nmatrix_elementwise_multiply(nmatrix_t *m1, nmatrix_t *m2,
                                  nmatrix_t *result) {
    assert(m1->n_dims == m2->n_dims);
    assert(m1->n_elements == m2->n_elements);
    assert(m1->n_dims == result->n_dims);
    assert(m1->n_elements == result->n_elements);
    for (int i = 0; i < m1->n_dims; i++) {
        assert(m1->dims[i] == m2->dims[i]);
        assert(m1->dims[i] == result->dims[i]);
    }

    for (int i = 0; i < m1->n_elements; i++) {
        result->matrix[i] = m1->matrix[i] * m2->matrix[i];
    }
}

void nmatrix_add(nmatrix_t *m1, nmatrix_t *m2,
                 nmatrix_t *result) {
    assert(m1->n_dims == m2->n_dims);
    assert(m1->n_elements == m2->n_elements);
    assert(m1->n_dims == result->n_dims);
    assert(m1->n_elements == result->n_elements);
    for (int i = 0; i < m1->n_dims; i++) {
        assert(m1->dims[i] == m2->dims[i]);
        assert(m1->dims[i] == result->dims[i]);
    }

    for (int i = 0; i < m1->n_elements; i++) {
        result->matrix[i] = m1->matrix[i] + m2->matrix[i];
    }
}

void nmatrix_sub(nmatrix_t *m1, nmatrix_t *m2,
                 nmatrix_t *result) {
    assert(m1->n_dims == m2->n_dims);
    assert(m1->n_elements == m2->n_elements);
    assert(m1->n_dims == result->n_dims);
    assert(m1->n_elements == result->n_elements);
    for (int i = 0; i < m1->n_dims; i++) {
        assert(m1->dims[i] == m2->dims[i]);
        assert(m1->dims[i] == result->dims[i]);
    }

    for (int i = 0; i < m1->n_elements; i++) {
        result->matrix[i] = m1->matrix[i] - m2->matrix[i];
    }
}


void nmatrix_transpose_2D(nmatrix_t *m,
                          nmatrix_t *result) {
    assert(m->n_dims == 2);
    assert(result->n_dims == 2);
    assert(m->n_elements == result->n_elements);
    assert(m->dims[0] == result->dims[1]);
    assert(m->dims[1] == result->dims[0]);

    for (int r = 0; r < m->dims[0]; r++) {
        for (int c = 0; c < m->dims[1]; c++) {
            result->matrix[c * result->dims[1] + r] = m->matrix[r * m->dims[1] + c];
        }
    }
}

// https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
void nmatrix_transpose(nmatrix_t *m, 
                       nmatrix_t *result) {
    if (m->n_dims == 1 || (m->n_dims == 2 && m->dims[1] == 1)) {
        nmatrix_memcpy(result, m);
        result->n_dims = 2;
        result->dims[0] = 1;
        result->dims[1] = m->dims[0];
        return;
    } if (m->n_dims == 2) {
        nmatrix_transpose_2D(m, result);
        return;
    }

    assert(m->n_dims == result->n_dims);
    assert(m->n_elements == result->n_elements);
    for (int i = 0; i < m->n_dims/2; i++) {
        assert(m->dims[i] == result->dims[result->n_dims - 1 - i]);
    }

    int strides[MAX_DIMS] = {0};
    int stride = 1;
    for (int i = m->n_dims - 1; i >= 0; i--) {
        strides[m->n_dims-1-i] = stride;
        stride *= m->dims[i];
    }
    
    int *dims = result->dims;
    int pos[MAX_DIMS] = {0};
    int offset = 0;
    for (int i = 0; i < m->n_elements; i++) {
        result->matrix[i] = m->matrix[offset];
        for (int dim = m->n_dims-1; dim >= 0; dim--) {
            offset += strides[dim];
            pos[dim]++;
            
            if (pos[dim] == dims[dim]) {
                pos[dim] = 0;
                offset -= strides[dim] * dims[dim];
                continue;
            }
            break;
        }
    }
}

bool nmatrix_equal(nmatrix_t *m1, nmatrix_t *m2) {
    if (m1->n_dims != m2->n_dims || m1->n_elements != m2->n_elements) {
        return false;
    }
    for (int i = 0; i < m1->n_dims; i++) {
        if (m1->dims[i] != m2->dims[i]) {
            return false;
        }
    }

    return memcmp(m1->matrix, m2->matrix, sizeof(float) * m1->n_elements) == 0;
}

void nmatrix_for_each_operator(nmatrix_t *m, float (*op)(float), 
                               nmatrix_t *result) {
    assert(m->n_dims == result->n_dims);
    assert(m->n_elements == result->n_elements);
    for (int i = 0; i < m->n_dims; i++) {
        assert(m->dims[i] == result->dims[i]);
    }

    for (int i = 0; i < m->n_elements; i++) {
        result->matrix[i] = op(m->matrix[i]);
    }
}

void nmatrix_print_nd(nmatrix_t *m, bool is_first, int tot_dims, int cur_dim) {
    if (m->n_dims == 2) {
        if (!is_first) {
            printf(",\n\n");
        }
        printf("[");

        for (int r = 0; r < m->dims[0]; r++) {
            if (r != 0) {
                printf(",\n");
                for (int i = 0; i < tot_dims-1 - !is_first; i++) {
                    printf(" ");
                }
            }
            printf("[");
            
            for (int c = 0; c < m->dims[1]; c++) {
                if (c != 0) {
                    printf(", ");
                }
                printf("%-6.3f", (float) round(m->matrix[r * m->dims[1] + c] * 1000)/1000);
            }
            printf("]");
        }
        printf("]");
        return;
    }

    if (!is_first) {
        printf(",\n");
    }
    printf("[");
    
    for (int outer_dim = 0; outer_dim < m->dims[0]; outer_dim++) {
        int stride = m->n_elements / m->dims[0];
        nmatrix_t matrix = nmatrix_constructor_array(stride, m->matrix + outer_dim * stride, m->n_dims-1, &m->dims[1]);
        nmatrix_print_nd(&matrix, outer_dim == 0, tot_dims, cur_dim+1);
    }
    printf("]");
}

void nmatrix_print(nmatrix_t *m) {
    printf("%u-Dimensional Matrix: (", m->n_dims);
    for (int i = 0; i < m->n_dims; i++) {
        if (i != 0) {
            printf(",");
        }
        printf("%u", m->dims[i]);
    }
    printf(")\n");

    if (m->n_dims == 1) {
        for (int i = 0; i < m->n_elements; i++) {
            printf("%-7.3f  ", (float) round(m->matrix[i] * 1000)/1000);
        }
    } else if (m->n_dims == 2) {
        for (int r = 0; r < m->dims[0]; r++) {
            printf("%-3d: ", r);
            for (int c = 0; c < m->dims[1]; c++) {
                printf("%-7.4f  ", (float) round(m->matrix[r * m->dims[1] + c] * 1000)/1000);
            }
            printf("\n");
        }
    } else {
        nmatrix_print_nd(m, true, m->n_dims, 0);
    }
    printf("\n\n");
}

void nmatrix_set_values_to_fit(nmatrix_t *m, int num_elements, float* elements) {
    assert(m->n_elements == num_elements);
    memcpy(m->matrix, elements, sizeof(float) * num_elements);
}