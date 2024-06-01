#include <util/matrix.h>

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <util/debug_memory.h>

mymatrix_t matrix_allocator(int r, int c) {
    mymatrix_t m;
    m.r = r;
    m.c = c;
    m.matrix = (float**) malloc(r * sizeof(float*));
    for (int i = 0; i < r; i++) {
        m.matrix[i] = (float*) malloc(c * sizeof(float));
        for (int j = 0; j < c; j++) {
            m.matrix[i][j] = 0;
        }
    }
    m.transposed = false;

    return m;
}

mymatrix_t matrix_constructor(int r, int c, float** matrix) {
    mymatrix_t m;
    m.r = r;
    m.c = c;
    m.matrix = matrix;
    m.transposed = false;
    return m;
}

void matrix_free(mymatrix_t matrix) {
    if (matrix.matrix == NULL) {
        return;
    }

    for (int r = 0; r < matrix.r; r++) {
        free(matrix.matrix[r]);
    }
    free(matrix.matrix);
}

mymatrix_t matrix_copy(mymatrix_t src) {
    mymatrix_t copy;
    copy.r = src.r;
    copy.c = src.c;
    copy.matrix = (float**) malloc(copy.r * sizeof(float*));
    for (int i = 0; i < copy.r; i++) {
        copy.matrix[i] = (float*) malloc(copy.c * sizeof(float));
    }
    matrix_memcpy(copy, src);
    return copy;
}

void matrix_memcpy(mymatrix_t dst, mymatrix_t src) {
    assert(dst.r == src.r && dst.c == src.c);

    if (dst.matrix == src.matrix) {
        // points to same matrix, dont need to do any work
        return;
    }

    for (int r = 0; r < dst.r; r++) {
        for (int c = 0; c < dst.c; c++) {
            dst.matrix[r][c] = src.matrix[r][c];
        }
    }
}

// use https://en.wikipedia.org/wiki/Strassen_algorithm for large matrices
// todo
void matrix_multiply(mymatrix_t m1, mymatrix_t m2,
                     mymatrix_t result) {
    assert(m1.c == m2.r);
    assert(m1.r == result.r && m2.c == result.c);

    for (int r = 0; r < result.r; r++) {
        for (int c = 0; c < result.c; c++) {
            float dot = 0;
            for (int i = 0; i < m1.c; i++) {
                dot += m1.matrix[r][i] * m2.matrix[i][c];
            }
            result.matrix[r][c] = dot;
        }
    }
}

void matrix_multiply_scalar(mymatrix_t m1, float scalar,
                            mymatrix_t result) {
    assert(m1.r == result.r && m1.c == result.c);
    for (int r = 0; r < result.r; r++) {
        for (int c = 0; c < result.c; c++) {
            result.matrix[r][c] = m1.matrix[r][c] * scalar;
        }
    }
}

void matrix_elementwise_multiply(mymatrix_t m1, mymatrix_t m2,
                                 mymatrix_t result) {
    assert(m1.r == m2.r && m1.c == m2.c);
    assert(m1.r == result.r && m2.c == result.c);

    for (int r = 0; r < m1.r; r++) {
        for (int c = 0; c < m2.c; c++) {
            result.matrix[r][c] = m1.matrix[r][c] * m2.matrix[r][c];
        }
    }
}

void matrix_add(mymatrix_t m1, mymatrix_t m2,
                mymatrix_t result) {
    assert(m1.r == m2.r && m1.c == m2.c);
    assert(m1.r == result.r && m2.c == result.c);

    for (int r = 0; r < result.r; r++) {
        for (int c = 0; c < result.c; c++) {
            result.matrix[r][c] = m1.matrix[r][c] + m2.matrix[r][c];
        }
    }
}

void matrix_add_row(mymatrix_t m1, unsigned int r1, mymatrix_t m2, unsigned int r2,
                    mymatrix_t result, unsigned int r_result) {
    assert(m1.c == m2.c && m1.c == result.c);
    assert(r1 < m1.r && r2 < m2.r);

    for (int c = 0; c < result.c; c++) {
        result.matrix[r_result][c] = m1.matrix[r1][c] + m2.matrix[r2][c];
    }
}

void matrix_sub(mymatrix_t m1, mymatrix_t m2,
                mymatrix_t result) {
    assert(m1.r == m2.r && m1.c == m2.c);
    assert(m1.r == result.r && m2.c == result.c);

    for (int r = 0; r < result.r; r++) {
        for (int c = 0; c < result.c; c++) {
            result.matrix[r][c] = m1.matrix[r][c] - m2.matrix[r][c];
        }
    }
}

void matrix_column_extend(mymatrix_t m, unsigned int factor,
                          mymatrix_t result) {
    assert(factor > 0);
    assert(m.r * factor == result.r && m.c == result.c);

    for (int copy = 0; copy < factor; copy++) {
        for (int r = 0; r < m.r; r++) {
            for (int c = 0; c < m.c; c++) {
                result.matrix[copy * m.r + r][c] = m.matrix[r][c];
            }
        }
    }
}

void matrix_row_extend(mymatrix_t m, unsigned int factor,
                       mymatrix_t result) {
    assert(factor > 0);
    assert(m.r == result.r && m.c * factor == result.c);

    for (int copy = 0; copy < factor; copy++) {
        for (int r = 0; r < m.r; r++) {
            for (int c = 0; c < m.c; c++) {
                result.matrix[r][copy * m.c] = m.matrix[r][c];
            }
        }
    }
}

void matrix_transpose(mymatrix_t m,
                      mymatrix_t result) {
    assert(m.r == result.c && m.c == result.r);

    for (int r = 0; r < result.r; r++) {
        for (int c = 0; c < result.c; c++) {
            result.matrix[r][c] = m.matrix[c][r];
        }
    }
}

bool matrix_equal(mymatrix_t m1, mymatrix_t m2) {
    if (m1.r != m2.r || m1.c != m2.c) {
        return false;
    }

    for (int r = 0; r < m1.r; r++) {
        for (int c = 0; c < m1.c; c++) {
            if (m1.matrix[r][c] != m2.matrix[r][c]) {
                return false;
            }
        }
    }
    return true;
}

void matrix_for_each_operator(mymatrix_t m, float (*op)(float),
                              mymatrix_t result) {
    assert(m.r == result.r && m.c == result.c);
    for (int r = 0; r < m.r; r++) {
        for (int c = 0; c < m.c; c++) {
            result.matrix[r][c] = op(m.matrix[r][c]);
        }
    }
}

void matrix_print(mymatrix_t m) {
    printf("%d x %d Matrix\n", m.r, m.c);
    for (int r = 0; r < m.r; r++) {
        printf("%-3d: ", r);
        for (int c = 0; c < m.c; c++) {
            printf("%-6f  ", (float)round(m.matrix[r][c] * 1000)/1000);
        }
        printf("\n");
    }
}

void matrix_set_values_to_fit(mymatrix_t m, float* elements, unsigned int num_elements) {
    assert(m.r * m.c == num_elements);
    int element_i = 0;
    for (int r = 0; r < m.r; r++) {
        for (int c = 0; c < m.c; c++) {
            m.matrix[r][c] = elements[element_i];
            element_i++;
        }
    }
}