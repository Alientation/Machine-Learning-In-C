#include <util/matrix.h>

#include <assert.h>
#include <stdlib.h>

matrix_t matrix_constructor(int r, int c, double** matrix) {
    assert(matrix != NULL);

    matrix_t m = {
        .r = r, .c = c, .matrix = matrix
    };
    return m;
}

void matrix_free(matrix_t *matrix) {
    if (matrix->matrix == NULL) {
        return;
    }

    for (int r = 0; r < matrix->r; r++) {
        free(matrix->matrix[r]);
    }
    free(matrix->matrix);
}

// use https://en.wikipedia.org/wiki/Strassen_algorithm for large matrices
// todo
void matrix_multiply(matrix_t *m1, matrix_t *m2,
                     matrix_t *result) {
    assert(m1->c == m2->r);
    assert(m1->matrix != NULL && m2->matrix != NULL);
    assert(m1->r == result->r && result->c == result->c);

    for (int r = 0; r < result->r; r++) {
        for (int c = 0; c < result->c; c++) {
            double dot = 0;
            for (int i = 0; i < m1->c; i++) {
                dot += m1->matrix[r][c+i] * m2->matrix[r+i][c];
            }
            result->matrix[r][c] = dot;
        }
    }
}

void matrix_add(matrix_t *m1, matrix_t *m2,
                matrix_t *result) {
    assert(m1->r == m2->r && m1->c == m2->c);
    assert(m1->matrix != NULL && m2->matrix != NULL);
    assert(m1->r == result->r && m2->c == result->c);

    for (int r = 0; r < result->r; r++) {
        for (int c = 0; c < result->c; c++) {
            result->matrix[r][c] = m1->matrix[r][c] + m2->matrix[r][c];
        }
    }
}

void matrix_add_row(matrix_t *m1, unsigned int r1, matrix_t *m2, unsigned int r2,
                    matrix_t *result, unsigned int r_result) {
    assert(m1->c == m2->c && m1->c == result->c);
    assert(r1 < m1->r && r2 < m2->r);
    assert(m1->matrix != NULL && m2->matrix != NULL);

    for (int c = 0; c < result->c; c++) {
        result->matrix[r_result][c] = m1->matrix[r1][c] + m2->matrix[r2][c];
    }
}

void matrix_sub(matrix_t *m1, matrix_t *m2,
                matrix_t *result) {
    assert(m1->r == m2->r && m1->c == m2->c);
    assert(m1->matrix != NULL && m2->matrix != NULL);
    assert(m1->r == result->r && m2->c == result->c);

    for (int r = 0; r < result->r; r++) {
        for (int c = 0; c < result->c; c++) {
            result->matrix[r][c] = m1->matrix[r][c] - m2->matrix[r][c];
        }
    }
}

void matrix_column_extend(matrix_t *m, unsigned int factor,
                          matrix_t *result) {
    assert(m->matrix != NULL);
    assert(factor > 0);
    assert(m->r * factor == result->r && m->c == result->c);

    for (int copy = 0; copy < factor; copy++) {
        for (int r = 0; r < m->r; r++) {
            for (int c = 0; c < m->c; c++) {
                result->matrix[copy * m->r + r][c] = m->matrix[r][c];
            }
        }
    }
}

void matrix_row_extend(matrix_t *m, unsigned int factor,
                       matrix_t *result) {
    assert(m->matrix != NULL);
    assert(factor > 0);
    assert(m->r == result->r && m->c * factor == result->c);

    for (int copy = 0; copy < factor; copy++) {
        for (int r = 0; r < m->r; r++) {
            for (int c = 0; c < m->c; c++) {
                result->matrix[r][copy * m->c] = m->matrix[r][c];
            }
        }
    }
}

void matrix_transpose(matrix_t *m,
                      matrix_t *result) {
    assert(m->r == result->c && m->c == result->r);

    for (int r = 0; r < result->r; r++) {
        for (int c = 0; c < result->c; c++) {
            result->matrix[r][c] = m->matrix[c][r];
        }
    }
}