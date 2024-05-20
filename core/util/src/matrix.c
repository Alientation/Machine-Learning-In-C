#include <util/matrix.h>

#include <stdlib.h>

matrix_t matrix_constructor(int r, int c, double** matrix) {
    assert(matrix != NULL);

    matrix_t m = {
        .r = r, .c = c, .matrix = matrix
    };
    return m;
}

// use https://en.wikipedia.org/wiki/Strassen_algorithm for large matrices
// todo
matrix_t matrix_multiply(matrix_t m1, matrix_t m2) {
    assert(m1.c == m2.r);
    assert(m1.matrix != NULL && m2.matrix != NULL);
}

matrix_t matrix_add(matrix_t m1, matrix_t m2) {
    assert(m1.r == m2.r);
    assert(m2.c == m2.c);
    assert(m1.matrix != NULL && m2.matrix != NULL);

    matrix_t new;
    new.r = m1.r;
    new.c = m1.c;
    new.matrix = malloc(m1.r * m1.c * sizeof(double));

    for (int r = 0; r < m1.r; r++) {
        for (int c = 0; c < m1.c; c++) {
            new.matrix[r][c] = m1.matrix[r][c] + m2.matrix[r][c];
        }
    }
    return new;
}

matrix_t matrix_sub(matrix_t m1, matrix_t m2) {
    assert(m1.r == m2.r);
    assert(m2.c == m2.c);
    assert(m1.matrix != NULL && m2.matrix != NULL);
    
    matrix_t new;
    new.r = m1.r;
    new.c = m1.c;
    new.matrix = malloc(new.r * new.c * sizeof(double));

    for (int r = 0; r < m1.r; r++) {
        for (int c = 0; c < m1.c; c++) {
            new.matrix[r][c] = m1.matrix[r][c] - m2.matrix[r][c];
        }
    }
    return new;
}

matrix_t matrix_column_extend(matrix_t m, unsigned int factor) {
    assert(m.matrix != NULL);
    assert(factor > 0);

    matrix_t new;
    new.r = m.r * factor;
    new.c = m.c;
    new.matrix = malloc(new.r * new.c * sizeof(double));
    for (int copy = 0; copy < factor; copy++) {
        for (int r = 0; r < m.r; r++) {
            for (int c = 0; c < m.c; c++) {
                new.matrix[copy * m.r + r][c] = m.matrix[r][c];
            }
        }
    }
    return new;
}

matrix_t matrix_row_extend(matrix_t m, unsigned int factor) {
    assert(m.matrix != NULL);
    assert(factor > 0);

    matrix_t new;
    new.r = m.r;
    new.c = m.c * factor;
    new.matrix = malloc(new.r * new.c * sizeof(double));
    for (int copy = 0; copy < factor; copy++) {
        for (int r = 0; r < m.r; r++) {
            for (int c = 0; c < m.c; c++) {
                new.matrix[r][copy * m.c] = m.matrix[r][c];
            }
        }
    }
    return new;
}

matrix_t matrix_transpose(matrix_t m) {
    matrix_t new = {
        .r = m.c,
        .c = m.r,
        .matrix = malloc(m.r * m.c * sizeof(double))
    };

    for (int r = 0; r < new.r; r++) {
        for (int c = 0; c < new.c; c++) {
            new.matrix[r][c] = m.matrix[c][r];
        }
    }
    return new;
}