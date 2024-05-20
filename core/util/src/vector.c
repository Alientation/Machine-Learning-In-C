#include <util/vector.h>

vector_t vector_constructor(unsigned int size, double* vector) {
    assert(size > 0);
    assert(vector != NULL);
    
    vector_t v = {
        .size = size, .vector = vector
    };
    return v;
}

matrix_t vector_to_matrix(vector_t vec) {
    matrix_t m = {
        .r = vec.size,
        .c = 1,
        .matrix = malloc(vec.size * sizeof(double))
    };

    for (int r = 0; r < m.r; r++) {
        m.matrix[r][0] = vec.vector[r];
    }
    return m;
}