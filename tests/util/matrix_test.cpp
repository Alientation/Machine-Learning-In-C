#include <tests/matrix_test.h>

TEST(matrix, matrix_allocator) {
    matrix_t *m = matrix_allocator(2,2);

    EXPECT_NE(m, nullptr);
    EXPECT_NE(m->matrix, nullptr);
    EXPECT_EQ(m->r, 2);
    EXPECT_EQ(m->c, 2);
    matrix_free(m);
    m = NULL;
}

TEST(matrix, matrix_constructor) {
    double a0[2] = {1,0};
    double a1[2] = {0,1};
    double *array[2] = {a0, a1};
    matrix_t *m = matrix_constructor(2, 2, (double**) array);

    EXPECT_EQ(m->r, 2);
    EXPECT_EQ(m->c, 2);

    EXPECT_EQ(m->matrix[0][0], 1);
    EXPECT_EQ(m->matrix[0][1], 0);
    EXPECT_EQ(m->matrix[1][0], 0);
    EXPECT_EQ(m->matrix[1][1], 1);

    free(m);
    m = NULL;
}

TEST(matrix, matrix_copy) {
    double a0[2] = {1,2};
    double a1[2] = {2,1};
    double *array[2] = {a0, a1};
    matrix_t *m = matrix_constructor(2, 2, (double**) array);
    matrix_t *copy = matrix_copy(m);

    free(m);
    m = NULL;

    EXPECT_EQ(copy->r, 2);
    EXPECT_EQ(copy->c, 2);

    EXPECT_EQ(copy->matrix[0][0], 1);
    EXPECT_EQ(copy->matrix[0][1], 2);
    EXPECT_EQ(copy->matrix[1][0], 2);
    EXPECT_EQ(copy->matrix[1][1], 1);   

    free(copy);
    copy = NULL;
}

TEST(matrix, matrix_memcpy) {
    double a0[2] = {1,2};
    double a1[2] = {2,1};
    double *array[2] = {a0, a1};
    matrix_t *m = matrix_constructor(2, 2, (double**) array);
    matrix_t *copy = matrix_allocator(2, 2);
    matrix_memcpy(copy, m);
    free(m);
    m = NULL;

    EXPECT_EQ(copy->r, 2);
    EXPECT_EQ(copy->c, 2);

    EXPECT_EQ(copy->matrix[0][0], 1);
    EXPECT_EQ(copy->matrix[0][1], 2);
    EXPECT_EQ(copy->matrix[1][0], 2);
    EXPECT_EQ(copy->matrix[1][1], 1);

    matrix_free(copy);
    copy = NULL;
}

TEST(matrix, matrix_multiply_identity) {
    double a0[2] = {1,0};
    double a1[2] = {0,1};
    double *array1[2] = {a0, a1};
    matrix_t *m1 = matrix_constructor(2, 2, (double**) array1);

    double a2[2] = {2, 1};
    double a3[2] = {3, 4};
    double *array2[2] = {a2, a3};
    matrix_t *m2 = matrix_constructor(2, 2, (double**) array2);

    matrix_t *m3 = matrix_allocator(2, 2);
    matrix_multiply(m1, m2, m3);
    free(m1);
    free(m2);

    EXPECT_EQ(m3->r, 2);
    EXPECT_EQ(m3->c, 2);

    EXPECT_EQ(m3->matrix[0][0], 2);
    EXPECT_EQ(m3->matrix[0][1], 1);
    EXPECT_EQ(m3->matrix[1][0], 3);
    EXPECT_EQ(m3->matrix[1][1], 4);

    matrix_free(m3);
}

TEST(matrix, matrix_multiply) {
    // todo
}