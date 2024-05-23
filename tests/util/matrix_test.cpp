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
    double a0[2] = {1,2};
    double *array1[1] = {a0};
    matrix_t *m1 = matrix_constructor(1, 2, (double**) array1);

    double a2[2] = {2, 1};
    double a3[2] = {3, 4};
    double *array2[2] = {a2, a3};
    matrix_t *m2 = matrix_constructor(2, 2, (double**) array2);

    matrix_t *m3 = matrix_allocator(1, 2);
    matrix_multiply(m1, m2, m3);
    free(m1);
    free(m2);

    EXPECT_EQ(m3->r, 1);
    EXPECT_EQ(m3->c, 2);

    EXPECT_EQ(m3->matrix[0][0], 8);
    EXPECT_EQ(m3->matrix[0][1], 9);

    matrix_free(m3);
}

TEST(matrix, matrix_multiply_scalar) {
    double a0[3] = {1,2,3};
    double a1[3] = {4,5,6};
    double *array1[2] = {a0, a1};
    matrix_t *m1 = matrix_constructor(2, 3, (double**) array1);
    matrix_t *m2 = matrix_allocator(2, 3);
    matrix_multiply_scalar(m1, 2, m2);

    free(m1);

    EXPECT_EQ(m2->r, 2);
    EXPECT_EQ(m2->c, 3);

    EXPECT_EQ(m2->matrix[0][0], 2);
    EXPECT_EQ(m2->matrix[0][1], 4);
    EXPECT_EQ(m2->matrix[0][2], 6);
    EXPECT_EQ(m2->matrix[1][0], 8);
    EXPECT_EQ(m2->matrix[1][1], 10);
    EXPECT_EQ(m2->matrix[1][2], 12);

    matrix_free(m2);
}

TEST(matrix, elementwise_multiply) {
    double a0[2] = {1,0};
    double a1[2] = {0,1};
    double *array1[2] = {a0, a1};
    matrix_t *m1 = matrix_constructor(2, 2, (double**) array1);

    double a2[2] = {2, 1};
    double a3[2] = {3, 4};
    double *array2[2] = {a2, a3};
    matrix_t *m2 = matrix_constructor(2, 2, (double**) array2);

    matrix_t *m3 = matrix_allocator(2, 2);
    matrix_elementwise_multiply(m1, m2, m3);
    free(m1);
    free(m2);

    EXPECT_EQ(m3->r, 2);
    EXPECT_EQ(m3->c, 2);

    EXPECT_EQ(m3->matrix[0][0], 2);
    EXPECT_EQ(m3->matrix[0][1], 0);
    EXPECT_EQ(m3->matrix[1][0], 0);
    EXPECT_EQ(m3->matrix[1][1], 4);

    matrix_free(m3);
}

TEST(matrix, add) {
    double a0[2] = {1,0};
    double a1[2] = {0,1};
    double *array1[2] = {a0, a1};
    matrix_t *m1 = matrix_constructor(2, 2, (double**) array1);

    double a2[2] = {2, 1};
    double a3[2] = {3, 4};
    double *array2[2] = {a2, a3};
    matrix_t *m2 = matrix_constructor(2, 2, (double**) array2);

    matrix_t *m3 = matrix_allocator(2, 2);
    matrix_add(m1, m2, m3);
    free(m1);
    free(m2);

    EXPECT_EQ(m3->r, 2);
    EXPECT_EQ(m3->c, 2);

    EXPECT_EQ(m3->matrix[0][0], 3);
    EXPECT_EQ(m3->matrix[0][1], 1);
    EXPECT_EQ(m3->matrix[1][0], 3);
    EXPECT_EQ(m3->matrix[1][1], 5);

    matrix_free(m3);
}

TEST(matrix, add_row) {
    double a0[2] = {1,0};
    double a1[2] = {0,1};
    double *array1[2] = {a0, a1};
    matrix_t *m1 = matrix_constructor(2, 2, (double**) array1);

    double a2[2] = {2, 1};
    double a3[2] = {3, 4};
    double *array2[2] = {a2, a3};
    matrix_t *m2 = matrix_constructor(2, 2, (double**) array2);

    matrix_t *m3 = matrix_allocator(2, 2);
    matrix_add_row(m1, 0, m2, 1, m3, 0);
    free(m1);
    free(m2);

    EXPECT_EQ(m3->r, 2);
    EXPECT_EQ(m3->c, 2);

    EXPECT_EQ(m3->matrix[0][0], 4);
    EXPECT_EQ(m3->matrix[0][1], 4);

    matrix_free(m3);
}

TEST(matrix, sub) {
    double a0[2] = {1,0};
    double a1[2] = {0,1};
    double *array1[2] = {a0, a1};
    matrix_t *m1 = matrix_constructor(2, 2, (double**) array1);

    double a2[2] = {2, 1};
    double a3[2] = {3, 4};
    double *array2[2] = {a2, a3};
    matrix_t *m2 = matrix_constructor(2, 2, (double**) array2);

    matrix_t *m3 = matrix_allocator(2, 2);
    matrix_sub(m1, m2, m3);
    free(m1);
    free(m2);

    EXPECT_EQ(m3->r, 2);
    EXPECT_EQ(m3->c, 2);

    EXPECT_EQ(m3->matrix[0][0], -1);
    EXPECT_EQ(m3->matrix[0][1], -1);
    EXPECT_EQ(m3->matrix[1][0], -3);
    EXPECT_EQ(m3->matrix[1][1], -3);

    matrix_free(m3);
}

TEST(matrix, column_extend) {
    // todo
}

TEST(matrix, row_extend) {
    // todo
}

TEST(matrix, transpose) {
    double a0[3] = {1,2,3};
    double a1[3] = {4,5,6};
    double *array1[2] = {a0, a1};
    matrix_t *m1 = matrix_constructor(2, 3, (double**) array1);

    matrix_t *m2 = matrix_allocator(3, 2);
    matrix_transpose(m1, m2);
    free(m1);

    EXPECT_EQ(m2->r, 3);
    EXPECT_EQ(m2->c, 2);

    EXPECT_EQ(m2->matrix[0][0], 1);
    EXPECT_EQ(m2->matrix[0][1], 4);
    EXPECT_EQ(m2->matrix[1][0], 2);
    EXPECT_EQ(m2->matrix[1][1], 5);
    EXPECT_EQ(m2->matrix[2][0], 3);
    EXPECT_EQ(m2->matrix[2][1], 6);

    matrix_free(m2);
}

TEST(matrix, equal) {
    double a0[2] = {2,1};
    double a1[2] = {3,4};
    double *array1[2] = {a0, a1};
    matrix_t *m1 = matrix_constructor(2, 2, (double**) array1);

    double a2[2] = {2, 1};
    double a3[2] = {3, 4};
    double *array2[2] = {a2, a3};
    matrix_t *m2 = matrix_constructor(2, 2, (double**) array2);

    EXPECT_EQ(matrix_equal(m1, m2), true);

    m1->matrix[0][0] = -2;
    EXPECT_EQ(matrix_equal(m1, m2), false);

    free(m1);
    free(m2);
}

TEST(matrix, for_each_operator) {
    // todo
}