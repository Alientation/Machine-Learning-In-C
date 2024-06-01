#include <tests/matrix_test.h>

TEST(matrix, matrix_allocator) {
    mymatrix_t m = matrix_allocator(2,2);

    EXPECT_NE(m.matrix, nullptr);
    EXPECT_EQ(m.r, 2);
    EXPECT_EQ(m.c, 2);
    matrix_free(m);
}

TEST(matrix, matrix_constructor) {
    float a0[2] = {1,0};
    float a1[2] = {0,1};
    float *array[2] = {a0, a1};
    mymatrix_t m = matrix_constructor(2, 2, (float**) array);

    EXPECT_EQ(m.r, 2);
    EXPECT_EQ(m.c, 2);

    EXPECT_EQ(m.matrix[0][0], 1);
    EXPECT_EQ(m.matrix[0][1], 0);
    EXPECT_EQ(m.matrix[1][0], 0);
    EXPECT_EQ(m.matrix[1][1], 1);
}

TEST(matrix, matrix_copy) {
    float a0[2] = {1,2};
    float a1[2] = {2,1};
    float *array[2] = {a0, a1};
    mymatrix_t m = matrix_constructor(2, 2, (float**) array);
    mymatrix_t copy = matrix_copy(m);

    EXPECT_EQ(copy.r, 2);
    EXPECT_EQ(copy.c, 2);

    EXPECT_EQ(copy.matrix[0][0], 1);
    EXPECT_EQ(copy.matrix[0][1], 2);
    EXPECT_EQ(copy.matrix[1][0], 2);
    EXPECT_EQ(copy.matrix[1][1], 1);   

    matrix_free(copy);
}

TEST(matrix, matrix_memcpy) {
    float a0[2] = {1,2};
    float a1[2] = {2,1};
    float *array[2] = {a0, a1};
    mymatrix_t m = matrix_constructor(2, 2, (float**) array);
    mymatrix_t copy = matrix_allocator(2, 2);
    matrix_memcpy(copy, m);

    EXPECT_EQ(copy.r, 2);
    EXPECT_EQ(copy.c, 2);

    EXPECT_EQ(copy.matrix[0][0], 1);
    EXPECT_EQ(copy.matrix[0][1], 2);
    EXPECT_EQ(copy.matrix[1][0], 2);
    EXPECT_EQ(copy.matrix[1][1], 1);

    matrix_free(copy);
}

TEST(matrix, matrix_multiply_identity) {
    float a0[2] = {1,0};
    float a1[2] = {0,1};
    float *array1[2] = {a0, a1};
    mymatrix_t m1 = matrix_constructor(2, 2, (float**) array1);

    float a2[2] = {2, 1};
    float a3[2] = {3, 4};
    float *array2[2] = {a2, a3};
    mymatrix_t m2 = matrix_constructor(2, 2, (float**) array2);

    mymatrix_t m3 = matrix_allocator(2, 2);
    matrix_multiply(m1, m2, m3);

    EXPECT_EQ(m3.r, 2);
    EXPECT_EQ(m3.c, 2);

    EXPECT_EQ(m3.matrix[0][0], 2);
    EXPECT_EQ(m3.matrix[0][1], 1);
    EXPECT_EQ(m3.matrix[1][0], 3);
    EXPECT_EQ(m3.matrix[1][1], 4);

    matrix_free(m3);
}

TEST(matrix, matrix_multiply) {
    float a0[2] = {1,2};
    float *array1[1] = {a0};
    mymatrix_t m1 = matrix_constructor(1, 2, (float**) array1);

    float a2[2] = {2, 1};
    float a3[2] = {3, 4};
    float *array2[2] = {a2, a3};
    mymatrix_t m2 = matrix_constructor(2, 2, (float**) array2);

    mymatrix_t m3 = matrix_allocator(1, 2);
    matrix_multiply(m1, m2, m3);

    EXPECT_EQ(m3.r, 1);
    EXPECT_EQ(m3.c, 2);

    EXPECT_EQ(m3.matrix[0][0], 8);
    EXPECT_EQ(m3.matrix[0][1], 9);

    matrix_free(m3);
}

TEST(matrix, matrix_multiply_scalar) {
    float a0[3] = {1,2,3};
    float a1[3] = {4,5,6};
    float *array1[2] = {a0, a1};
    mymatrix_t m1 = matrix_constructor(2, 3, (float**) array1);
    mymatrix_t m2 = matrix_allocator(2, 3);
    matrix_multiply_scalar(m1, 2, m2);

    EXPECT_EQ(m2.r, 2);
    EXPECT_EQ(m2.c, 3);

    EXPECT_EQ(m2.matrix[0][0], 2);
    EXPECT_EQ(m2.matrix[0][1], 4);
    EXPECT_EQ(m2.matrix[0][2], 6);
    EXPECT_EQ(m2.matrix[1][0], 8);
    EXPECT_EQ(m2.matrix[1][1], 10);
    EXPECT_EQ(m2.matrix[1][2], 12);

    matrix_free(m2);
}

TEST(matrix, elementwise_multiply) {
    float a0[2] = {1,0};
    float a1[2] = {0,1};
    float *array1[2] = {a0, a1};
    mymatrix_t m1 = matrix_constructor(2, 2, (float**) array1);

    float a2[2] = {2, 1};
    float a3[2] = {3, 4};
    float *array2[2] = {a2, a3};
    mymatrix_t m2 = matrix_constructor(2, 2, (float**) array2);

    mymatrix_t m3 = matrix_allocator(2, 2);
    matrix_elementwise_multiply(m1, m2, m3);

    EXPECT_EQ(m3.r, 2);
    EXPECT_EQ(m3.c, 2);

    EXPECT_EQ(m3.matrix[0][0], 2);
    EXPECT_EQ(m3.matrix[0][1], 0);
    EXPECT_EQ(m3.matrix[1][0], 0);
    EXPECT_EQ(m3.matrix[1][1], 4);

    matrix_free(m3);
}

TEST(matrix, add) {
    float a0[2] = {1,0};
    float a1[2] = {0,1};
    float *array1[2] = {a0, a1};
    mymatrix_t m1 = matrix_constructor(2, 2, (float**) array1);

    float a2[2] = {2, 1};
    float a3[2] = {3, 4};
    float *array2[2] = {a2, a3};
    mymatrix_t m2 = matrix_constructor(2, 2, (float**) array2);

    mymatrix_t m3 = matrix_allocator(2, 2);
    matrix_add(m1, m2, m3);

    EXPECT_EQ(m3.r, 2);
    EXPECT_EQ(m3.c, 2);

    EXPECT_EQ(m3.matrix[0][0], 3);
    EXPECT_EQ(m3.matrix[0][1], 1);
    EXPECT_EQ(m3.matrix[1][0], 3);
    EXPECT_EQ(m3.matrix[1][1], 5);

    matrix_free(m3);
}

TEST(matrix, add_row) {
    float a0[2] = {1,0};
    float a1[2] = {0,1};
    float *array1[2] = {a0, a1};
    mymatrix_t m1 = matrix_constructor(2, 2, (float**) array1);

    float a2[2] = {2, 1};
    float a3[2] = {3, 4};
    float *array2[2] = {a2, a3};
    mymatrix_t m2 = matrix_constructor(2, 2, (float**) array2);

    mymatrix_t m3 = matrix_allocator(2, 2);
    matrix_add_row(m1, 0, m2, 1, m3, 0);

    EXPECT_EQ(m3.r, 2);
    EXPECT_EQ(m3.c, 2);

    EXPECT_EQ(m3.matrix[0][0], 4);
    EXPECT_EQ(m3.matrix[0][1], 4);

    matrix_free(m3);
}

TEST(matrix, sub) {
    float a0[2] = {1,0};
    float a1[2] = {0,1};
    float *array1[2] = {a0, a1};
    mymatrix_t m1 = matrix_constructor(2, 2, (float**) array1);

    float a2[2] = {2, 1};
    float a3[2] = {3, 4};
    float *array2[2] = {a2, a3};
    mymatrix_t m2 = matrix_constructor(2, 2, (float**) array2);

    mymatrix_t m3 = matrix_allocator(2, 2);
    matrix_sub(m1, m2, m3);

    EXPECT_EQ(m3.r, 2);
    EXPECT_EQ(m3.c, 2);

    EXPECT_EQ(m3.matrix[0][0], -1);
    EXPECT_EQ(m3.matrix[0][1], -1);
    EXPECT_EQ(m3.matrix[1][0], -3);
    EXPECT_EQ(m3.matrix[1][1], -3);

    matrix_free(m3);
}

TEST(matrix, column_extend) {
    // todo
}

TEST(matrix, row_extend) {
    // todo
}

TEST(matrix, transpose) {
    float a0[3] = {1,2,3};
    float a1[3] = {4,5,6};
    float *array1[2] = {a0, a1};
    mymatrix_t m1 = matrix_constructor(2, 3, (float**) array1);

    mymatrix_t m2 = matrix_allocator(3, 2);
    matrix_transpose(m1, m2);

    EXPECT_EQ(m2.r, 3);
    EXPECT_EQ(m2.c, 2);

    EXPECT_EQ(m2.matrix[0][0], 1);
    EXPECT_EQ(m2.matrix[0][1], 4);
    EXPECT_EQ(m2.matrix[1][0], 2);
    EXPECT_EQ(m2.matrix[1][1], 5);
    EXPECT_EQ(m2.matrix[2][0], 3);
    EXPECT_EQ(m2.matrix[2][1], 6);

    matrix_free(m2);
}

TEST(matrix, equal) {
    float a0[2] = {2,1};
    float a1[2] = {3,4};
    float *array1[2] = {a0, a1};
    mymatrix_t m1 = matrix_constructor(2, 2, (float**) array1);

    float a2[2] = {2, 1};
    float a3[2] = {3, 4};
    float *array2[2] = {a2, a3};
    mymatrix_t m2 = matrix_constructor(2, 2, (float**) array2);

    EXPECT_EQ(matrix_equal(m1, m2), true);

    m1.matrix[0][0] = -2;
    EXPECT_EQ(matrix_equal(m1, m2), false);
}

TEST(matrix, for_each_operator) {
    // todo
}