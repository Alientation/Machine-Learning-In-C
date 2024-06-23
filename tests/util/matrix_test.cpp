#include <tests/matrix_test.h>

TEST(matrix, matrix_allocator) {
    mymatrix_t m = matrix_allocator(2,2);

    EXPECT_NE(m.matrix, nullptr);
    EXPECT_EQ(m.r, 2);
    EXPECT_EQ(m.c, 2);
    matrix_free(m);
}

TEST(nmatrix, nmatrix_allocator) {
    nmatrix_t m = nmatrix_allocator(2, 2, 3);

    EXPECT_NE(m.matrix, nullptr);
    EXPECT_EQ(m.n_dims, 2);
    EXPECT_EQ(m.dims[0], 2);
    EXPECT_EQ(m.dims[1], 3);
    EXPECT_EQ(m.n_elements, 6);
    for (int i = 0; i < m.n_elements; i++) {
        EXPECT_EQ(m.matrix[i], 0.0);
    }
    nmatrix_free(&m);
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

TEST(nmatrix, nmatrix_constructor) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, 2, 2, 3);
    
    EXPECT_EQ(m.n_dims, 2);
    EXPECT_EQ(m.dims[0], 2);
    EXPECT_EQ(m.dims[1], 3);
    EXPECT_EQ(m.n_elements, 6);
    
    EXPECT_EQ(m.matrix[0], 0.);
    EXPECT_EQ(m.matrix[1], 1.);
    EXPECT_EQ(m.matrix[2], 2.);
    EXPECT_EQ(m.matrix[3], 3.);
    EXPECT_EQ(m.matrix[4], 4.);
    EXPECT_EQ(m.matrix[5], 5.);
}

TEST(nmatrix, nmatrix_reshape) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, 2, 2, 3);

    nmatrix_reshape(&m, 2, 3, 2);
    
    EXPECT_EQ(m.n_elements, 6);
    EXPECT_EQ(m.n_dims, 2);
    EXPECT_EQ(m.dims[0], 3);
    EXPECT_EQ(m.dims[1], 2);

    EXPECT_EQ(m.matrix[0], 0.);
    EXPECT_EQ(m.matrix[1], 1.);
    EXPECT_EQ(m.matrix[2], 2.);
    EXPECT_EQ(m.matrix[3], 3.);
    EXPECT_EQ(m.matrix[4], 4.);
    EXPECT_EQ(m.matrix[5], 5.);
}

TEST(nmatrix, check_nmatrix_shape) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, 2, 2, 3);

    EXPECT_TRUE(check_nmatrix_shape(&m, 2, 2, 3));
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

TEST(nmatrix, nmatrix_copy) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, 2, 2, 3);
    nmatrix_t copy = nmatrix_copy(&m);
    
    EXPECT_TRUE(nmatrix_equal(&m, &copy));
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

TEST(nmatrix, nmatrix_memcpy) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, 2, 2, 3);
    nmatrix_t copy = nmatrix_allocator(2, 2, 3);
    nmatrix_memcpy(&copy, &m);

    EXPECT_TRUE(nmatrix_equal(&copy, &m));
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

TEST(nmatrix, nmatrix_multiply) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, 2, 2, 3);

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, 2, 3, 2);

    float expected[4] = {5, 2, 32, 20};
    nmatrix_t exp = nmatrix_constructor(4, expected, 2, 2, 2);

    nmatrix_t result = nmatrix_allocator(2, 2, 2);
    nmatrix_multiply(&m1, &m2, &result);
    
    nmatrix_print(&exp);
    nmatrix_print(&result);

    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_multiply_stacked) {
    float a1[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nmatrix_t m1 = nmatrix_constructor(12, a1, 3, 2, 3, 2);

    float a2[12] = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(12, a2, 3, 2, 2, 3);

    float expected[18] = {8,7,6,46,41,36,84,75,66,44,31,18,58,41,24,72,51,30};
    nmatrix_t exp = nmatrix_constructor(18, expected, 3, 2, 3, 3);

    nmatrix_t result = nmatrix_allocator(3, 2, 3, 3);
    nmatrix_multiply(&m1, &m2, &result);

    EXPECT_TRUE(nmatrix_equal(&exp, &result));
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

TEST(nmatrix, nmatrix_multiply_scalar) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, 2, 2, 3);

    float expected[6];
    for (int i = 0; i < 6; i++) {
        expected[i] = a[i] * 2;
    }
    nmatrix_t exp = nmatrix_constructor(6, expected, 2, 2, 3);

    nmatrix_t result = nmatrix_allocator(2, 2, 3);
    nmatrix_multiply_scalar(&m, 2, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(matrix, matrix_elementwise_multiply) {
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

TEST(nmatrix, nmatrix_elementwise_multiply) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, 2, 2, 3);

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, 2, 2, 3);

    float expected[6];
    for (int i = 0; i < 6; i++) {
        expected[i] = a1[i] * a2[i];
    }
    nmatrix_t exp = nmatrix_constructor(6, expected, 2, 2, 3);

    nmatrix_t result = nmatrix_allocator(2, 2, 3);
    nmatrix_elementwise_multiply(&m1, &m2, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(matrix, matrix_add) {
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

TEST(nmatrix, nmatrix_add) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, 2, 2, 3);

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, 2, 2, 3);

    float expected[6];
    for (int i = 0; i < 6; i++) {
        expected[i] = a1[i] + a2[i];
    }
    nmatrix_t exp = nmatrix_constructor(6, expected, 2, 2, 3);

    nmatrix_t result = nmatrix_allocator(2, 2, 3);
    nmatrix_add(&m1, &m2, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(matrix, matrix_add_row) {
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

TEST(matrix, matrix_sub) {
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

TEST(nmatrix, nmatrix_sub) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, 2, 2, 3);

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, 2, 2, 3);

    float expected[6];
    for (int i = 0; i < 6; i++) {
        expected[i] = a1[i] - a2[i];
    }
    nmatrix_t exp = nmatrix_constructor(6, expected, 2, 2, 3);

    nmatrix_t result = nmatrix_allocator(2, 2, 3);
    nmatrix_sub(&m1, &m2, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(matrix, matrix_column_extend) {
    // todo
}

TEST(matrix, matrix_row_extend) {
    // todo
}

TEST(matrix, matrix_transpose) {
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

TEST(nmatrix, nmatrix_transpose) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, 2, 2, 3);

    float expected[6] = {0, 3, 1, 4, 2, 5};
    nmatrix_t exp = nmatrix_constructor(6, expected, 2, 3, 2);

    nmatrix_t result = nmatrix_allocator(2, 3, 2);
    nmatrix_transpose(&m, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_transpose_3d) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, 3, 1, 2, 3);

    float expected[6] = {0, 3, 1, 4, 2, 5};
    nmatrix_t exp = nmatrix_constructor(6, expected, 3, 3, 2, 1);

    nmatrix_t result = nmatrix_allocator(3, 3, 2, 1);
    nmatrix_transpose(&m, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(matrix, matrix_equal) {
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

TEST(nmatrix, nmatrix_equal_true) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, 2, 2, 3);

    float a2[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m2 = nmatrix_constructor(6, a2, 2, 2, 3);

    EXPECT_TRUE(nmatrix_equal(&m1, &m2));
}

TEST(nmatrix, nmatrix_nequal_ndims) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, 2, 2, 3);

    float a2[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m2 = nmatrix_constructor(6, a2, 3, 1, 2, 3);

    EXPECT_FALSE(nmatrix_equal(&m1, &m2));
}

TEST(nmatrix, nmatrix_nequal_dims) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, 2, 2, 3);

    float a2[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m2 = nmatrix_constructor(6, a2, 2, 3, 2);

    EXPECT_FALSE(nmatrix_equal(&m1, &m2));
}

TEST(nmatrix, nmatrix_nequal_elements) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, 2, 2, 3);

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, 2, 2, 3);

    EXPECT_FALSE(nmatrix_equal(&m1, &m2));
}

TEST(matrix, matrix_for_each_operator) {
    // todo
}

TEST(nmatrix, nmatrix_for_each_operator) {
    // todo
}