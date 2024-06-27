#include <tests/matrix_test.h>

#define SHAPE(...) nshape_constructor(__VA_ARGS__)
TEST(nmatrix, nmatrix_allocator) {
    nmatrix_t m = nmatrix_allocator(SHAPE(2, 2, 3));

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

TEST(nmatrix, nmatrix_constructor) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, SHAPE(2, 2, 3));
    
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
    nmatrix_t m = nmatrix_constructor(6, a, SHAPE(2, 2, 3));

    nmatrix_reshape(&m, SHAPE(2, 3, 2));
    
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
    nmatrix_t m = nmatrix_constructor(6, a, SHAPE(2, 2, 3));

    EXPECT_TRUE(check_nmatrix_shape(&m, SHAPE(2, 2, 3)));
}

TEST(nmatrix, nmatrix_copy) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, SHAPE(2, 2, 3));
    nmatrix_t copy = nmatrix_copy(&m);
    
    EXPECT_TRUE(nmatrix_equal(&m, &copy));
}

TEST(nmatrix, nmatrix_memcpy) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, SHAPE(2, 2, 3));
    nmatrix_t copy = nmatrix_allocator(SHAPE(2, 2, 3));
    nmatrix_memcpy(&copy, &m);

    EXPECT_TRUE(nmatrix_equal(&copy, &m));
}

TEST(nmatrix, nmatrix_multiply) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, SHAPE(2, 2, 3));

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, SHAPE(2, 3, 2));

    float expected[4] = {5, 2, 32, 20};
    nmatrix_t exp = nmatrix_constructor(4, expected, SHAPE(2, 2, 2));

    nmatrix_t result = nmatrix_allocator(SHAPE(2, 2, 2));
    nmatrix_multiply(&m1, &m2, &result);
    
    nmatrix_print(&exp);
    nmatrix_print(&result);

    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_multiply_stacked) {
    float a1[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nmatrix_t m1 = nmatrix_constructor(12, a1, SHAPE(3, 2, 3, 2));

    float a2[12] = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(12, a2, SHAPE(3, 2, 2, 3));

    float expected[18] = {8,7,6,46,41,36,84,75,66,44,31,18,58,41,24,72,51,30};
    nmatrix_t exp = nmatrix_constructor(18, expected, SHAPE(3, 2, 3, 3));

    nmatrix_t result = nmatrix_allocator(SHAPE(3, 2, 3, 3));
    nmatrix_multiply(&m1, &m2, &result);

    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_multiply_scalar) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, SHAPE(2, 2, 3));

    float expected[6];
    for (int i = 0; i < 6; i++) {
        expected[i] = a[i] * 2;
    }
    nmatrix_t exp = nmatrix_constructor(6, expected, SHAPE(2, 2, 3));

    nmatrix_t result = nmatrix_allocator(SHAPE(2, 2, 3));
    nmatrix_multiply_scalar(&m, 2, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_elementwise_multiply) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, SHAPE(2, 2, 3));

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, SHAPE(2, 2, 3));

    float expected[6];
    for (int i = 0; i < 6; i++) {
        expected[i] = a1[i] * a2[i];
    }
    nmatrix_t exp = nmatrix_constructor(6, expected, SHAPE(2, 2, 3));

    nmatrix_t result = nmatrix_allocator(SHAPE(2, 2, 3));
    nmatrix_elementwise_multiply(&m1, &m2, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_add) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, SHAPE(2, 2, 3));

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, SHAPE(2, 2, 3));

    float expected[6];
    for (int i = 0; i < 6; i++) {
        expected[i] = a1[i] + a2[i];
    }
    nmatrix_t exp = nmatrix_constructor(6, expected, SHAPE(2, 2, 3));

    nmatrix_t result = nmatrix_allocator(SHAPE(2, 2, 3));
    nmatrix_add(&m1, &m2, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_sub) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, SHAPE(2, 2, 3));

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, SHAPE(2, 2, 3));

    float expected[6];
    for (int i = 0; i < 6; i++) {
        expected[i] = a1[i] - a2[i];
    }
    nmatrix_t exp = nmatrix_constructor(6, expected, SHAPE(2, 2, 3));

    nmatrix_t result = nmatrix_allocator(SHAPE(2, 2, 3));
    nmatrix_sub(&m1, &m2, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_transpose) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, SHAPE(2, 2, 3));

    float expected[6] = {0, 3, 1, 4, 2, 5};
    nmatrix_t exp = nmatrix_constructor(6, expected, SHAPE(2, 3, 2));

    nmatrix_t result = nmatrix_allocator(SHAPE(2, 3, 2));
    nmatrix_transpose(&m, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_transpose_3d) {
    float a[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m = nmatrix_constructor(6, a, SHAPE(3, 1, 2, 3));

    float expected[6] = {0, 3, 1, 4, 2, 5};
    nmatrix_t exp = nmatrix_constructor(6, expected, SHAPE(3, 3, 2, 1));

    nmatrix_t result = nmatrix_allocator(SHAPE(3, 3, 2, 1));
    nmatrix_transpose(&m, &result);
    
    EXPECT_TRUE(nmatrix_equal(&exp, &result));
}

TEST(nmatrix, nmatrix_equal_true) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, SHAPE(2, 2, 3));

    float a2[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m2 = nmatrix_constructor(6, a2, SHAPE(2, 2, 3));

    EXPECT_TRUE(nmatrix_equal(&m1, &m2));
}

TEST(nmatrix, nmatrix_nequal_ndims) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, SHAPE(2, 2, 3));

    float a2[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m2 = nmatrix_constructor(6, a2, SHAPE(3, 1, 2, 3));

    EXPECT_FALSE(nmatrix_equal(&m1, &m2));
}

TEST(nmatrix, nmatrix_nequal_dims) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, SHAPE(2, 2, 3));

    float a2[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m2 = nmatrix_constructor(6, a2, SHAPE(2, 3, 2));

    EXPECT_FALSE(nmatrix_equal(&m1, &m2));
}

TEST(nmatrix, nmatrix_nequal_elements) {
    float a1[6] = {0, 1, 2, 3, 4, 5};
    nmatrix_t m1 = nmatrix_constructor(6, a1, SHAPE(2, 2, 3));

    float a2[6] = {5, 4, 3, 2, 1, 0};
    nmatrix_t m2 = nmatrix_constructor(6, a2, SHAPE(2, 2, 3));

    EXPECT_FALSE(nmatrix_equal(&m1, &m2));
}

TEST(nmatrix, nmatrix_for_each_operator) {
    // todo
}