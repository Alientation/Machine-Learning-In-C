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
    double array[2][2] = {{1,0},{0,1}};
    matrix_t *m = matrix_constructor(2, 2, (double**) array);

    EXPECT_NE(m, nullptr);
    EXPECT_NE(m->matrix, nullptr);
    EXPECT_EQ(m->r, 2);
    EXPECT_EQ(m->c, 2);

    free(m);
    m = NULL;
}