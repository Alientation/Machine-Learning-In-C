#include <util/math.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>




float random_normal_distribution_BoxMullerTransform(float standard_deviation) {
    // standard normal distribution with var(x) = 1
    float U1 = random_uniform_range(1);
    float U2 = random_uniform_range(1);
    float Z0 = sqrt(-2 * log10(U1) / log10(exp(1))) * cos(2 * acos(-1) * U2);
    
    // scale to correct standard deviation
    // Var(Y) = Var(aX) = a^2Var(X)
    // Var(X) = 1
    // Var(Y) = standard_deviation ^ 2
    // standard_deviation^2 = a^2
    // standard_deviation = a
    Z0 *= standard_deviation;

    #if DEBUG
    printf("UNIFORM RANDOM: %f\n", (float) U1);
    printf("UNIFORM RANDOM: %f\n", (float) U2);

    printf("NORMAL DISTRIBUTED RANDOM: %f\n\n", (float) Z0);
    #endif
    
    return Z0;
}

float random_uniform_range(float a) {
    return (float) rand() / (float) (RAND_MAX / a);
}

float sigmoid(float z) {
    return 1. / (1 + exp(-z));
}

float relu(float z) {
    return fmax(0, z);
}

float sigmoid_prime(float z) {
    z = sigmoid(z);
    return z * (1-z);
}

float relu_prime(float z) {
    return z > 0;
}




// https://github.com/ekmett/approximate/
float fast_exp(float z) {
    union { float f; int x; } u;
    u.x = (int) (12102203 * z + 1064866805);
    return u.f;
}