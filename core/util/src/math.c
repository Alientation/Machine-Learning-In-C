#include <stdlib.h>
#include <math.h>

#include <util/math.h>



double random_normal_distribution_BoxMullerTransform(double standard_deviation) {
    // standard normal distribution with var(x) = 1
    double U1 = random_uniform_range(1);
    double U2 = random_uniform_range(1);
    double Z0 = sqrt(-2 * log10(U1) / log10(exp(1))) * cos(2 * acos(-1) * U2);
    
    // scale to correct standard deviation
    // Var(Y) = Var(aX) = a^2Var(X)
    // Var(X) = 1
    // Var(Y) = standard_deviation ^ 2
    // standard_deviation^2 = a^2
    // standard_deviation = a

    return Z0 / standard_deviation;
}

double random_uniform_range(double a) {
    return (double) rand() / (double) (RAND_MAX / a);
}