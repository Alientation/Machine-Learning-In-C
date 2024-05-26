#pragma once
#ifndef MY_MATH_H
#define MY_MATH_H

// random double precision floating point number following the normal distribution ()
// using the Box Muller Transform algorithm
double random_normal_distribution_BoxMullerTransform(double standard_deviation);

// random double precision floating point number from 0 to a_max
double random_uniform_range(double a_max);


double sigmoid(double z);
double relu(double z);
double sigmoid_prime(double z);
double relu_prime(double z);


#endif // MY_MATH_H