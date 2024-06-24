#pragma once
#ifndef MY_MATH_H
#define MY_MATH_H

// random float precision floating point number following the normal distribution ()
// using the Box Muller Transform algorithm
float random_normal_distribution_BoxMullerTransform(float standard_deviation);

// random float precision floating point number from 0 to a_max
float random_uniform_range(float a_max);


float sigmoid(float z);
float relu(float z);
float sigmoid_prime(float z);
float relu_prime(float z);

float fast_exp(float z);

#endif // MY_MATH_H