#pragma once
#ifndef VECTOR_H
#define VECTOR_H

#include <util/matrix.h>

#include <stdlib.h>

typedef struct Vector {
    unsigned int size;
    double* vector;
} vector_t;

vector_t vector_constructor(unsigned int size, double* vector); 

matrix_t vector_to_matrix(vector_t vec);


#endif