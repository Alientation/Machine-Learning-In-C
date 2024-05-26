#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <util/matrix.h>

#include <assert.h>
#include <memory.h>
#include <stdlib.h>


/**
 * Some Goals
 * 
 * line or circle detector
 * network visualizer
 * digit detector
 * batch/mini batch, stochastic gradient descent
 * add more variety of layers (convolution, pooling, reshaping)
 * add variable learning rates for each layer support
 * matrix broadcasting??? support for multi-dimension matrices?????
 */



typedef struct Layer layer_t;
typedef struct Model model_t;
typedef struct Input_Layer input_layer_t;
typedef struct Dense_Layer dense_layer_t;
typedef struct Activation_Layer activation_layer_t;
typedef struct Output_Layer output_layer_t;

// first layer of the model
typedef struct Input_Layer {
    matrix_t* (*feed_forward)(layer_t *this, matrix_t *input);
    matrix_t *input_values;
} input_layer_t;

// fully connected inner layer of the model
// n: number of neurons in this layer
// m: number of neurons in the previous layer
typedef struct Dense_Layer {
    matrix_t* (*feed_forward)(layer_t *this, matrix_t *input);
    matrix_t* (*back_propagation)(layer_t *this, matrix_t *input_gradient, double learning_rate);
    // n x 1, this layer's neurons which contain the "output" values
    matrix_t *activation_values;

    // W.X + b= Y
    // n x m
    // connecting the previous layer to this layer
    // the edges are the weights
    matrix_t *weights;
    matrix_t *transposed_weights; // to not have to reallocate memory
    matrix_t *transposed_inputs; 

    // n x 1
    matrix_t *bias;

    // dE/dX = W.T * dE/dY
    // m x 1
    matrix_t *d_cost_wrt_input;
    // same dimensions as weight and bias matrices
    matrix_t *d_cost_wrt_weight;
    matrix_t *d_cost_wrt_bias;
} dense_layer_t;

// passes in the activation values of the previous layer into the activation function
typedef struct Activation_Layer {
    matrix_t* (*feed_forward)(layer_t *this, matrix_t *input);
    matrix_t* (*back_propagation)(layer_t *this, matrix_t *input_gradient);
    matrix_t *activated_values;
    // dE/dX = W.T * dE/dY
    // m x 1
    matrix_t *d_cost_wrt_input;
} activation_layer_t;

// final layer, compute the cost and derivatives to initiate backprop
typedef struct Output_Layer {
    // void (*feed_forward)(void); // dummy variable to pad the back_prop function to the same location as other layer structs
    matrix_t* (*make_guess)(layer_t *this, matrix_t *output);
    matrix_t* (*back_propagation)(layer_t *this, matrix_t *expected_output);
    matrix_t *output_values;
    // dE/dX = W.T * dE/dY
    // m x 1
    matrix_t *d_cost_wrt_input;
    matrix_t *guess;
} output_layer_t;

// unionized all layers under a common identity
typedef struct Layer {
    enum {
        INPUT,
        DENSE,
        ACTIVATION,
        OUTPUT
    } type;

    union {
        input_layer_t input;
        dense_layer_t dense;
        activation_layer_t activation;
        output_layer_t output;
    } layer;

    // doubly linked
    layer_t *next;
    layer_t *prev;
} layer_t;

// nn model
typedef struct Model {
    unsigned int num_layers;
    layer_t *input_layer; // first layer
    layer_t *output_layer; // last layer
} model_t;

matrix_t *input_feed_forward(layer_t *this, matrix_t *input); 

matrix_t *dense_feed_forward(layer_t *this, matrix_t *input);
matrix_t *dense_back_propagation(layer_t *this, matrix_t *d_error_wrt_output, double learning_rate);

matrix_t *activation_feed_forward_sigmoid(layer_t *this, matrix_t *input);
matrix_t *activation_feed_forward_relu(layer_t *this, matrix_t *input);
// matrix_t *activation_feed_forward_softmax(layer_t *this, matrix_t *input); //todo
// todo tanh
matrix_t *activation_back_propagation_sigmoid(layer_t *this, matrix_t *d_cost_wrt_output);
matrix_t *activation_back_propagation_relu(layer_t *this, matrix_t *d_cost_wrt_output);
// matrix_t *activation_back_propagation_softmax(layer_t *this, matrix_t *input); //todo
// todo tanh back prop

matrix_t *output_make_guess_one_hot_encoded(layer_t *this, matrix_t *output);
matrix_t *output_make_guess_passforward(layer_t *this, matrix_t *output);
matrix_t *output_make_guess_round(layer_t *this, matrix_t *output);
matrix_t *output_make_guess_softmax(layer_t *this, matrix_t *output);
matrix_t *output_back_propagation_mean_squared(layer_t *this, matrix_t *expected_output);
matrix_t *output_back_propagation_cross_entropy(layer_t *this, matrix_t *expected_output);

double output_cost_mean_squared(layer_t *this, matrix_t *expected_output);
double output_cost_cross_entropy(layer_t *this, matrix_t *expected_output);

// frees allocated memory for the layer
void layer_free(layer_t *layer);
matrix_t* layer_get_neurons(layer_t *layer);

void model_free(model_t *model);
void model_add_layer(model_t *model, layer_t *layer);

// adds an layers to the model
// todo in future, specify dimensions instead of supply matrix to be then copied
layer_t* layer_input(model_t *model, matrix_t *input);
layer_t* layer_dense(model_t *model, matrix_t *neurons);
layer_t* layer_activation(model_t *model, matrix_t* (*feed_forward)(layer_t*, matrix_t*), matrix_t* (*back_propagation)(layer_t*, matrix_t*));
layer_t* layer_output(model_t *model, matrix_t* (*make_guess)(layer_t*, matrix_t*), matrix_t* (*back_propagation)(layer_t*, matrix_t*));


matrix_t* model_predict(model_t *model, matrix_t *input, 
               matrix_t *output);

void model_initialize_matrix_normal_distribution(matrix_t *model, double mean, double standard_deviation);
void model_back_propagate(model_t *model, matrix_t *expected_output, double learning_rate);
double model_train(model_t *model, matrix_t **inputs, matrix_t **expected_outputs, unsigned int num_examples, double learning_rate);
void model_test(model_t *model, matrix_t **inputs, matrix_t **expected_outputs, unsigned int num_tests);


#endif