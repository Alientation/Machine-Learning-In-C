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
 * Add separate dropout layer, the current way dropout works is by modifying the input into a dense layer, but since the input matrix is
 * likely the previous layer's output matrix, that will interfere with backpropagation which would explain the very low accuracy and the common fluxtuations it has
 * 
 * weight regularlization techniques (L2/L1)
 * weight clipping?
 * add momentum (remembers previous gradients)
 * layer/batch normalization?
 * weights should probably be normalized
 * save model to file
 * load model from file  *so we can pretrain models
 * batch/mini batch, full gradient descent (apparently we are using stochastic gradient descent)
 * add more variety of layers (convolution, pooling, reshaping)
 * add variable learning rates for each layer support
 * matrix broadcasting??? support for multi-dimension matrices?????
 * 
 * 
 * Performance History
 * 6/23/2024 - prior to nmatrix refactor, 3200 28x28 examples per second for 12k test/train size (0.8 split)
 * 6/23/2024 - after nmatrix refactor, same test results in 2800 examples per second, likely because the size of the nmatrix struct is large and passing
 *              them around without using references is slow
 * 6/23/2024 - temporarily limiting nmatrix dimensions to 2 results in 2850 examples per second.. so that was not the problem
 */



typedef struct Layer layer_t;
typedef struct NeuralNetworkModel neural_network_model_t;
typedef struct Input_Layer input_layer_t;
typedef struct Dense_Layer dense_layer_t;
typedef struct Activation_Layer activation_layer_t;
typedef struct Output_Layer output_layer_t;

typedef struct Layer_Function {
    nmatrix_t (*feed_forward)(layer_t *this, nmatrix_t input);
    nmatrix_t (*back_propagation)(layer_t *this, nmatrix_t input_gradient, float learning_rate);
} layer_function_t;

extern const layer_function_t input_functions;
extern const layer_function_t dense_functions;
extern const layer_function_t dropout_functions;
extern const layer_function_t activation_functions_sigmoid;
extern const layer_function_t activation_functions_relu;
extern const layer_function_t activation_functions_softmax;
extern const layer_function_t output_functions_meansquared;
extern const layer_function_t output_functions_crossentropy;

// first layer of the model
typedef struct Input_Layer {
    layer_function_t functions;
    nmatrix_t input_values;
    neural_network_model_t *model;
} input_layer_t;

// fully connected inner layer of the model
// n: number of neurons in this layer
// m: number of neurons in the previous layer
typedef struct Dense_Layer {
    layer_function_t functions;
    // n x 1, this layer's neurons which contain the "output" values
    nmatrix_t activation_values;

    // W.X + b= Y
    // n x m
    // connecting the previous layer to this layer
    // the edges are the weights
    nmatrix_t weights;
    nmatrix_t transposed_weights; // to not have to reallocate memory
    nmatrix_t transposed_inputs; 

    // n x 1
    nmatrix_t bias;

    // dE/dX = W.T * dE/dY
    // m x 1
    nmatrix_t d_cost_wrt_input;
    // same dimensions as weight and bias matrices
    nmatrix_t d_cost_wrt_weight;
    nmatrix_t d_cost_wrt_bias;
    neural_network_model_t *model;
} dense_layer_t;

typedef struct Dropout_Layer {
    layer_function_t functions;
    nmatrix_t output;

    nmatrix_t d_cost_wrt_input;
    float dropout;
    neural_network_model_t *model;
} dropout_layer_t;

// passes in the activation values of the previous layer into the activation function
typedef struct Activation_Layer {
    layer_function_t functions;
    nmatrix_t activated_values;
    // dE/dX = W.T * dE/dY
    // m x 1
    nmatrix_t d_cost_wrt_input;
    neural_network_model_t *model;
} activation_layer_t;

// final layer, compute the cost and derivatives to initiate backprop
typedef struct Output_Layer {
    layer_function_t functions;
    nmatrix_t (*make_guess)(layer_t *this, nmatrix_t output);
    nmatrix_t output_values;
    // dE/dX = W.T * dE/dY
    // m x 1
    nmatrix_t d_cost_wrt_input;
    nmatrix_t guess;
    float (*loss)(layer_t *this, nmatrix_t expected_output);
    neural_network_model_t *model;
} output_layer_t;

// unionized all layers under a common identity
typedef struct Layer {
    enum {
        INPUT,
        DENSE,
        DROPOUT,
        ACTIVATION,
        OUTPUT
    } type;

    union {
        input_layer_t input;
        dense_layer_t dense;
        dropout_layer_t dropout;
        activation_layer_t activation;
        output_layer_t output;
    } layer;

    // doubly linked
    layer_t *next;
    layer_t *prev;
} layer_t;

// nn model
// todo store more useful information of the model like
//  - training accuracy, avg error, epoch/iterations count
// for the visualizer
typedef struct NeuralNetworkModel {
    unsigned int num_layers;
    layer_t *input_layer; // first layer
    layer_t *output_layer; // last layer

    // info data
    bool is_training;
} neural_network_model_t;

typedef struct TrainingInfo {
    neural_network_model_t *model;
    unsigned int train_size;
    nmatrix_t *train_x; // stored as an array of columns
    nmatrix_t *train_y;
    unsigned int batch_size;
    float learning_rate;

    // when training stops, either condition is met => stops training
    unsigned int target_epochs;
    float target_accuracy; 

    unsigned int test_size;
    nmatrix_t *test_x;
    nmatrix_t *test_y;
    
    // stats
    bool in_progress;
    float train_accuracy;
    float train_correct;
    float test_accuracy;
    float test_correct;
    float avg_train_error;
    float avg_test_error;
    unsigned int epoch;
    unsigned int train_index;
    unsigned int test_index;

    // todo add some data collection like list of error/accuracy and which epoch it occured in
    // for data viz
} training_info_t;

nmatrix_t output_make_guess_one_hot_encoded(layer_t *this, nmatrix_t output);
nmatrix_t output_make_guess_passforward(layer_t *this, nmatrix_t output);
nmatrix_t output_make_guess_round(layer_t *this, nmatrix_t output);
nmatrix_t output_make_guess_softmax(layer_t *this, nmatrix_t output);

float output_cost_mean_squared(layer_t *this, nmatrix_t expected_output);
float output_cost_categorical_cross_entropy(layer_t *this, nmatrix_t expected_output);

// frees allocated memory for the layer
void layer_free(layer_t *layer);
nmatrix_t layer_get_neurons(layer_t *layer);

void model_free(neural_network_model_t *model);
void model_add_layer(neural_network_model_t *model, layer_t *layer);

// adds an layers to the model
// todo in future, specify dimensions instead of supply matrix to be then copied
layer_t* layer_input(neural_network_model_t *model, nmatrix_t input);
layer_t* layer_dense(neural_network_model_t *model, nmatrix_t neurons);
layer_t* layer_dropout(neural_network_model_t *model, float dropout);
layer_t* layer_activation(neural_network_model_t *model, layer_function_t functions);
layer_t* layer_output(neural_network_model_t *model, nmatrix_t (*make_guess)(layer_t*, nmatrix_t), layer_function_t functions, float (*loss)(layer_t*, nmatrix_t));

char* get_layer_name(layer_t *layer);
char* get_activation_function_name(activation_layer_t *layer);
char* get_output_function_name(output_layer_t *layer);
char* get_output_guess_function_name(output_layer_t *layer);
nmatrix_t layer_get_neurons(layer_t *layer);

nmatrix_t model_predict(neural_network_model_t *model, nmatrix_t input, 
               nmatrix_t output);

void model_initialize_matrix_normal_distribution(nmatrix_t model, float mean, float standard_deviation);
void model_back_propagate(neural_network_model_t *model, nmatrix_t expected_output, float learning_rate);
float model_train(neural_network_model_t *model, nmatrix_t *inputs, nmatrix_t *expected_outputs, unsigned int num_examples, float learning_rate);
void model_test(neural_network_model_t *model, nmatrix_t *inputs, nmatrix_t *expected_outputs, unsigned int num_tests);
nmatrix_t model_calculate(neural_network_model_t *model);


void training_info_free(training_info_t *training_info);

void model_train_info(training_info_t *training_info);
void model_test_info(training_info_t *training_info);

int unpack_one_hot_encoded(nmatrix_t one_hot_encoded);


#endif // MODEL_H