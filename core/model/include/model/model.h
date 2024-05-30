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
 * add dropout
 * weight regularlization techniques (L2/L1)
 * weight clipping?
 * add momentum (remembers previous gradients)
 * layer/batch normalization?
 * weights should probably be normalized
 * save model to file
 * load model from file  *so we can pretrain models
 * digit detector
 * batch/mini batch, full gradient descent (apparently we are using stochastic gradient descent)
 * add more variety of layers (convolution, pooling, reshaping)
 * add variable learning rates for each layer support
 * matrix broadcasting??? support for multi-dimension matrices?????
 */



typedef struct Layer layer_t;
typedef struct NeuralNetworkModel neural_network_model_t;
typedef struct Input_Layer input_layer_t;
typedef struct Dense_Layer dense_layer_t;
typedef struct Activation_Layer activation_layer_t;
typedef struct Output_Layer output_layer_t;

// first layer of the model
typedef struct Input_Layer {
    mymatrix_t (*feed_forward)(layer_t *this, mymatrix_t input);
    mymatrix_t input_values;
} input_layer_t;

// fully connected inner layer of the model
// n: number of neurons in this layer
// m: number of neurons in the previous layer
typedef struct Dense_Layer {
    mymatrix_t (*feed_forward)(layer_t *this, mymatrix_t input);
    mymatrix_t (*back_propagation)(layer_t *this, mymatrix_t input_gradient, double learning_rate);
    // n x 1, this layer's neurons which contain the "output" values
    mymatrix_t activation_values;

    // W.X + b= Y
    // n x m
    // connecting the previous layer to this layer
    // the edges are the weights
    mymatrix_t weights;
    mymatrix_t transposed_weights; // to not have to reallocate memory
    mymatrix_t transposed_inputs; 

    // n x 1
    mymatrix_t bias;

    // dE/dX = W.T * dE/dY
    // m x 1
    mymatrix_t d_cost_wrt_input;
    // same dimensions as weight and bias matrices
    mymatrix_t d_cost_wrt_weight;
    mymatrix_t d_cost_wrt_bias;
} dense_layer_t;

// passes in the activation values of the previous layer into the activation function
typedef struct Activation_Layer {
    mymatrix_t (*feed_forward)(layer_t *this, mymatrix_t input);
    mymatrix_t (*back_propagation)(layer_t *this, mymatrix_t input_gradient);
    mymatrix_t activated_values;
    // dE/dX = W.T * dE/dY
    // m x 1
    mymatrix_t d_cost_wrt_input;
} activation_layer_t;

// final layer, compute the cost and derivatives to initiate backprop
typedef struct Output_Layer {
    // void (*feed_forward)(void); // dummy variable to pad the back_prop function to the same location as other layer structs
    mymatrix_t (*make_guess)(layer_t *this, mymatrix_t output);
    mymatrix_t (*back_propagation)(layer_t *this, mymatrix_t expected_output);
    mymatrix_t output_values;
    // dE/dX = W.T * dE/dY
    // m x 1
    mymatrix_t d_cost_wrt_input;
    mymatrix_t guess;
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
// todo store more useful information of the model like
//  - training accuracy, avg error, epoch/iterations count
// for the visualizer
typedef struct NeuralNetworkModel {
    unsigned int num_layers;
    layer_t *input_layer; // first layer
    layer_t *output_layer; // last layer

    // info data
    
} neural_network_model_t;

typedef struct TrainingInfo {
    neural_network_model_t *model;
    unsigned int train_size;
    mymatrix_t *train_x; // stored as an array of columns
    mymatrix_t *train_y;
    unsigned int batch_size;
    double learning_rate;

    // when training stops, either condition is met => stops training
    unsigned int target_epochs;
    double target_accuracy; 

    unsigned int test_size;
    mymatrix_t *test_x;
    mymatrix_t *test_y;
    
    // stats
    bool in_progress;
    double train_accuracy;
    double test_accuracy;
    double avg_train_error;
    double avg_test_error;
    unsigned int epoch;
    unsigned int train_index;
    unsigned int test_index;

    // todo add some data collection like list of error/accuracy and which epoch it occured in
    // for data viz
} training_info_t;

mymatrix_t input_feed_forward(layer_t *this, mymatrix_t input); 

mymatrix_t dense_feed_forward(layer_t *this, mymatrix_t input);
mymatrix_t dense_back_propagation(layer_t *this, mymatrix_t d_error_wrt_output, double learning_rate);

mymatrix_t activation_feed_forward_sigmoid(layer_t *this, mymatrix_t input);
mymatrix_t activation_feed_forward_relu(layer_t *this, mymatrix_t input);
// matrix_t activation_feed_forward_softmax(layer_t *this, matrix_t input); //todo
// todo tanh
mymatrix_t activation_back_propagation_sigmoid(layer_t *this, mymatrix_t d_cost_wrt_output);
mymatrix_t activation_back_propagation_relu(layer_t *this, mymatrix_t d_cost_wrt_output);
// matrix_t activation_back_propagation_softmax(layer_t *this, matrix_t input); //todo
// todo tanh back prop

mymatrix_t output_make_guess_one_hot_encoded(layer_t *this, mymatrix_t output);
mymatrix_t output_make_guess_passforward(layer_t *this, mymatrix_t output);
mymatrix_t output_make_guess_round(layer_t *this, mymatrix_t output);
mymatrix_t output_make_guess_softmax(layer_t *this, mymatrix_t output);
mymatrix_t output_back_propagation_mean_squared(layer_t *this, mymatrix_t expected_output);
mymatrix_t output_back_propagation_cross_entropy(layer_t *this, mymatrix_t expected_output);

double output_cost_mean_squared(layer_t *this, mymatrix_t expected_output);
double output_cost_cross_entropy(layer_t *this, mymatrix_t expected_output);

// frees allocated memory for the layer
void layer_free(layer_t *layer);
mymatrix_t layer_get_neurons(layer_t *layer);

void model_free(neural_network_model_t *model);
void model_add_layer(neural_network_model_t *model, layer_t *layer);

// adds an layers to the model
// todo in future, specify dimensions instead of supply matrix to be then copied
layer_t* layer_input(neural_network_model_t *model, mymatrix_t input);
layer_t* layer_dense(neural_network_model_t *model, mymatrix_t neurons);
layer_t* layer_activation(neural_network_model_t *model, mymatrix_t (*feed_forward)(layer_t*, mymatrix_t), mymatrix_t (*back_propagation)(layer_t*, mymatrix_t));
layer_t* layer_output(neural_network_model_t *model, mymatrix_t (*make_guess)(layer_t*, mymatrix_t), mymatrix_t (*back_propagation)(layer_t*, mymatrix_t));

char* get_layer_name(layer_t *layer);
char* get_activation_function_name(activation_layer_t *layer);
char* get_output_function_name(output_layer_t *layer);
mymatrix_t layer_get_neurons(layer_t *layer);

mymatrix_t model_predict(neural_network_model_t *model, mymatrix_t input, 
               mymatrix_t output);

void model_initialize_matrix_normal_distribution(mymatrix_t model, double mean, double standard_deviation);
void model_back_propagate(neural_network_model_t *model, mymatrix_t expected_output, double learning_rate);
double model_train(neural_network_model_t *model, mymatrix_t *inputs, mymatrix_t *expected_outputs, unsigned int num_examples, double learning_rate);
void model_test(neural_network_model_t *model, mymatrix_t *inputs, mymatrix_t *expected_outputs, unsigned int num_tests);


void model_train_info(training_info_t *training_info);


#endif // MODEL_H