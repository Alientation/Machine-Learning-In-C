#pragma once
#ifndef MODEL_H
#define MODEL_H

typedef struct InputLayer {
    double **layer;
    int r;
    int c;
} input_layer_t, output_layer_t;

typedef struct ActivationLayer {
    // pointer to function that takes in and outputs a layer
    void (*activation)(input_layer_t*, output_layer_t*);
} activation_layer_t;

typedef struct NeuronLayer {
    // weight matrix + bias
    // Y = X.W + b   <- bias added to each row of the output matrix
    double **weight; // r x c
    int r;
    int c;
    double* bias; // 1 x c
} neuron_layer_t;

typedef struct Layer {
    union Layer {
        input_layer_t input;
        activation_layer_t activation;
        neuron_layer_t neuron;
        output_layer_t output;
    } layer;

    enum LayerType {
        LayerType_INPUT,
        LayerType_ACTIVATION,
        LayerType_NEURON,
        LayerType_OUTPUT
    } type;
} layer_t;

typedef struct Model {
    // contain info about the various layers of the model
    // input -> neuron -> activation -> output
    layer_t* layers;
    unsigned int num_layers;
} model_t;


// todo return useful stats/info out
void model_run(model_t *model, input_layer_t *input,
                output_layer_t *output);
void model_train(model_t *model, int num_data, input_layer_t **inputs, output_layer_t **outputs);
void model_test(model_t *model, int num_data, input_layer_t **inputs, output_layer_t **outputs);

/**
 * runs a specified layer with a passed in input layer 
 * and writes to output layer
 */
void layer_run(layer_t *layer, input_layer_t *input,
                output_layer_t *output);

#endif