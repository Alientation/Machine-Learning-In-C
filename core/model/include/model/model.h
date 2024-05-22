#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <util/matrix.h>

#include <assert.h>
#include <memory.h>
#include <stdlib.h>

typedef struct Layer layer_t;
typedef struct Model model_t;
typedef struct Input_Layer input_layer_t;
typedef struct Dense_Layer dense_layer_t;
typedef struct Activation_Layer activation_layer_t;
typedef struct Output_Layer output_layer_t;


typedef struct Input_Layer {
    matrix_t* (*feed_forward)(input_layer_t *this, matrix_t *input);
    matrix_t *input_values;
} input_layer_t;

// n: number of neurons in this layer
// m: number of neurons in the previous layer
typedef struct Dense_Layer {
    matrix_t* (*feed_forward)(dense_layer_t *this, matrix_t *input);
    matrix_t* (*back_propagation)(dense_layer_t *this, matrix_t *input_gradient);
    // n x 1, this layer's neurons
    matrix_t *activation_values;

    // W.X + b= Y
    // n x m
    // connecting the previous layer to this layer
    // the edges are the weights
    matrix_t *weights;

    // n x 1
    matrix_t *bias;
} dense_layer_t;

typedef struct Activation_Layer {
    matrix_t* (*feed_forward)(activation_layer_t *this, matrix_t *input);
    matrix_t* (*back_propagation)(activation_layer_t *this, matrix_t *input_gradient);
    matrix_t *activated_values;
} activation_layer_t;

typedef struct Output_Layer {
    void (*feed_forward)(void); // dummy variable to pad the back_prop function to the same location as other layer structs
    matrix_t* (*back_propagation)(activation_layer_t *this, matrix_t *input_gradient);
    matrix_t *output_values;
} output_layer_t;


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

    layer_t *next;
    layer_t *prev;
} layer_t;

typedef struct Model {
    unsigned int num_layers;
    layer_t *input_layer;
    layer_t *output_layer;


} model_t;

matrix_t *input_feed_forward(input_layer_t *this, matrix_t* input) {
    matrix_memcpy(this->input_values, input);
    return this->input_values;
}

void layer_free(layer_t *layer) {
    switch (layer->type) {
        case INPUT:
            matrix_free(layer->layer.input.input_values);
            break;
        case DENSE:
            matrix_free(layer->layer.dense.activation_values);
            matrix_free(layer->layer.dense.weights);
            matrix_free(layer->layer.dense.bias);
            break;
        case ACTIVATION:
            matrix_free(layer->layer.activation.activated_values);
            break;
        case OUTPUT:
            matrix_free(layer->layer.output.output_values);
            break;
    }
}

matrix_t* layer_get_neurons(layer_t *layer) {
    switch (layer->type) {
        case INPUT:
            return layer->layer.input.input_values;
            break;
        case DENSE:
            return layer->layer.dense.activation_values;
            break;
        case ACTIVATION:
            return layer->layer.activation.activated_values;
            break;
        case OUTPUT:
            return layer->layer.output.output_values;
            break;
    }
}

void model_free(model_t *model) {
    layer_t *current = model->input_layer;
    for (int i = 0; i < model->num_layers; i++) {
        current = current->next;
        layer_free(current->prev);
    }
    assert(current == NULL); // ensure freed all layers
}

void model_add_layer(model_t *model, layer_t *layer) {
    if (model->input_layer == NULL) {
        model->input_layer = layer;
        model->output_layer = layer;
    } else {
        model->output_layer->next = layer;
        model->output_layer->next->prev = model->output_layer;
        model->output_layer = layer;
    }
    model->num_layers++;
}

void layer_input(model_t *model, matrix_t *input) {
    assert(model->num_layers == 0 && model->input_layer == NULL);
    // for now, column vector
    assert(input->c == 1);

    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = INPUT;
    layer->layer.input.input_values = matrix_copy(input);
    layer->layer.input.feed_forward = input_feed_forward;
    model_add_layer(model, layer);
}

void layer_dense(model_t *model, matrix_t *neurons) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    // for now, column vector
    assert(neurons->c == 1);

    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = DENSE;
    layer->layer.dense.activation_values = matrix_copy(neurons);

    layer->layer.dense.weights = malloc(sizeof(matrix_t));
    layer->layer.dense.weights->matrix = malloc(layer_get_neurons(model->output_layer)->r * neurons->r * sizeof(double));
    layer->layer.dense.bias = malloc(sizeof(matrix_t));
    layer->layer.dense.bias->matrix = malloc(neurons->r * 1 * sizeof(double));
    model_add_layer(model, layer);
}



void model_run(model_t *model, matrix_t *input, 
               matrix_t *output) {
    
    layer_t *current = model->input_layer;
    matrix_t *prev_output = input;
    for (int i = 0; i < model->num_layers-1; i++) {
        assert(current->type != OUTPUT);

        prev_output = current->layer.input.feed_forward(model, prev_output);
        current = current->next;
    }

    matrix_memcpy(output, prev_output);
}

#endif