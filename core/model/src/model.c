#include <model/model.h>
#include <util/math.h>

#include <stdio.h>
#include <math.h>

matrix_t *input_feed_forward(layer_t *this, matrix_t *input) {
    matrix_memcpy(this->layer.input.input_values, input);
    return this->layer.input.input_values;
}

matrix_t *dense_feed_forward(layer_t *this, matrix_t *input) {
    matrix_multiply(this->layer.dense.weights, input, this->layer.dense.activation_values);
    return this->layer.dense.weights;
}

matrix_t *dense_back_propagation(layer_t *this, matrix_t *d_error_wrt_output) {
    // transpose input matrix
    matrix_t *X = layer_get_neurons(this->prev);
    matrix_t *X_T = matrix_allocator(X->c, X->r);
    matrix_transpose(X, X_T);

    matrix_multiply(d_error_wrt_output, X_T, this->layer.dense.d_cost_wrt_input);
    return this->layer.dense.d_cost_wrt_input;
}

// todo move to util library
double sigmoid(double z) {
    return 1. / (1 + exp(-z));
}

matrix_t *activation_feed_forward_sigmoid(layer_t *this, matrix_t *input) {
    matrix_for_each_operator(input, sigmoid, this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}


// todo move to util library
double relu(double z) {
    return fmax(0, z);
}

matrix_t *activation_feed_forward_relu(layer_t *this, matrix_t *input) {
    matrix_for_each_operator(input, relu, this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}

double sigmoid_prime(double z) {
    // todo
}

matrix_t *activation_back_propagation_sigmoid(layer_t *this, matrix_t *d_cost_wrt_output) {
    matrix_t *X = layer_get_neurons(this->prev);

    matrix_for_each_operator(X, sigmoid_prime, this->layer.activation.activated_values);
    matrix_elementwise_multiply(d_cost_wrt_output, this->layer.activation.activated_values, this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}

double relu_prime(double z) {
    // todo
}

matrix_t *activation_back_propagation_relu(layer_t *this, matrix_t *d_cost_wrt_output) {
    matrix_t *X = layer_get_neurons(this->prev);

    matrix_for_each_operator(X, relu_prime, this->layer.activation.activated_values);
    matrix_elementwise_multiply(d_cost_wrt_output, this->layer.activation.activated_values, this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}

matrix_t *output_make_guess_one_hot_encoded(layer_t *this, matrix_t *output) {
    double max = -INFINITY;
    for (int r = 0; r < output->r; r++) {
        for (int c = 0; c < output->c; c++) {
            max = fmax(max, output->matrix[r][c]);
        }
    }

    for (int r = 0; r < output->r; r++) {
        for (int c = 0; c < output->c; c++) {
            this->layer.output.guess->matrix[r][c] = output->matrix[r][c] == max ? 1 : 0;
        }
    }
    return this->layer.output.guess;
}

matrix_t *output_back_propagation_mean_squared(layer_t *this, matrix_t *expected_output) {
    matrix_t *output = this->layer.output.output_values;
    matrix_sub(output, expected_output, this->layer.output.d_cost_wrt_input);
    matrix_multiply_scalar(this->layer.output.d_cost_wrt_input, 2.0 / output->r, this->layer.output.d_cost_wrt_input);
    return this->layer.output.d_cost_wrt_input;
}

matrix_t *output_back_propagation_cross_entropy(layer_t *this, matrix_t *expected_output) {

}

double output_cost_mean_squared(layer_t *this, matrix_t *expected_output) {
    double mean_squared = 0;
    matrix_t *actual_output = layer_get_neurons(this);
    for (int r = 0; r < actual_output->r; r++) {
        for (int c = 0; c < actual_output->c; c++) {
            mean_squared += pow(actual_output->matrix[r][c] - expected_output->matrix[r][c], 2);
        }
    }
    mean_squared *= 2.0 / expected_output->r;
    return mean_squared;
}

double output_cost_cross_entropy(layer_t *this, matrix_t *expected_output) {
    // TODO
    return -1.0;
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

    assert(0);
}

void model_free(model_t *model) {
    layer_t *current = model->input_layer;
    for (int layer_i = 0; layer_i < model->num_layers; layer_i++) {
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


layer_t* layer_input(model_t *model, matrix_t *input) {
    assert(model->num_layers == 0 && model->input_layer == NULL);
    // for now, column vector
    assert(input->c == 1);

    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = INPUT;
    input_layer_t *input_layer = &layer->layer.input;
    input_layer->input_values = matrix_copy(input);
    input_layer->feed_forward = input_feed_forward;

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_dense(model_t *model, matrix_t *neurons) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    // for now, column vector
    assert(neurons->c == 1);

    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = DENSE;

    dense_layer_t *dense = &layer->layer.dense;
    dense->activation_values = matrix_copy(neurons);
    matrix_t *prev_output = layer_get_neurons(model->output_layer);
    dense->weights = matrix_allocator(prev_output->r, neurons->r);
    dense->bias = matrix_allocator(neurons->r, 1);
    dense->d_cost_wrt_input = matrix_allocator(prev_output->r, 1);

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_activation(model_t *model, matrix_t* (*feed_forward)(layer_t*, matrix_t*), matrix_t* (*back_propagation)(layer_t*, matrix_t*)) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    
    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = ACTIVATION;
    activation_layer_t *activation = &layer->layer.activation;
    matrix_t *prev_output = layer_get_neurons(model->output_layer);
    activation->activated_values = matrix_allocator(prev_output->r, prev_output->c);
    activation->feed_forward = feed_forward;
    activation->back_propagation = back_propagation;
    activation->d_cost_wrt_input = matrix_allocator(prev_output->r, 1);

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_output(model_t *model, matrix_t* (*make_guess)(layer_t*, matrix_t*), matrix_t* (*back_propagation)(layer_t*, matrix_t*)) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    
    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = OUTPUT;
    output_layer_t *output = &layer->layer.output;
    matrix_t *prev_output = layer_get_neurons(model->output_layer);
    output->output_values = matrix_allocator(prev_output->r, prev_output->c);
    output->make_guess = make_guess;
    output->back_propagation = back_propagation;
    output->d_cost_wrt_input = matrix_allocator(prev_output->r, 1);
    output->guess = matrix_allocator(prev_output->r, 1);

    model_add_layer(model, layer);
    return layer;
}

void model_initialize_matrix_normal_distribution(matrix_t *matrix, double mean, double standard_deviation) {
    for (int r = 0; r < matrix->r; r++) {
        for (int c = 0; c < matrix->c; c++) {
            matrix->matrix[r][c] = random_normal_distribution_BoxMullerTransform(standard_deviation) + mean;
        }
    }
}

void model_run(model_t *model, matrix_t *input, 
               matrix_t *output) {
    
    layer_t *current = model->input_layer;
    matrix_t *prev_output = input;
    for (int layer_i = 0; layer_i < model->num_layers-1; layer_i++) {
        assert(current->type != OUTPUT);
        
        // since the different layer structs are arranged in a way that the function pointers are in the same "locations"
        // this should work for all layers without having to use a switch
        prev_output = current->layer.input.feed_forward(current, prev_output);
        current = current->next;
    }

    matrix_memcpy(output, prev_output);
}

void model_back_propagate(model_t *model, matrix_t *expected_output, double learning_rate) {
    layer_t *current = model->output_layer;
    matrix_t *d_cost_wrt_Y = expected_output;
    for (int layer_i = model->num_layers - 1; layer_i >= 1; layer_i--) {
        assert(current->type != INPUT);

        switch (current->type) {
            case DENSE:
                d_cost_wrt_Y = current->layer.dense.back_propagation(current, d_cost_wrt_Y, learning_rate);
                break;
            case ACTIVATION:
                d_cost_wrt_Y = current->layer.activation.back_propagation(current, d_cost_wrt_Y);
                break;
            case OUTPUT:
                d_cost_wrt_Y = current->layer.output.back_propagation(current, d_cost_wrt_Y);
        }
        current = current->prev;
    }
}

void model_train(model_t *model, matrix_t **inputs, matrix_t **expected_outputs, unsigned int num_examples, double learning_rate) {
    matrix_t *output = matrix_copy(model->output_layer->layer.output.output_values);
    for (int example_i = 0; example_i < num_examples; example_i++) {
        model_run(model, inputs[example_i], output);
        model_back_propagate(model, expected_outputs[example_i], learning_rate);
    }
}

void model_test(model_t *model, matrix_t **inputs, matrix_t **expected_outputs, unsigned int num_tests) {
    matrix_t *output = matrix_copy(model->output_layer->layer.output.output_values);
    int passed = 0;
    for (int test_i = 0; test_i < num_tests; test_i++) {
        model_run(model, inputs[test_i], output);

        matrix_t *guess = model->output_layer->layer.output.make_guess(model->output_layer, output);
        if (matrix_equal(expected_outputs[test_i], guess)) {
            passed++;
        }
    }

    // 2 decimal places
    double accuracy = ((int)(100 * passed / (double) num_tests)) / 100;

    printf("%f", accuracy, passed, num_tests);
}