#include <model/model.h>
#include <util/math.h>

#include <stdio.h>
#include <math.h>

#include <util/debug_memory.h>

char* get_layer_name(layer_t *layer) {
    switch (layer->type) {
        case INPUT:
            return "Input";
        case DENSE:
            return "Dense";
        case ACTIVATION:
            return "Activation";
        case OUTPUT:
            return "Output";
        default:
            assert(0);
    }
}

char* get_activation_function_name(activation_layer_t *layer) {
    if (layer->feed_forward == activation_feed_forward_sigmoid) {
        return "Sigmoid";
    } else if (layer->feed_forward == activation_feed_forward_relu) {
        return "RELU";
    } else {
        assert(0);
    }
}

char* get_output_function_name(output_layer_t *layer) {
    if (layer->back_propagation == output_back_propagation_mean_squared) {
        return "Mean Squared Loss";
    } else if (layer->back_propagation == output_back_propagation_cross_entropy) {
        return "Cross Entropy Loss";
    } else {
        assert(0);
    }
}

mymatrix_t input_feed_forward(layer_t *this, mymatrix_t input) {
    matrix_memcpy(this->layer.input.input_values, input);
    return this->layer.input.input_values;
}

mymatrix_t dense_feed_forward(layer_t *this, mymatrix_t input) {
    matrix_multiply(this->layer.dense.weights, input, this->layer.dense.activation_values);
    matrix_add(this->layer.dense.activation_values, this->layer.dense.bias, this->layer.dense.activation_values);
    return this->layer.dense.activation_values;
}

mymatrix_t dense_back_propagation(layer_t *this, mymatrix_t d_error_wrt_output, float learning_rate) {
    // transpose input matrix
    // todo see if this needs optimization, likely does since allocation on every iteration is a lot of work
    mymatrix_t X = layer_get_neurons(this->prev);
    mymatrix_t X_T = this->layer.dense.transposed_inputs;
    matrix_transpose(X, X_T);

    mymatrix_t W = this->layer.dense.weights;
    mymatrix_t W_T = this->layer.dense.transposed_weights;
    matrix_transpose(W, W_T);

    matrix_multiply(d_error_wrt_output, X_T, this->layer.dense.d_cost_wrt_weight);
    matrix_memcpy(this->layer.dense.d_cost_wrt_bias, d_error_wrt_output);
    matrix_multiply(W_T, d_error_wrt_output, this->layer.dense.d_cost_wrt_input);

    // apply adjustments
    matrix_multiply_scalar(this->layer.dense.d_cost_wrt_weight, learning_rate, this->layer.dense.d_cost_wrt_weight);
    matrix_multiply_scalar(this->layer.dense.d_cost_wrt_bias, learning_rate, this->layer.dense.d_cost_wrt_bias);
    
    matrix_sub(this->layer.dense.weights, this->layer.dense.d_cost_wrt_weight, this->layer.dense.weights);
    matrix_sub(this->layer.dense.bias, this->layer.dense.d_cost_wrt_bias, this->layer.dense.bias);

    return this->layer.dense.d_cost_wrt_input;
}

mymatrix_t activation_feed_forward_sigmoid(layer_t *this, mymatrix_t input) {
    matrix_for_each_operator(input, sigmoid, this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}

mymatrix_t activation_feed_forward_relu(layer_t *this, mymatrix_t input) {
    matrix_for_each_operator(input, relu, this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}

mymatrix_t activation_back_propagation_sigmoid(layer_t *this, mymatrix_t d_cost_wrt_output) {
    mymatrix_t X = layer_get_neurons(this->prev);

    matrix_for_each_operator(X, sigmoid_prime, this->layer.activation.activated_values);
    matrix_elementwise_multiply(d_cost_wrt_output, this->layer.activation.activated_values, this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}

mymatrix_t activation_back_propagation_relu(layer_t *this, mymatrix_t d_cost_wrt_output) {
    mymatrix_t X = layer_get_neurons(this->prev);

    matrix_for_each_operator(X, relu_prime, this->layer.activation.activated_values);
    matrix_elementwise_multiply(d_cost_wrt_output, this->layer.activation.activated_values, this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}

mymatrix_t output_make_guess_one_hot_encoded(layer_t *this, mymatrix_t output) {
    float max = -INFINITY;
    for (int r = 0; r < output.r; r++) {
        for (int c = 0; c < output.c; c++) {
            max = fmax(max, output.matrix[r][c]);
        }
    }

    for (int r = 0; r < output.r; r++) {
        for (int c = 0; c < output.c; c++) {
            this->layer.output.guess.matrix[r][c] = (output.matrix[r][c] == max) ? 1 : 0;
        }
    }
    return this->layer.output.guess;
}

mymatrix_t output_make_guess_passforward(layer_t *this, mymatrix_t output) {
    matrix_memcpy(this->layer.output.guess, output);
    return this->layer.output.guess;
}

mymatrix_t output_make_guess_round(layer_t *this, mymatrix_t output) {
    for (int r = 0; r < output.r; r++) {
        for (int c = 0; c < output.c; c++) {
            this->layer.output.guess.matrix[r][c] = round(output.matrix[r][c]);
        }
    }
    return this->layer.output.guess;
}

mymatrix_t output_make_guess_softmax(layer_t *this, mymatrix_t output) {
    float sum = 0;
    for (int r = 0; r < output.r; r++) {
        for (int c = 0; c < output.c; c++) {
            sum += exp(output.matrix[r][c]);
        }
    }

    for (int r = 0; r < output.r; r++) {
        for (int c = 0; c < output.c; c++) {
            this->layer.output.guess.matrix[r][c] = exp(output.matrix[r][c]) / sum;
        }
    }
    return this->layer.output.guess;
}

mymatrix_t output_back_propagation_mean_squared(layer_t *this, mymatrix_t expected_output) {
    mymatrix_t output = this->layer.output.output_values;
    matrix_sub(output, expected_output, this->layer.output.d_cost_wrt_input);
    matrix_multiply_scalar(this->layer.output.d_cost_wrt_input, 2.0 / (float) output.r, this->layer.output.d_cost_wrt_input);
    return this->layer.output.d_cost_wrt_input;
}

mymatrix_t output_back_propagation_cross_entropy(layer_t *this, mymatrix_t expected_output) {
    // todo
    assert(0);
    mymatrix_t output;
    return output;
}

float output_cost_mean_squared(layer_t *this, mymatrix_t expected_output) {
    float mean_squared = 0;
    mymatrix_t actual_output = layer_get_neurons(this);
    for (int r = 0; r < actual_output.r; r++) {
        for (int c = 0; c < actual_output.c; c++) {
            mean_squared += pow(expected_output.matrix[r][c] - actual_output.matrix[r][c], 2);
        }
    }
    mean_squared /= expected_output.r;
    return mean_squared;
}

float output_cost_cross_entropy(layer_t *this, mymatrix_t expected_output) {
    // TODO
    assert(0);
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
            matrix_free(layer->layer.dense.d_cost_wrt_input);
            matrix_free(layer->layer.dense.d_cost_wrt_weight);
            matrix_free(layer->layer.dense.d_cost_wrt_bias);
            break;
        case ACTIVATION:
            matrix_free(layer->layer.activation.activated_values);
            matrix_free(layer->layer.activation.d_cost_wrt_input);
            break;
        case OUTPUT:
            matrix_free(layer->layer.output.output_values);
            matrix_free(layer->layer.output.guess);
            matrix_free(layer->layer.output.d_cost_wrt_input);
            break;
    }
    free(layer);
}

mymatrix_t layer_get_neurons(layer_t *layer) {
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

void model_free(neural_network_model_t *model) {
    layer_t *current = model->input_layer;
    for (int layer_i = 0; layer_i < model->num_layers; layer_i++) {
        layer_t *prev = current;
        current = current->next;
        layer_free(prev);
    }
    assert(current == NULL); // ensure freed all layers
}

void model_add_layer(neural_network_model_t *model, layer_t *layer) {
    if (model->input_layer == NULL) {
        model->input_layer = layer;
        model->output_layer = layer;
        model->input_layer->prev = NULL;
    } else {
        model->output_layer->next = layer;
        model->output_layer->next->prev = model->output_layer;
        model->output_layer = layer;
    }
    model->num_layers++;
    model->output_layer->next = NULL;
}


layer_t* layer_input(neural_network_model_t *model, mymatrix_t input) {
    assert(model->num_layers == 0 && model->input_layer == NULL);
    // for now, column vector
    assert(input.c == 1);

    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = INPUT;
    input_layer_t *input_layer = &layer->layer.input;
    input_layer->input_values = matrix_copy(input);
    input_layer->feed_forward = input_feed_forward;

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_dense(neural_network_model_t *model, mymatrix_t neurons) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    // for now, column vector
    assert(neurons.c == 1);

    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = DENSE;

    dense_layer_t *dense = &layer->layer.dense;
    dense->activation_values = matrix_copy(neurons);
    mymatrix_t prev_output = layer_get_neurons(model->output_layer);
    dense->weights = matrix_allocator(neurons.r, prev_output.r);
    dense->transposed_weights = matrix_allocator(dense->weights.c, dense->weights.r);
    dense->transposed_inputs = matrix_allocator(prev_output.c, prev_output.r);
    dense->bias = matrix_allocator(neurons.r, 1);
    dense->d_cost_wrt_input = matrix_allocator(prev_output.r, 1);
    dense->d_cost_wrt_weight = matrix_allocator(dense->weights.r, dense->weights.c);
    dense->d_cost_wrt_bias = matrix_allocator(dense->bias.r, dense->bias.c);

    dense->back_propagation = dense_back_propagation;
    dense->feed_forward = dense_feed_forward;

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_activation(neural_network_model_t *model, mymatrix_t (*feed_forward)(layer_t*, mymatrix_t), mymatrix_t (*back_propagation)(layer_t*, mymatrix_t)) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    
    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = ACTIVATION;
    activation_layer_t *activation = &layer->layer.activation;
    mymatrix_t prev_output = layer_get_neurons(model->output_layer);
    activation->activated_values = matrix_allocator(prev_output.r, prev_output.c);
    activation->feed_forward = feed_forward;
    activation->back_propagation = back_propagation;
    activation->d_cost_wrt_input = matrix_allocator(prev_output.r, 1);

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_output(neural_network_model_t *model, mymatrix_t (*make_guess)(layer_t*, mymatrix_t), mymatrix_t (*back_propagation)(layer_t*, mymatrix_t)) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    
    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = OUTPUT;
    output_layer_t *output = &layer->layer.output;
    mymatrix_t prev_output = layer_get_neurons(model->output_layer);
    output->output_values = matrix_allocator(prev_output.r, prev_output.c);
    output->make_guess = make_guess;
    output->back_propagation = back_propagation;
    output->d_cost_wrt_input = matrix_allocator(prev_output.r, 1);
    output->guess = matrix_allocator(prev_output.r, 1);

    model_add_layer(model, layer);
    return layer;
}

void model_initialize_matrix_normal_distribution(mymatrix_t matrix, float mean, float standard_deviation) {
    for (int r = 0; r < matrix.r; r++) {
        for (int c = 0; c < matrix.c; c++) {
            matrix.matrix[r][c] = random_normal_distribution_BoxMullerTransform(standard_deviation) + mean;
        }
    }
}

mymatrix_t model_predict(neural_network_model_t *model, mymatrix_t input, 
                        mymatrix_t output) {
    
    layer_t *current = model->input_layer;
    mymatrix_t prev_output = input;
    for (int layer_i = 0; layer_i < model->num_layers-1; layer_i++) {
        assert(current->type != OUTPUT);
        
        // since the different layer structs are arranged in a way that the function pointers are in the same "locations"
        // this should work for all layers without having to use a switch
        prev_output = current->layer.input.feed_forward(current, prev_output);
        current = current->next;
    }

    model->output_layer->layer.output.make_guess(model->output_layer, prev_output);
    matrix_memcpy(current->layer.output.output_values, prev_output);
    matrix_memcpy(output, prev_output);
    return output;
}

void model_back_propagate(neural_network_model_t *model, mymatrix_t expected_output, float learning_rate) {
    layer_t *current = model->output_layer;
    mymatrix_t d_cost_wrt_Y = expected_output;
    for (int layer_i = model->num_layers; layer_i > 1; layer_i--) {
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

        // printf("\n%s: de/dy: \n", get_layer_name(current->next));
        // matrix_print(d_cost_wrt_Y);
    }
}

float model_train(neural_network_model_t *model, mymatrix_t *inputs, mymatrix_t *expected_outputs, unsigned int num_examples, float learning_rate) {
    mymatrix_t output = matrix_copy(model->output_layer->layer.output.output_values);
    float avg_error = 0;
    for (int example_i = 0; example_i < num_examples; example_i++) {
        model_predict(model, inputs[example_i], output);
        avg_error += output_cost_mean_squared(model->output_layer, expected_outputs[example_i]);
        model_back_propagate(model, expected_outputs[example_i], learning_rate);
    }
    avg_error /= (float) num_examples;
    // printf("train avg error=%f", (float) avg_error);
    matrix_free(output);
    return avg_error;
}

void model_test(neural_network_model_t *model, mymatrix_t *inputs, mymatrix_t *expected_outputs, unsigned int num_tests) {
    mymatrix_t output = matrix_copy(model->output_layer->layer.output.output_values);
    int passed = 0;
    for (int test_i = 0; test_i < num_tests; test_i++) {
        model_predict(model, inputs[test_i], output);

        printf("input\n");
        matrix_print(inputs[test_i]);
        printf("guess vs expected\n");
        matrix_print(model->output_layer->layer.output.guess);
        matrix_print(expected_outputs[test_i]);
        printf("\n");

        mymatrix_t guess = model->output_layer->layer.output.make_guess(model->output_layer, output);
        if (matrix_equal(expected_outputs[test_i], guess)) {
            passed++;
        }
    }

    // 2 decimal places
    float accuracy = ((int)(100.0 * (float) passed / (float) num_tests)) / 100.0;

    printf("accuracy: %f  passed=%d, total=%d\n", accuracy, passed, num_tests);

    matrix_free(output);
}

void model_train_info(training_info_t *training_info) {
    // todo support batch size in future
    assert(training_info->batch_size == 1);
    neural_network_model_t *model = training_info->model;
    
    unsigned int *epoch = &training_info->epoch;
    unsigned int target_epochs = training_info->target_epochs;
    unsigned int *train_index = &training_info->train_index;
    unsigned int train_size = training_info->train_size;
    unsigned int *test_index = &training_info->test_index;
    unsigned int test_size = training_info->test_size;

    output_layer_t output_layer = model->output_layer->layer.output;
    mymatrix_t *train_x = training_info->train_x;
    mymatrix_t *train_y = training_info->train_y;
    mymatrix_t *test_x = training_info->test_x;
    mymatrix_t *test_y = training_info->test_y;

    mymatrix_t actual_output = matrix_allocator(output_layer.output_values.r, output_layer.output_values.c);
    for (*epoch = 0; *epoch < target_epochs; (*epoch)++) {
        // perform training
        float avg_train_error = 0;
        int passed_train = 0;
        for (*train_index = 0; *train_index < train_size; (*train_index)++) {
            model_predict(model, train_x[*train_index], actual_output);
            avg_train_error += output_cost_mean_squared(model->output_layer, train_y[*train_index]);
            model_back_propagate(model, train_y[*train_index], training_info->learning_rate);

            mymatrix_t model_guess = output_layer.make_guess(model->output_layer, actual_output);
            if (matrix_equal(train_y[*train_index], model_guess)) {
                passed_train++;
            }
        }
        training_info->avg_train_error = avg_train_error / (float) train_size;
        training_info->train_accuracy = ((int)(100.0 * (float) passed_train / (float) train_size)) / 100.0;

        // perform test
        float avg_test_error = 0;
        int passed_test = 0;
        for (*test_index = 0; *test_index < test_size; (*test_index)++) {
            model_predict(model, test_x[*test_index], actual_output);
            avg_test_error += output_cost_mean_squared(model->output_layer, test_y[*test_index]);
            
            mymatrix_t model_guess = output_layer.make_guess(model->output_layer, actual_output);
            if (matrix_equal(test_y[*test_index], model_guess)) {
                passed_test++;
            }
        }

        training_info->avg_test_error = avg_test_error / (float) test_size;
        training_info->test_accuracy = ((int)(100.0 * (float) passed_test / (float) test_size)) / 100.0;


        // check if we can stop
        if (training_info->target_accuracy <= training_info->test_accuracy && training_info->target_accuracy <= training_info->train_accuracy) {
            break; // finish early
        }
    }
    matrix_free(actual_output);
}