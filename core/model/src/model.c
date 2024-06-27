#include <model/model.h>
#include <util/math.h>
#include <unistd.h>

#include <stdio.h>
#include <math.h>

#include <util/debug_memory.h>
#define SHAPE(...) nshape_constructor(__VA_ARGS__)



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


layer_t* layer_input(neural_network_model_t *model, nmatrix_t input) {
    assert(model->num_layers == 0 && model->input_layer == NULL);

    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = INPUT;
    input_layer_t *input_layer = &layer->layer.input;
    input_layer->input_values = nmatrix_copy(&input);
    input_layer->functions = input_functions;
    input_layer->model = model;

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_dense(neural_network_model_t *model, nmatrix_t neurons) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);

    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = DENSE;

    dense_layer_t *dense = &layer->layer.dense;
    dense->activation_values = nmatrix_copy(&neurons);
    nmatrix_t prev_output = layer_get_neurons(model->output_layer);
    dense->weights = nmatrix_allocator(SHAPE(2, neurons.dims[0], prev_output.dims[0]));
    dense->transposed_weights = nmatrix_allocator(SHAPE(2, dense->weights.dims[1], dense->weights.dims[0]));
    dense->transposed_inputs = nmatrix_allocator(SHAPE(2, prev_output.dims[1], prev_output.dims[0]));
    dense->bias = nmatrix_allocator(SHAPE(2, neurons.dims[0], 1));
    dense->d_cost_wrt_input = nmatrix_allocator(SHAPE(2, prev_output.dims[0], 1));
    dense->d_cost_wrt_weight = nmatrix_allocator(SHAPE(2, dense->weights.dims[0], dense->weights.dims[1]));
    dense->d_cost_wrt_bias = nmatrix_allocator(SHAPE(2, dense->bias.dims[0], dense->bias.dims[1]));
    dense->d_cost_wrt_weight_sum = nmatrix_copy(&dense->d_cost_wrt_weight);
    dense->d_cost_wrt_bias_sum = nmatrix_copy(&dense->d_cost_wrt_bias);
    dense->model = model;

    dense->functions = dense_functions;

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_dropout(neural_network_model_t *model, float dropout) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);

    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = DROPOUT;
    dropout_layer_t *dropout_layer = &layer->layer.dropout;
    nmatrix_t prev_output = layer_get_neurons(model->output_layer);
    dropout_layer->output = nmatrix_allocator(SHAPE(2, prev_output.dims[0], prev_output.dims[1]));
    dropout_layer->functions = dropout_functions;
    dropout_layer->d_cost_wrt_input = nmatrix_allocator(SHAPE(2, prev_output.dims[0], prev_output.dims[1]));

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_activation(neural_network_model_t *model, layer_function_t functions) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    
    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = ACTIVATION;
    activation_layer_t *activation = &layer->layer.activation;
    nmatrix_t prev_output = layer_get_neurons(model->output_layer);
    activation->activated_values = nmatrix_allocator(SHAPE(2, prev_output.dims[0], prev_output.dims[1]));
    activation->functions = functions;
    activation->d_cost_wrt_input = nmatrix_allocator(SHAPE(2, prev_output.dims[0], prev_output.dims[1]));

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_output(neural_network_model_t *model, nmatrix_t (*make_guess)(layer_t*, nmatrix_t), layer_function_t functions, 
        float (*loss)(layer_t*, nmatrix_t)) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    // if (functions.back_propagation == output_functions_crossentropy.back_propagation) { // TODO I DONT THINK THIS MATTERS
    //     assert(make_guess == output_make_guess_softmax);
    // }
    
    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = OUTPUT;
    output_layer_t *output = &layer->layer.output;
    nmatrix_t prev_output = layer_get_neurons(model->output_layer);
    output->output_values = nmatrix_allocator(SHAPE(2, prev_output.dims[0], prev_output.dims[1]));
    output->make_guess = make_guess;
    output->functions = functions;
    output->d_cost_wrt_input = nmatrix_allocator(SHAPE(2, prev_output.dims[0], 1));
    output->guess = nmatrix_allocator(SHAPE(2, prev_output.dims[0], 1));
    output->loss = loss;

    model_add_layer(model, layer);
    return layer;
}

void model_initialize_matrix_normal_distribution(nmatrix_t matrix, float mean, float standard_deviation) {
    for (int i = 0; i < matrix.n_elements; i++) {
        matrix.matrix[i] = random_normal_distribution_BoxMullerTransform(standard_deviation) + mean;
    }
}

nmatrix_t model_predict(neural_network_model_t *model, nmatrix_t input, 
                        nmatrix_t output) {
    layer_t *current = model->input_layer;
    nmatrix_t prev_output = input;
    int num_iterations = model->num_layers-1;
    for (int layer_i = 0; layer_i < num_iterations; layer_i++) {
        assert(current->type != OUTPUT);
        
        // since the different layer structs are arranged in a way that the function pointers are in the same "locations"
        // this should work for all layers without having to use a switch
        prev_output = current->layer.input.functions.feed_forward(current, prev_output);
        current = current->next;
    }

    model->output_layer->layer.output.make_guess(model->output_layer, prev_output);
    nmatrix_memcpy(&current->layer.output.output_values, &prev_output);
    nmatrix_memcpy(&output, &prev_output);
    return output;
}

void model_back_propagate(neural_network_model_t *model, nmatrix_t expected_output, float learning_rate) {
    layer_t *current = model->output_layer;
    nmatrix_t d_cost_wrt_Y = expected_output;
    for (int layer_i = model->num_layers; layer_i > 1; layer_i--) {
        assert(current->type != INPUT);

        // works for all layers since the member variables align.
        // TODO in future, move "functions" outside and into the actual generic layer struct, not the union
        d_cost_wrt_Y = current->layer.dense.functions.back_propagation(current, d_cost_wrt_Y, learning_rate);
        current = current->prev;

        // printf("\n%s: de/dy: \n", get_layer_name(current->next));
        // matrix_print(d_cost_wrt_Y);
    }
}

void model_gradient_descent(neural_network_model_t *model) {
    layer_t *current = model->input_layer;
    for (int layer_i = 0; layer_i < model->num_layers; layer_i++) {
        if (current->type == DENSE) {
            nmatrix_sub(&current->layer.dense.weights, &current->layer.dense.d_cost_wrt_weight_sum, &current->layer.dense.weights);
            nmatrix_sub(&current->layer.dense.bias, &current->layer.dense.d_cost_wrt_bias_sum, &current->layer.dense.bias);

            nmatrix_memset(&current->layer.dense.d_cost_wrt_weight_sum, 0);
            nmatrix_memset(&current->layer.dense.d_cost_wrt_bias_sum, 0);
        }

        current = current->next;
    }
}

float model_train(neural_network_model_t *model, nmatrix_t *inputs, nmatrix_t *expected_outputs, unsigned int num_examples, float learning_rate) {
    nmatrix_t output = model->output_layer->layer.output.output_values;
    float avg_error = 0;
    model->is_training = true;
    for (int example_i = 0; example_i < num_examples; example_i++) {
        model_predict(model, inputs[example_i], output);
        // avg_error += output_cost_mean_squared(model->output_layer, expected_outputs[example_i]);
        avg_error += model->output_layer->layer.output.loss(model->output_layer, expected_outputs[example_i]);
        model_back_propagate(model, expected_outputs[example_i], learning_rate);
        model_gradient_descent(model);
    }
    avg_error /= (float) num_examples;
    // printf("train avg error=%f", (float) avg_error);
    model->is_training = false;
    return avg_error;
}

void model_test(neural_network_model_t *model, nmatrix_t *inputs, nmatrix_t *expected_outputs, unsigned int num_tests) {
    nmatrix_t output = model->output_layer->layer.output.output_values;
    int passed = 0;
    for (int test_i = 0; test_i < num_tests; test_i++) {
        model_predict(model, inputs[test_i], output);

        printf("input\n");
        nmatrix_print(&inputs[test_i]);
        printf("guess vs expected\n");
        nmatrix_print(&model->output_layer->layer.output.guess);
        nmatrix_print(&expected_outputs[test_i]);
        printf("\n");

        nmatrix_t guess = model->output_layer->layer.output.make_guess(model->output_layer, output);
        if (nmatrix_equal(&expected_outputs[test_i], &guess)) {
            passed++;
        }
    }

    // 2 decimal places
    float accuracy = ((int)(100.0 * (float) passed / (float) num_tests)) / 100.0;
    printf("accuracy: %f  passed=%d, total=%d\n", accuracy, passed, num_tests);
}

nmatrix_t model_calculate(neural_network_model_t *model) {
    layer_t *current = model->input_layer;
    nmatrix_t prev_output = model->input_layer->layer.input.input_values;
    for (int layer_i = 0; layer_i < model->num_layers-1; layer_i++) {
        assert(current->type != OUTPUT);
        
        // since the different layer structs are arranged in a way that the function pointers are in the same "locations"
        // this should work for all layers without having to use a switch
        prev_output = current->layer.input.functions.feed_forward(current, prev_output);
        current = current->next;
    }

    model->output_layer->layer.output.make_guess(model->output_layer, prev_output);
    nmatrix_memcpy(&current->layer.output.output_values, &prev_output);
    return current->layer.output.guess;
}

void training_info_free(training_info_t *training_info) {
    free_nmatrix_list(training_info->train_size, training_info->train_x);
    free_nmatrix_list(training_info->train_size, training_info->train_y);
    free_nmatrix_list(training_info->test_size, training_info->test_x);
    free_nmatrix_list(training_info->test_size, training_info->test_y);
}

void model_train_info(training_info_t *training_info) {
    neural_network_model_t *model = training_info->model;
    
    int batch_size = training_info->batch_size;
    unsigned int *epoch = &training_info->epoch;
    unsigned int target_epochs = training_info->target_epochs;
    unsigned int *train_index = &training_info->train_index;
    unsigned int train_size = training_info->train_size;
    float train_size_reciprocal = 1.0 / train_size;
    unsigned int *test_index = &training_info->test_index;
    unsigned int test_size = training_info->test_size;
    float test_size_reciprocal = 1.0 / test_size;

    output_layer_t output_layer = model->output_layer->layer.output;
    nmatrix_t *train_x = training_info->train_x;
    nmatrix_t *train_y = training_info->train_y;
    nmatrix_t *test_x = training_info->test_x;
    nmatrix_t *test_y = training_info->test_y;

    nmatrix_t actual_output = nmatrix_copy(&output_layer.output_values);
    int print_every = target_epochs < 10 ? 10 : target_epochs / 10;
    for (*epoch = 0; *epoch < target_epochs; (*epoch)++) {
        // perform training
        float avg_train_error = 0;
        int passed_train = 0;
        for (*train_index = 0; *train_index < train_size; (*train_index)++) {
            model_predict(model, train_x[*train_index], actual_output);
            avg_train_error += output_layer.loss(model->output_layer, train_y[*train_index]);
            model_back_propagate(model, train_y[*train_index], training_info->learning_rate);

            if ((1 + *train_index) % batch_size == 0 || *train_index == train_size-1) {
                model_gradient_descent(model);
            }

            nmatrix_t model_guess = output_layer.make_guess(model->output_layer, actual_output);
            if (nmatrix_equal(&train_y[*train_index], &model_guess)) {
                passed_train++;
            }
        }
        training_info->avg_train_error = avg_train_error * train_size_reciprocal;
        training_info->train_accuracy = ((int)(100000.0 * passed_train * train_size_reciprocal)) * 0.00001;

        // perform test
        float avg_test_error = 0;
        int passed_test = 0;
        for (*test_index = 0; *test_index < test_size; (*test_index)++) {
            model_predict(model, test_x[*test_index], actual_output);
            avg_test_error += output_layer.loss(model->output_layer, test_y[*test_index]);
            
            nmatrix_t model_guess = output_layer.make_guess(model->output_layer, actual_output);
            if (nmatrix_equal(&test_y[*test_index], &model_guess)) {
                passed_test++;
            }
        }

        training_info->avg_test_error = avg_test_error * test_size_reciprocal;
        training_info->test_accuracy = ((int)(100000.0 * passed_test * test_size_reciprocal)) * 0.00001;

        if ((((*epoch) + 1) % print_every == 0 && *epoch != 0) || *epoch == target_epochs - 1) {
            printf("==== Epoch %d ==== \ntrain_error: %f, train_accuracy: %f\ntest_error: %f, test_accuracy: %f (passed=%d)\n\n", (*epoch) + 1, 
                    training_info->avg_train_error, training_info->train_accuracy, 
                    training_info->avg_test_error, training_info->test_accuracy,
                    passed_test); 
        }

        // check if we can stop
        if (training_info->target_accuracy <= training_info->test_accuracy && training_info->target_accuracy <= training_info->train_accuracy) {
            break; // finish early
        }
    }
    nmatrix_free(&actual_output);
}

void model_test_info(training_info_t *training_info) {
    // perform test
    neural_network_model_t *model = training_info->model;
    output_layer_t output_layer = model->output_layer->layer.output;
    nmatrix_t actual_output = nmatrix_copy(&output_layer.output_values);
    nmatrix_t *test_x = training_info->test_x;
    nmatrix_t *test_y = training_info->test_y;
    unsigned int *test_index = &training_info->test_index;
    unsigned int test_size = training_info->test_size;
    float avg_test_error = 0;
    int passed_test = 0;
    for (*test_index = 0; *test_index < test_size; (*test_index)++) {
        model_predict(model, test_x[*test_index], actual_output);
        avg_test_error += output_layer.loss(model->output_layer, test_y[*test_index]);
        
        nmatrix_t model_guess = output_layer.make_guess(model->output_layer, actual_output);
        if (nmatrix_equal(&test_y[*test_index], &model_guess)) {
            passed_test++;
        }
    }

    training_info->avg_test_error = avg_test_error / (float) test_size;
    training_info->test_accuracy = ((int)(100.0 * (float) passed_test / (float) test_size)) / 100.0;

    nmatrix_free(&actual_output);
}



int unpack_one_hot_encoded(nmatrix_t one_hot_encoded) {
    assert(one_hot_encoded.n_dims == 1 || one_hot_encoded.dims[1] == 1);
    
    for (int i = 0; i < one_hot_encoded.n_elements; i++) {
        if (one_hot_encoded.matrix[i] != 0) {
            return i;
        }
    }
    assert(0);
}