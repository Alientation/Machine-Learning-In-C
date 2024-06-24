#include <model/model.h>
#include <util/math.h>
#include <unistd.h>

#include <stdio.h>
#include <math.h>

#include <util/debug_memory.h>

nmatrix_t feedforward_donothing(layer_t *this, nmatrix_t input) {
    assert(0);
    return input;
}

nmatrix_t backpropagation_donothing(layer_t *this, nmatrix_t d_error_wrt_output, float learning_rate) {
    assert(0);
    return d_error_wrt_output;
}

nmatrix_t input_feed_forward(layer_t *this, nmatrix_t input) {
    nmatrix_memcpy(&this->layer.input.input_values, &input);
    return this->layer.input.input_values;
}
const layer_function_t input_functions = {
    .back_propagation = backpropagation_donothing,
    .feed_forward = input_feed_forward
};

nmatrix_t dense_feed_forward(layer_t *this, nmatrix_t input) { 
    nmatrix_multiply(&this->layer.dense.weights, &input, &this->layer.dense.activation_values);
    nmatrix_add(&this->layer.dense.activation_values, &this->layer.dense.bias, &this->layer.dense.activation_values);
    return this->layer.dense.activation_values;
}

nmatrix_t dense_back_propagation(layer_t *this, nmatrix_t d_error_wrt_output, float learning_rate) {
    // transpose input matrix
    // todo see if this needs optimization, likely does since allocation on every iteration is a lot of work
    nmatrix_t X = layer_get_neurons(this->prev);
    nmatrix_t X_T = this->layer.dense.transposed_inputs;
    nmatrix_transpose(&X, &X_T);

    nmatrix_t W = this->layer.dense.weights;
    nmatrix_t W_T = this->layer.dense.transposed_weights;
    nmatrix_transpose(&W, &W_T);

    nmatrix_multiply(&d_error_wrt_output, &X_T, &this->layer.dense.d_cost_wrt_weight);
    nmatrix_memcpy(&this->layer.dense.d_cost_wrt_bias, &d_error_wrt_output);
    nmatrix_multiply(&W_T, &d_error_wrt_output, &this->layer.dense.d_cost_wrt_input);

    // apply adjustments
    nmatrix_multiply_scalar(&this->layer.dense.d_cost_wrt_weight, learning_rate, &this->layer.dense.d_cost_wrt_weight);
    nmatrix_multiply_scalar(&this->layer.dense.d_cost_wrt_bias, learning_rate, &this->layer.dense.d_cost_wrt_bias);
    
    nmatrix_sub(&this->layer.dense.weights, &this->layer.dense.d_cost_wrt_weight, &this->layer.dense.weights);
    nmatrix_sub(&this->layer.dense.bias, &this->layer.dense.d_cost_wrt_bias, &this->layer.dense.bias);

    return this->layer.dense.d_cost_wrt_input;
}

const layer_function_t dense_functions = {
    .feed_forward = dense_feed_forward,
    .back_propagation = dense_back_propagation,
};

nmatrix_t dropout_feedforward(layer_t *this, nmatrix_t input) {
    float keep = 1 - this->layer.dropout.dropout;
    if (keep < 1) {
        if (this->layer.dropout.model->is_training) {
            for (int i = 0; i < input.n_elements; i++) {
                this->layer.dropout.output.matrix[i] = input.matrix[i] * (random_uniform_range(1) <= keep);
            }
        } else {
            for (int i = 0; i < input.n_elements; i++) {
                this->layer.dropout.output.matrix[i] = input.matrix[i] * keep;
            }
        }
    } else {
        nmatrix_memcpy(&this->layer.dropout.output, &input);
    }
}

nmatrix_t dropout_backpropagation(layer_t *this, nmatrix_t d_error_wrt_output, float learning_rate) {
    nmatrix_memcpy(&this->layer.dropout.d_cost_wrt_input, &d_error_wrt_output);
    return this->layer.dropout.d_cost_wrt_input;
}

const layer_function_t dropout_functions = {
    .feed_forward = dropout_feedforward,
    .back_propagation = dropout_backpropagation,
};

nmatrix_t activation_feed_forward_sigmoid(layer_t *this, nmatrix_t input) {
    // nmatrix_for_each_operator(&input, sigmoid, &this->layer.activation.activated_values);
    for (int i = 0; i < input.n_elements; i++) {
        this->layer.activation.activated_values.matrix[i] = 1. / (1 + exp(-input.matrix[i]));
    }
    
    return this->layer.activation.activated_values;
}

nmatrix_t activation_feed_forward_relu(layer_t *this, nmatrix_t input) {
    // nmatrix_for_each_operator(&input, relu, &this->layer.activation.activated_values);
    for (int i = 0; i < input.n_elements; i++) {
        this->layer.activation.activated_values.matrix[i] = fmax(0, input.matrix[i]);
    }
    
    return this->layer.activation.activated_values;
}

nmatrix_t activation_feed_forward_softmax(layer_t *this, nmatrix_t input) {
    float sum = 0;
    for (int i = 0; i < input.n_elements; i++) {
        sum += exp(input.matrix[i]);
    }

    float inv_sum = 1.0/sum;
    for (int i = 0; i < input.n_elements; i++) {
        this->layer.activation.activated_values.matrix[i] = exp(input.matrix[i]) * inv_sum;
    }
    return this->layer.activation.activated_values;
}

nmatrix_t activation_back_propagation_sigmoid(layer_t *this, nmatrix_t d_cost_wrt_output, float learning_rate) {
    nmatrix_t X = layer_get_neurons(this->prev);

    // nmatrix_for_each_operator(&X, sigmoid_prime, &this->layer.activation.activated_values);
    for (int i = 0; i < X.n_elements; i++) {
        float z = 1.0 / (1 + exp(-X.matrix[i]));
        this->layer.activation.activated_values.matrix[i] = z * (1-z);
    }

    nmatrix_elementwise_multiply(&d_cost_wrt_output, &this->layer.activation.activated_values, &this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}

nmatrix_t activation_back_propagation_relu(layer_t *this, nmatrix_t d_cost_wrt_output, float learning_rate) {
    nmatrix_t X = layer_get_neurons(this->prev);

    // nmatrix_for_each_operator(&X, relu_prime, &this->layer.activation.activated_values);
    for (int i = 0; i < X.n_elements; i++) {
        this->layer.activation.activated_values.matrix[i] = X.matrix[i] > 0;
    }

    nmatrix_elementwise_multiply(&d_cost_wrt_output, &this->layer.activation.activated_values, &this->layer.activation.activated_values);
    return this->layer.activation.activated_values;
}

// this uses the trick described here  https://stackoverflow.com/questions/58461808/understanding-backpropagation-with-softmax
// which requires another layer before this to compute the partial derivatives
nmatrix_t activation_back_propagation_softmax(layer_t *this, nmatrix_t d_cost_wrt_output, float learning_rate) {
    nmatrix_memcpy(&this->layer.activation.activated_values, &d_cost_wrt_output);
    return this->layer.activation.activated_values;
}

const layer_function_t activation_functions_sigmoid = {
    .feed_forward = activation_feed_forward_sigmoid,
    .back_propagation = activation_back_propagation_sigmoid,
};

const layer_function_t activation_functions_relu = {
    .feed_forward = activation_feed_forward_relu,
    .back_propagation = activation_back_propagation_relu,
};

const layer_function_t activation_functions_softmax = {
    .feed_forward = activation_feed_forward_softmax,
    .back_propagation = activation_back_propagation_softmax,
};

// todo TANH

nmatrix_t output_make_guess_one_hot_encoded(layer_t *this, nmatrix_t output) {
    float max = -INFINITY;
    for (int i = 0; i < output.n_elements; i++) {
        max = fmax(max, output.matrix[i]);
    }

    for (int i = 0; i < output.n_elements; i++) {
        this->layer.output.guess.matrix[i] = (output.matrix[i] == max) ? 1 : 0;
    }
    return this->layer.output.guess;
}

nmatrix_t output_make_guess_passforward(layer_t *this, nmatrix_t output) {
    nmatrix_memcpy(&this->layer.output.guess, &output);
    return this->layer.output.guess;
}

nmatrix_t output_make_guess_round(layer_t *this, nmatrix_t output) {
    for (int i = 0; i < output.n_elements; i++) {
        this->layer.output.guess.matrix[i] = round(output.matrix[i]);
    }
    return this->layer.output.guess;
}

nmatrix_t output_make_guess_softmax(layer_t *this, nmatrix_t output) {
    float sum = 0;
    for (int i = 0; i < output.n_elements; i++) {
        sum += exp(output.matrix[i]);
    }

    float inv_sum = 1.0 / sum;
    for (int i = 0; i < output.n_elements; i++) {
        this->layer.output.guess.matrix[i] = exp(output.matrix[i]) * inv_sum;
    }
    return this->layer.output.guess;
}

nmatrix_t output_back_propagation_mean_squared(layer_t *this, nmatrix_t expected_output, float learning_rate) {
    nmatrix_t output = this->layer.output.output_values;
    nmatrix_sub(&output, &expected_output, &this->layer.output.d_cost_wrt_input);
    nmatrix_multiply_scalar(&this->layer.output.d_cost_wrt_input, 2.0 / (float) output.n_elements, &this->layer.output.d_cost_wrt_input);
    return this->layer.output.d_cost_wrt_input;
}

// Categorical Cross Entropy with Softmax
// gradient = Y - y*
nmatrix_t output_back_propagation_categorical_cross_entropy(layer_t *this, nmatrix_t expected_output, float learning_rate) {
    nmatrix_t output = this->layer.output.output_values;
    nmatrix_sub(&output, &expected_output, &this->layer.output.d_cost_wrt_input);
    return this->layer.output.d_cost_wrt_input;
}

float output_cost_mean_squared(layer_t *this, nmatrix_t expected_output) {
    float mean_squared = 0;
    nmatrix_t actual_output = layer_get_neurons(this);
    for (int i = 0; i < actual_output.n_elements; i++) {
        mean_squared += pow(expected_output.matrix[i] - actual_output.matrix[i], 2);
    }
    mean_squared /= expected_output.n_elements;
    return mean_squared;
}

const float epsilon = 0.0001;
float output_cost_categorical_cross_entropy(layer_t *this, nmatrix_t expected_output) {
    float cross_entropy = 0;
    nmatrix_t actual_output = this->layer.output.guess; // USES guess (assumption that softmax is used)
    for (int i = 0; i < actual_output.n_elements; i++) {
        cross_entropy += expected_output.matrix[i] * log10(actual_output.matrix[i] + epsilon);
    }
    return -cross_entropy;
}

const layer_function_t output_functions_meansquared = {
    .feed_forward = feedforward_donothing,
    .back_propagation = output_back_propagation_mean_squared,
};
const layer_function_t output_functions_crossentropy = {
    .feed_forward = feedforward_donothing,
    .back_propagation = output_back_propagation_categorical_cross_entropy
};



char* get_layer_name(layer_t *layer) {
    switch (layer->type) {
        case INPUT:
            return "Input";
        case DENSE:
            return "Dense";
        case DROPOUT:
            return "Dropout";
        case ACTIVATION:
            return "Activation";
        case OUTPUT:
            return "Output";
        default:
            assert(0);
    }
}

char* get_activation_function_name(activation_layer_t *layer) {
    if (layer->functions.feed_forward == activation_feed_forward_sigmoid) {
        return "Sigmoid";
    } else if (layer->functions.feed_forward == activation_feed_forward_relu) {
        return "RELU";
    } else if (layer->functions.feed_forward == activation_feed_forward_softmax) {
        return "Softmax";
    } else {
        assert(0);
    }
}

char* get_output_function_name(output_layer_t *layer) {
    if (layer->functions.back_propagation == output_back_propagation_mean_squared) {
        return "Mean Squared Loss";
    } else if (layer->functions.back_propagation == output_back_propagation_categorical_cross_entropy) {
        return "Cross Entropy Loss";
    } else {
        assert(0);
    }
}

char* get_output_guess_function_name(output_layer_t *layer) {
    if (layer->make_guess == output_make_guess_one_hot_encoded) {
        return "One Hot Encoded";
    } else if (layer->make_guess == output_make_guess_passforward) {
        return "Passforward";
    } else if (layer->make_guess == output_make_guess_round) {
        return "Round";
    } else if (layer->make_guess == output_make_guess_softmax) {
        return "Softmax";
    } else {
        assert(0);
    }
}

void layer_free(layer_t *layer) {
    switch (layer->type) {
        case INPUT:
            nmatrix_free(&layer->layer.input.input_values);
            break;
        case DENSE:
            nmatrix_free(&layer->layer.dense.activation_values);
            nmatrix_free(&layer->layer.dense.weights);
            nmatrix_free(&layer->layer.dense.bias);
            nmatrix_free(&layer->layer.dense.d_cost_wrt_input);
            nmatrix_free(&layer->layer.dense.d_cost_wrt_weight);
            nmatrix_free(&layer->layer.dense.d_cost_wrt_bias);
            break;
        case DROPOUT:
            nmatrix_free(&layer->layer.dropout.output);
            nmatrix_free(&layer->layer.dropout.d_cost_wrt_input);
        case ACTIVATION:
            nmatrix_free(&layer->layer.activation.activated_values);
            nmatrix_free(&layer->layer.activation.d_cost_wrt_input);
            break;
        case OUTPUT:
            nmatrix_free(&layer->layer.output.output_values);
            nmatrix_free(&layer->layer.output.guess);
            nmatrix_free(&layer->layer.output.d_cost_wrt_input);
            break;
    }
    free(layer);
}

nmatrix_t layer_get_neurons(layer_t *layer) {
    switch (layer->type) {
        case INPUT:
            return layer->layer.input.input_values;
        case DROPOUT:
            return layer->layer.dropout.output;
        case DENSE:
            return layer->layer.dense.activation_values;
        case ACTIVATION:
            return layer->layer.activation.activated_values;
        case OUTPUT:
            return layer->layer.output.guess;
        default:
            assert(0);
    }
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
    dense->weights = nmatrix_allocator(2, neurons.dims[0], prev_output.dims[0]);
    dense->transposed_weights = nmatrix_allocator(2, dense->weights.dims[1], dense->weights.dims[0]);
    dense->transposed_inputs = nmatrix_allocator(2, prev_output.dims[1], prev_output.dims[0]);
    dense->bias = nmatrix_allocator(2, neurons.dims[0], 1);
    dense->d_cost_wrt_input = nmatrix_allocator(2, prev_output.dims[0], 1);
    dense->d_cost_wrt_weight = nmatrix_allocator(2, dense->weights.dims[0], dense->weights.dims[1]);
    dense->d_cost_wrt_bias = nmatrix_allocator(2, dense->bias.dims[0], dense->bias.dims[1]);
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
    dropout_layer->output = nmatrix_allocator(2, prev_output.dims[0], prev_output.dims[1]);
    dropout_layer->functions = dropout_functions;
    dropout_layer->d_cost_wrt_input = nmatrix_allocator(2, prev_output.dims[0], prev_output.dims[1]);

    model_add_layer(model, layer);
    return layer;
}

layer_t* layer_activation(neural_network_model_t *model, layer_function_t functions) {
    assert(model->num_layers > 0 && model->input_layer != NULL && model->input_layer->type == INPUT);
    
    layer_t *layer = malloc(sizeof(layer_t));
    layer->type = ACTIVATION;
    activation_layer_t *activation = &layer->layer.activation;
    nmatrix_t prev_output = layer_get_neurons(model->output_layer);
    activation->activated_values = nmatrix_allocator(2, prev_output.dims[0], prev_output.dims[1]);
    activation->functions = functions;
    activation->d_cost_wrt_input = nmatrix_allocator(2, prev_output.dims[0], prev_output.dims[1]);

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
    output->output_values = nmatrix_allocator(2, prev_output.dims[0], prev_output.dims[1]);
    output->make_guess = make_guess;
    output->functions = functions;
    output->d_cost_wrt_input = nmatrix_allocator(2, prev_output.dims[0], 1);
    output->guess = nmatrix_allocator(2, prev_output.dims[0], 1);
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

float model_train(neural_network_model_t *model, nmatrix_t *inputs, nmatrix_t *expected_outputs, unsigned int num_examples, float learning_rate) {
    nmatrix_t output = model->output_layer->layer.output.output_values;
    float avg_error = 0;
    model->is_training = true;
    for (int example_i = 0; example_i < num_examples; example_i++) {
        model_predict(model, inputs[example_i], output);
        // avg_error += output_cost_mean_squared(model->output_layer, expected_outputs[example_i]);
        avg_error += model->output_layer->layer.output.loss(model->output_layer, expected_outputs[example_i]);
        model_back_propagate(model, expected_outputs[example_i], learning_rate);
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
    // todo support batch size in future
    assert(training_info->batch_size == 1);
    neural_network_model_t *model = training_info->model;
    
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