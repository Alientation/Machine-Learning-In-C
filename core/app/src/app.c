#include <app/app.h>
#include <app/visualizer.h>

#include <stdio.h>
#include <pthread.h>

#include <util/profiler.h>
#include <util/debug_memory.h>

void free_matrix_list(mymatrix_t *matrix_list, int size) {
    for (int i = 0; i < size; i++) {
        matrix_free(matrix_list[i]);
    }

    free(matrix_list);
}

int main(void) {
    CLOCK_MARK
    neural_network_model_t nnmodel;
    // todo, app should connect to visualizer's start button and only start the training process when it is clicked

    // training_info_t training_info = nn_XOR(&nnmodel);
    training_info_t training_info = nn_digit_recognizer(&nnmodel);
    training_info.model = &nnmodel;
    model_calculate(&nnmodel);

    pthread_t thread_id;
    visualizer_argument_t vis_args = {
        .model = &nnmodel,
        .training_info = training_info,
        .model_name = "Digit Recognizer",
        .allow_drawing_panel_as_model_input = true,
        .label_guess = NULL,
    };
    pthread_create(&thread_id, NULL, window_run, &vis_args);

    // clean up
    pthread_join(thread_id, NULL);   
    model_free(&nnmodel);

    free_matrix_list(training_info.train_x, training_info.train_size);
    free_matrix_list(training_info.train_y, training_info.train_size);
    free_matrix_list(training_info.test_x, training_info.test_size);
    free_matrix_list(training_info.test_y, training_info.test_size);

    return EXIT_SUCCESS;
}

training_info_t nn_digit_recognizer(neural_network_model_t *model_digit) {
    // idea, 2 hidden layers, input matrix 28x28 => 784 x 1 -> 32 x 1 -> 16 x 1 -> 10 x 1 output
    model_digit->input_layer = NULL;
    model_digit->output_layer = NULL;
    model_digit->num_layers = 0;

    mymatrix_t input = matrix_allocator(784, 1);
    mymatrix_t dense_1 = matrix_allocator(32, 1);
    mymatrix_t dense_2 = matrix_allocator(16, 1);
    mymatrix_t output = matrix_allocator(10, 1);

    layer_t *input_layer = layer_input(model_digit, input);
    layer_t *dense_layer_1 = layer_dense(model_digit, dense_1);
    layer_t *activation_layer_1 = layer_activation(model_digit, activation_functions_relu);
    layer_t *dense_layer_2 = layer_dense(model_digit, dense_2);
    layer_t *activation_layer_2 = layer_activation(model_digit, activation_functions_relu);
    layer_t *dense_layer_3 = layer_dense(model_digit, output);
    layer_t *activation_layer_3 = layer_activation(model_digit, activation_functions_relu);
    layer_t *output_layer = layer_output(model_digit, output_make_guess_softmax, output_functions_crossentropy);

    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.weights, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.bias, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.weights, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.bias, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_3->layer.dense.weights, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_3->layer.dense.bias, 0, 0.2);

    training_info_t training_info;
    training_info.in_progress = false;
    training_info.train_size = 0;
    training_info.train_x = NULL;
    training_info.train_y = NULL;
    training_info.test_size = 0;
    training_info.test_x = NULL;
    training_info.test_y = NULL;

    training_info.batch_size = 1;
    training_info.learning_rate = 0.1;
    training_info.target_epochs = 200;
    training_info.target_accuracy = 1;

    matrix_free(input);
    matrix_free(dense_1);
    matrix_free(dense_2);
    matrix_free(output);

    return training_info;
}


training_info_t nn_XOR(neural_network_model_t *model_xor) {
    model_xor->input_layer = NULL;
    model_xor->output_layer = NULL;
    model_xor->num_layers = 0;

    mymatrix_t input = matrix_allocator(2, 1);
    mymatrix_t dense_1 = matrix_allocator(2, 1);
    mymatrix_t output = matrix_allocator(1, 1);

    layer_t *input_layer = layer_input(model_xor, input);
    layer_t *dense_layer_1 = layer_dense(model_xor, dense_1);
    layer_t *activation_layer_1 = layer_activation(model_xor, activation_functions_sigmoid);
    layer_t *dense_layer_2 = layer_dense(model_xor, output);
    layer_t *activation_layer_2 = layer_activation(model_xor, activation_functions_sigmoid);
    layer_t *output_layer = layer_output(model_xor, output_make_guess_round, output_functions_meansquared);

    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.weights, 0, 0.2);    
    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.bias, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.weights, 0, 0.2);    
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.bias, 0, 0.2);

    const int num_examples = 4;
    const int input_size = 2;
    const int output_size = 1;
    float raw_input_data[][2] = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    float raw_output_data[][1] = {
        {0},
        {1},
        {1},
        {0}
    };

    mymatrix_t *input_data = malloc(num_examples * sizeof(mymatrix_t));
    mymatrix_t *output_data = malloc(num_examples * sizeof(mymatrix_t));
    for (int i = 0; i < num_examples; i++) {
        input_data[i] = matrix_allocator(input_size, 1);
        output_data[i] = matrix_allocator(output_size, 1);
        matrix_set_values_to_fit(input_data[i], raw_input_data[i], input_size);
        matrix_set_values_to_fit(output_data[i], raw_output_data[i], output_size);
    }

    training_info_t training_info;
    training_info.in_progress = false;
    training_info.train_size = num_examples;
    training_info.train_x = input_data;
    training_info.train_y = output_data;
    training_info.test_size = 0;
    training_info.test_x = NULL;
    training_info.test_y = NULL;

    training_info.batch_size = 1;
    training_info.learning_rate = 0.1;
    training_info.target_epochs = 2000000;
    training_info.target_accuracy = 1;

    matrix_free(input);
    matrix_free(dense_1);
    matrix_free(output);

    return training_info;
}

training_info_t nn_AND(neural_network_model_t *model_and) {
    model_and->input_layer = NULL;
    model_and->output_layer = NULL;
    model_and->num_layers = 0;

    mymatrix_t input = matrix_allocator(2, 1);
    mymatrix_t dense_1 = matrix_allocator(2, 1);
    mymatrix_t output = matrix_allocator(1, 1);

    layer_t *input_layer = layer_input(model_and, input);
    layer_t *dense_layer_1 = layer_dense(model_and, dense_1);
    layer_t *activation_layer_1 = layer_activation(model_and, activation_functions_sigmoid);
    layer_t *dense_layer_2 = layer_dense(model_and, output);
    layer_t *activation_layer_2 = layer_activation(model_and, activation_functions_sigmoid);
    layer_t *output_layer = layer_output(model_and, output_make_guess_round, output_functions_meansquared);

    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.weights, 0, 0.2);    
    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.bias, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.weights, 0, 0.2);    
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.bias, 0, 0.2);

    const int num_examples = 4;
    const int input_size = 2;
    const int output_size = 1;
    float raw_input_data[][2] = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    float raw_output_data[][1] = {
        {0},
        {0},
        {0},
        {1}
    };

    mymatrix_t *input_data = malloc(num_examples * sizeof(mymatrix_t));
    mymatrix_t *output_data = malloc(num_examples * sizeof(mymatrix_t));
    for (int i = 0; i < num_examples; i++) {
        input_data[i] = matrix_allocator(input_size, 1);
        output_data[i] = matrix_allocator(output_size, 1);
        matrix_set_values_to_fit(input_data[i], raw_input_data[i], input_size);
        matrix_set_values_to_fit(output_data[i], raw_output_data[i], output_size);
    }

    training_info_t training_info;
    training_info.in_progress = false;
    training_info.train_size = num_examples;
    training_info.train_x = input_data;
    training_info.train_y = output_data;
    training_info.test_size = 0;
    training_info.test_x = NULL;
    training_info.test_y = NULL;

    training_info.batch_size = 1;
    training_info.learning_rate = 0.1;
    training_info.target_epochs = 2000000;
    training_info.target_accuracy = 1;

    matrix_free(input);
    matrix_free(dense_1);
    matrix_free(output);

    return training_info;
}