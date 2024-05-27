#include <app/app.h>
#include <model/model.h>
#include <app/visualizer.h>

#include <stdio.h>
#include <pthread.h>

#include <util/profiler.h>
#include <util/debug_memory.h>

int main(void) {
    CLOCK_MARK
    
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, window_run, NULL);

    nn_AND();
    // nn_XOR();
    // nn_binary_digit_recognizer();
    
    pthread_join(thread_id, NULL);
    return EXIT_SUCCESS;
}

void nn_binary_digit_recognizer() {
    neural_network_model_t model_binary_digit_recognizer = {
        .input_layer = NULL,
        .output_layer = NULL,
        .num_layers = 0
    };

    // idea, 2 hidden layers, input matrix 10x10 => 100 x 1, output 2x1
}


void nn_XOR() {
    neural_network_model_t model_xor = {
        .input_layer = NULL,
        .output_layer = NULL,
        .num_layers = 0
    };

    mymatrix_t input = matrix_allocator(2, 1);
    mymatrix_t dense_1 = matrix_allocator(2, 1);
    mymatrix_t dense_2 = matrix_allocator(1, 1);

    layer_t *input_layer = layer_input(&model_xor, input);
    layer_t *dense_layer_1 = layer_dense(&model_xor, dense_1);
    layer_t *activation_layer_1 = layer_activation(&model_xor, activation_feed_forward_sigmoid, activation_back_propagation_sigmoid);
    layer_t *dense_layer_2 = layer_dense(&model_xor, dense_2);
    layer_t *activation_layer_2 = layer_activation(&model_xor, activation_feed_forward_sigmoid, activation_back_propagation_sigmoid);
    layer_t *output_layer = layer_output(&model_xor, output_make_guess_round, output_back_propagation_mean_squared);

    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.weights, 0, 0.2);    
    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.bias, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.weights, 0, 0.2);    
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.bias, 0, 0.2);

    const int num_examples = 4;
    const int input_size = 2;
    const int output_size = 1;
    double raw_input_data[][2] = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    double raw_output_data[][1] = {
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

    printf("\nInitial Test XOR\n");
    model_test(&model_xor, input_data, output_data, num_examples);

    const int num_epochs = 2000000;
    const int num_epoch_prints = 50;
    const int epochs_print = num_epoch_prints == 0 ? INT_MAX : num_epochs / num_epoch_prints;
    printf("Training epochs=%d\n", num_epochs);
    for (int i = 0; i < num_epochs; i++) {
        

        // printf("----\nepoch %d\n", i+1);
        double avg_error = model_train(&model_xor, input_data, output_data, num_examples, 0.1);

        if (i != 0 && (i+1) % epochs_print == 0) {
            printf("----\nepoch %d\n", i+1);
            printf("avg error: %f\n", avg_error);
            // printf("\ndense_layer_1 weights:\n");
            // matrix_print(dense_layer_1->layer.dense.weights);
            // printf("\ndense_layer_1 bias:\n");
            // matrix_print(dense_layer_1->layer.dense.bias);
            
            // printf("\ndense_layer_2 weights:\n");
            // matrix_print(dense_layer_2->layer.dense.weights);
            // printf("\ndense_layer_2 bias:\n");
            // matrix_print(dense_layer_2->layer.dense.bias);
        }
    }

    printf("\nTesting XOR\n");
    model_test(&model_xor, input_data, output_data, num_examples);


    matrix_free(input);
    matrix_free(dense_1);
    matrix_free(dense_2);
    model_free(&model_xor);

    for (int i = 0; i < num_examples; i++) {
        matrix_free(input_data[i]);
        matrix_free(output_data[i]);
    }
    free(input_data);
    free(output_data);
}

void nn_AND() {
    neural_network_model_t model_and = {
        .input_layer = NULL,
        .output_layer = NULL,
        .num_layers = 0
    };

    mymatrix_t input = matrix_allocator(2, 1);
    mymatrix_t dense_1 = matrix_allocator(2, 1);
    mymatrix_t dense_2 = matrix_allocator(1, 1);

    layer_t *input_layer = layer_input(&model_and, input);
    layer_t *dense_layer_1 = layer_dense(&model_and, dense_1);
    layer_t *activation_layer_1 = layer_activation(&model_and, activation_feed_forward_sigmoid, activation_back_propagation_sigmoid);
    layer_t *dense_layer_2 = layer_dense(&model_and, dense_2);
    layer_t *activation_layer_2 = layer_activation(&model_and, activation_feed_forward_sigmoid, activation_back_propagation_sigmoid);
    layer_t *output_layer = layer_output(&model_and, output_make_guess_round, output_back_propagation_mean_squared);

    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.weights, 0, 0.2);    
    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.bias, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.weights, 0, 0.2);    
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.bias, 0, 0.2);

    const int num_examples = 4;
    const int input_size = 2;
    const int output_size = 1;
    double raw_input_data[][2] = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    double raw_output_data[][1] = {
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

    printf("\nInitial Test AND\n");
    model_test(&model_and, input_data, output_data, num_examples);

    const int num_epochs = 2000000;
    const int num_epoch_prints = 50;
    const int epochs_print = num_epoch_prints == 0 ? INT_MAX : num_epochs / num_epoch_prints;
    printf("Training epochs=%d\n", num_epochs);
    for (int i = 0; i < num_epochs; i++) {
        

        // printf("----\nepoch %d\n", i+1);
        double avg_error = model_train(&model_and, input_data, output_data, num_examples, 0.1);

        if (i != 0 && (i+1) % epochs_print == 0) {
            printf("----\nepoch %d\n", i+1);
            printf("avg error: %f\n", avg_error);
            // printf("\ndense_layer_1 weights:\n");
            // matrix_print(dense_layer_1->layer.dense.weights);
            // printf("\ndense_layer_1 bias:\n");
            // matrix_print(dense_layer_1->layer.dense.bias);
            
            // printf("\ndense_layer_2 weights:\n");
            // matrix_print(dense_layer_2->layer.dense.weights);
            // printf("\ndense_layer_2 bias:\n");
            // matrix_print(dense_layer_2->layer.dense.bias);
        }
    }

    printf("\nTesting AND\n");
    model_test(&model_and, input_data, output_data, num_examples);


    matrix_free(input);
    matrix_free(dense_1);
    matrix_free(dense_2);
    model_free(&model_and);

    for (int i = 0; i < num_examples; i++) {
        matrix_free(input_data[i]);
        matrix_free(output_data[i]);
    }
    free(input_data);
    free(output_data);
}