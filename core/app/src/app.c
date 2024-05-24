#include <model/model.h>

#include <stdio.h>


int main() {
    // AND training
    model_t model = {
        .input_layer = NULL,
        .output_layer = NULL,
        .num_layers = 0
    };

    matrix_t *input = matrix_allocator(2, 1);
    matrix_t *dense_1 = matrix_allocator(2, 1);
    matrix_t *dense_2 = matrix_allocator(1, 1);

    layer_t *input_layer = layer_input(&model, input);
    layer_t *dense_layer_1 = layer_dense(&model, dense_1);
    layer_t *activation_layer_1 = layer_activation(&model, activation_feed_forward_sigmoid, activation_back_propagation_sigmoid);
    layer_t *dense_layer_2 = layer_dense(&model, dense_2);
    layer_t *activation_layer_2 = layer_activation(&model, activation_feed_forward_sigmoid, activation_back_propagation_sigmoid);
    layer_t *output_layer = layer_output(&model, output_make_guess_round, output_back_propagation_mean_squared);

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

    matrix_t **input_data = malloc(num_examples * sizeof(matrix_t*));
    matrix_t **output_data = malloc(num_examples * sizeof(matrix_t*));
    for (int i = 0; i < num_examples; i++) {
        input_data[i] = matrix_allocator(input_size, 1);
        output_data[i] = matrix_allocator(output_size, 1);
        matrix_set_values_to_fit(input_data[i], raw_input_data[i], input_size);
        matrix_set_values_to_fit(output_data[i], raw_output_data[i], output_size);
    }

    printf("\nInitial Test\n");
    model_test(&model, input_data, output_data, num_examples);

    const int num_epochs = 100000;
    const int num_epoch_prints = 10;
    const int epochs_print = num_epochs / num_epoch_prints;
    printf("Training epochs=%d\n", num_epochs);
    for (int i = 0; i < num_epochs; i++) {
        
        // printf("----\nepoch %d\n", i+1);
        double avg_error = model_train(&model, input_data, output_data, num_examples, 0.1);

        if (i != 0 && (i+1) % epochs_print == 0) {
            printf("----\nepoch %d\n", i+1);
            printf("avg error: %f\n", avg_error);
            printf("\ndense_layer_1 weights:\n");
            matrix_print(dense_layer_1->layer.dense.weights);
            printf("\ndense_layer_1 bias:\n");
            matrix_print(dense_layer_1->layer.dense.bias);
            
            printf("\ndense_layer_2 weights:\n");
            matrix_print(dense_layer_2->layer.dense.weights);
            printf("\ndense_layer_2 bias:\n");
            matrix_print(dense_layer_2->layer.dense.bias);
        }
    }

    printf("\nTesting\n");
    model_test(&model, input_data, output_data, num_examples);
    return 0;
}