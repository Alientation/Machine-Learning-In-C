#include <model/model.h>


int main() {
    // AND training
    model_t model = {
        .input_layer = NULL,
        .output_layer = NULL,
        .num_layers = 0
    };

    matrix_t *input = matrix_allocator(2, 1);
    // matrix_t *dense = matrix_allocator(2, 1);
    matrix_t *output = matrix_allocator(1, 1);

    layer_t *input_layer = layer_input(&model, input);
    // layer_t *dense_layer = layer_dense(&model, dense);
    layer_t *output_layer = layer_output(&model, output_make_guess_one_hot_encoded, output_back_propagation_mean_squared);

    // model_initialize_matrix_normal_distribution(dense_layer->layer.dense.weights, 0, 0.02);    
    // model_initialize_matrix_normal_distribution(dense_layer->layer.dense.bias, 0, 0.02);    




    return 0;
}