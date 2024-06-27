#include <app/app.h>
#include <app/visualizer.h>

#include <stdio.h>
#include <pthread.h>

#include <util/profiler.h>
#include <util/debug_memory.h>

#define SHAPE(...) nshape_constructor(__VA_ARGS__)

static const char* digit_outputs[10] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};

static const char* xor_outputs[2] = {"0", "1"};

static const training_info_t DEFAULT_TRAIN_INFO = {
    .in_progress = false,
    .train_size = 0,
    .train_x = NULL,
    .train_y = NULL,
    .test_size = 0,
    .test_x = NULL,
    .test_y = NULL,
    .train_accuracy = 0,
    .avg_train_error = 0,
    .train_correct = 0,
    .test_accuracy = 0,
    .avg_test_error = 0,
    .test_correct = 0,
    .batch_size = 1,
    .learning_rate = .01,
    .target_epochs = 1,
    .target_accuracy = 1,
};

int main(void) {
    CLOCK_MARK
    neural_network_model_t nnmodel = {
        .is_training = false,
        .batch_size = 1,
    };
    pthread_t thread_id;

    // training_info_t training_info = nn_XOR(&nnmodel);
    // training_info.model = &nnmodel;
    // model_calculate(&nnmodel);
    // visualizer_argument_t vis_args = {
    //     .model = &nnmodel,
    //     .training_info = &training_info,
    //     .is_batch_size_active = false,
    //     .is_learning_rate_active = false,
    //     .is_target_epochs_active = false,
    //     .is_target_accuracy_active = false,
    //     .model_name = "XOR",
    //     .output_labels = xor_outputs,
    //     .num_labels = 2,
    //     .default_dataset_directory = "",
    //     .allow_drawing_panel_as_model_input = false,
    // };
    
    training_info_t training_info = nn_digit_recognizer(&nnmodel);
    training_info.model = &nnmodel;
    model_calculate(&nnmodel);

    visualizer_argument_t vis_args = {
        .model = &nnmodel,
        .training_info = &training_info,
        .is_batch_size_active = false,
        .is_learning_rate_active = false,
        .is_target_epochs_active = false,
        .is_target_accuracy_active = false,

        .model_name = "Digit Recognizer",
        .output_labels = digit_outputs,
        .num_labels = 10,
        .default_dataset_directory = "images\\digits",
        .allow_drawing_panel_as_model_input = true,
    };

    pthread_create(&thread_id, NULL, window_run, &vis_args);

    // clean up
    pthread_join(thread_id, NULL);   
    model_free(&nnmodel);

    training_info_free(&training_info);

    return EXIT_SUCCESS;
}

training_info_t nn_digit_recognizer(neural_network_model_t *model_digit) {
    // idea, 2 hidden layers, input matrix 28x28 => 784 x 1 -> 32 x 1 -> 16 x 1 -> 10 x 1 output
    model_digit->input_layer = NULL;
    model_digit->output_layer = NULL;
    model_digit->num_layers = 0;

    nmatrix_t input = nmatrix_allocator(SHAPE(2, 784, 1));
    nmatrix_t dense_1 = nmatrix_allocator(SHAPE(2, 32, 1));
    nmatrix_t dense_2 = nmatrix_allocator(SHAPE(2, 16, 1));
    nmatrix_t output = nmatrix_allocator(SHAPE(2, 10, 1));

    layer_t *input_layer = layer_input(model_digit, input);
    layer_t *dense_layer_1 = layer_dense(model_digit, dense_1);
    layer_t *activation_layer_1 = layer_activation(model_digit, activation_functions_relu);
    // layer_t *dropout_layer_1 = layer_dropout(model_digit, 0.2);
    layer_t *dense_layer_2 = layer_dense(model_digit, dense_2);
    layer_t *activation_layer_2 = layer_activation(model_digit, activation_functions_sigmoid);
    // layer_t *dropout_layer_2 = layer_dropout(model_digit, 0.5);
    layer_t *dense_layer_3 = layer_dense(model_digit, output);
    layer_t *activation_layer_3 = layer_activation(model_digit, activation_functions_softmax);
    layer_t *output_layer = layer_output(model_digit, output_make_guess_one_hot_encoded, output_functions_crossentropy, output_cost_categorical_cross_entropy);
    
    model_initialize_matrix_normal_distribution(dense_layer_1->layer.dense.weights, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_2->layer.dense.weights, 0, 0.2);
    model_initialize_matrix_normal_distribution(dense_layer_3->layer.dense.weights, 0, 0.2);

    nmatrix_free(&input);
    nmatrix_free(&dense_1);
    nmatrix_free(&dense_2);
    nmatrix_free(&output);

    return DEFAULT_TRAIN_INFO;
}


training_info_t nn_XOR(neural_network_model_t *model_xor) {
    model_xor->input_layer = NULL;
    model_xor->output_layer = NULL;
    model_xor->num_layers = 0;

    nmatrix_t input = nmatrix_allocator(SHAPE(2, 2, 1));
    nmatrix_t dense_1 = nmatrix_allocator(SHAPE(2, 2, 1));
    nmatrix_t output = nmatrix_allocator(SHAPE(2, 1, 1));

    layer_t *input_layer = layer_input(model_xor, input);
    layer_t *dense_layer_1 = layer_dense(model_xor, dense_1);
    layer_t *activation_layer_1 = layer_activation(model_xor, activation_functions_sigmoid);
    layer_t *dense_layer_2 = layer_dense(model_xor, output);
    layer_t *activation_layer_2 = layer_activation(model_xor, activation_functions_sigmoid);
    layer_t *output_layer = layer_output(model_xor, output_make_guess_round, output_functions_meansquared, output_cost_mean_squared);

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

    nmatrix_t *input_data = malloc(num_examples * sizeof(nmatrix_t));
    nmatrix_t *output_data = malloc(num_examples * sizeof(nmatrix_t));
    for (int i = 0; i < num_examples; i++) {
        input_data[i] = nmatrix_allocator(SHAPE(2, input_size, 1));
        output_data[i] = nmatrix_allocator(SHAPE(2, output_size, 1));
        nmatrix_set_values_to_fit(&input_data[i], input_size, raw_input_data[i]);
        nmatrix_set_values_to_fit(&output_data[i], output_size, raw_output_data[i]);
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
    training_info.target_epochs = 200;
    training_info.target_accuracy = 1;

    nmatrix_free(&input);
    nmatrix_free(&dense_1);
    nmatrix_free(&output);

    return training_info;
}

training_info_t nn_AND(neural_network_model_t *model_and) {
    model_and->input_layer = NULL;
    model_and->output_layer = NULL;
    model_and->num_layers = 0;

    nmatrix_t input = nmatrix_allocator(SHAPE(2, 2, 1));
    nmatrix_t dense_1 = nmatrix_allocator(SHAPE(2, 2, 1));
    nmatrix_t output = nmatrix_allocator(SHAPE(2, 1, 1));

    layer_t *input_layer = layer_input(model_and, input);
    layer_t *dense_layer_1 = layer_dense(model_and, dense_1);
    layer_t *activation_layer_1 = layer_activation(model_and, activation_functions_sigmoid);
    layer_t *dense_layer_2 = layer_dense(model_and, output);
    layer_t *activation_layer_2 = layer_activation(model_and, activation_functions_sigmoid);
    layer_t *output_layer = layer_output(model_and, output_make_guess_round, output_functions_meansquared, output_cost_mean_squared);

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

    nmatrix_t *input_data = malloc(num_examples * sizeof(nmatrix_t));
    nmatrix_t *output_data = malloc(num_examples * sizeof(nmatrix_t));
    for (int i = 0; i < num_examples; i++) {
        input_data[i] = nmatrix_allocator(SHAPE(2, input_size, 1));
        output_data[i] = nmatrix_allocator(SHAPE(2, output_size, 1));
        nmatrix_set_values_to_fit(&input_data[i], input_size, raw_input_data[i]);
        nmatrix_set_values_to_fit(&output_data[i], output_size, raw_output_data[i]);
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

    nmatrix_free(&input);
    nmatrix_free(&dense_1);
    nmatrix_free(&output);

    return training_info;
}