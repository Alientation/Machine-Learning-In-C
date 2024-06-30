#include <model/model.h>
#include <util/math.h>

#include <math.h>

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

    nmatrix_add(&this->layer.dense.d_cost_wrt_weight_sum, &this->layer.dense.d_cost_wrt_weight, &this->layer.dense.d_cost_wrt_weight_sum);
    nmatrix_add(&this->layer.dense.d_cost_wrt_bias_sum, &this->layer.dense.d_cost_wrt_bias, &this->layer.dense.d_cost_wrt_bias_sum);

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
            nmatrix_free(&layer->layer.dense.d_cost_wrt_weight_sum);
            nmatrix_free(&layer->layer.dense.d_cost_wrt_bias_sum);
            break;
        case DROPOUT:
            nmatrix_free(&layer->layer.dropout.output);
            nmatrix_free(&layer->layer.dropout.d_cost_wrt_input);
            break;
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