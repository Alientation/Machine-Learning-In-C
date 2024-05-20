#include <model/model.h>

#include <assert.h>
#include <memory.h>
#include <stdlib.h>

static output_layer_t* layer_run_activation(activation_layer_t *activation_layer, input_layer_t *input,
                                            output_layer_t *output) {
    activation_layer->activation(input, output);
}

static output_layer_t* layer_run_neuron(neuron_layer_t *neuron_layer, input_layer_t *input,
                                        output_layer_t *output) {
    // W.X + B

}

void layer_run(layer_t *layer, input_layer_t *input, output_layer_t *output) {
    switch (layer->type) {
        case LayerType_ACTIVATION:
            layer_run_activation(&layer->layer.activation, input, output);
            break;
        case LayerType_NEURON:
            layer_run_neuron(&layer->layer.neuron, input, output);
            break;
        case LayerType_INPUT:
        case LayerType_OUTPUT:
        default:
            assert(0);
    }
}


void model_run(model_t *model, input_layer_t *input,
                            output_layer_t *output) {
    assert(model->layers[0].type == LayerType_INPUT);
    assert(model->layers[model->num_layers-1].type == LayerType_OUTPUT);

    output_layer_t calc_layer;
    memcpy(&calc_layer, input, sizeof(*input));

    for (int layer_i = 1; layer_i < model->num_layers; layer_i++) {
        layer_run(&model->layers[layer_i], &calc_layer, &calc_layer);
    }

    memcpy(output, &calc_layer, sizeof(calc_layer));
}

void model_train(model_t *model, int num_data, input_layer_t **inputs, output_layer_t **outputs) {
    assert(model->layers[0].type == LayerType_INPUT);
    assert(model->layers[model->num_layers-1].type == LayerType_OUTPUT);
}

void model_test(model_t *model, int num_data, input_layer_t **inputs, output_layer_t **outputs) {
    assert(model->layers[0].type == LayerType_INPUT);
    assert(model->layers[model->num_layers-1].type == LayerType_OUTPUT);
}