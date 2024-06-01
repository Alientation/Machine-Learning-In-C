#include <app/visualizer.h>
#include <app/visutil.h>

#include <raylib.h>
#define RAYGUI_IMPLEMENTATION
#include <app/raygui.h>

#include <pthread.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

#define PROFILER_DISABLE_FUNCTION_RETURN
#include <util/profiler.h>
#include <util/debug_memory.h>

/*
GOALS

modify input node values to see what the model will output
weights lines coloring based on the strength
button to start training, selection of train/test data, graph of train/test accuracy and output loss
*/
extern training_info_t training_info;
bool is_training = false;
bool is_testing = false;
bool playground_state = false;

static bool is_window_open = false;
static const int screenWidth = 800;
static const int screenHeight = 450;
static const int modelWidth = 760;
static const int modelHeight = 300;
static const int modelX = 20;
static const int modelY = 20;

static int model_start_x = modelX;
static int model_start_y = modelY;

static const Color node_positive_color = {
    .a = 128,
    .r = 150, .g = 255, .b = 173
};
static const Color node_negative_color = {
    .a = 128,
    .r = 255, .g = 178, .b = 150
};

static const int weight_value_precision = 9;
static const int node_value_precision = 7;
static const int node_value_font_size = 10;
static const int node_radius = 25;
static const int node_gap = 20;
static const int layer_font_size = 16;
static const int layer_name_offset_y = 20;
static const int layer_gap = 60;

static const int mouse_hover_distance_threshold = 5;

static const int tooltip_border = 10;
static const int tooltip_height = 30;
static const int tooltip_width = 100;
static const int tooltip_fontsize = 16;
static const Color tooltip_fontcolor = BLACK;
static const Color tooltip_background_color = {
    .a = 255,
    .r = 227,
    .g = 215,
    .b = 252
};

static const int tooltip_buffer_size = 100;
static bool show_tooltip = false;
static char tooltip_msg[100];


void* window_run(void *vargp) {
    assert(!is_window_open);

    neural_network_model_t *model = (neural_network_model_t*) vargp;

    
    is_window_open = true;

    InitWindow(screenWidth, screenHeight, "model visualizer");
    SetTargetFPS(60);

    window_keep_open(model, 0);
}

Vector2 get_node_position(int layer_index, int r, mymatrix_t nodes) {
    int layer_height = nodes.r * (node_gap + 2 * node_radius) - node_gap;
    int layer_start_y = model_start_y + (modelHeight - layer_height) / 2;

    int x = model_start_x + layer_index * (layer_gap + 2 * node_radius) + node_radius;
    int y = layer_start_y + r * (node_gap + 2 * node_radius) + node_radius;

    return (Vector2) {.x = x, .y = y};
}

void OpenTooltip(const char* msg) {
    if (show_tooltip) {
        return;
    }

    int size = strlen(msg);
    memcpy(tooltip_msg, msg, size);
    tooltip_msg[tooltip_buffer_size-1] = '\0'; // safety null character
    show_tooltip = true;
}

// todo, in future, don't draw redundant Activation/Output layer neurons, group all activation/output layers after a dense layer together
void DrawLayerEdges(int layer_index, layer_t *layer, layer_t *prev) {
    mymatrix_t this_neurons = layer_get_neurons(layer);
    mymatrix_t prev_neurons = layer_get_neurons(prev);
    
    if (layer->type == DENSE) { // fully connected
        mymatrix_t weights = layer->layer.dense.weights;
        for (int r2 = 0; r2 < this_neurons.r; r2++) {
            // so we can color each weight based on its respective value to other weights
            // connecting to the same output neuron
            float max_weight = -1;
            for (int r1 = 0; r1 < prev_neurons.r; r1++) {
                float value = fabs(weights.matrix[r2][r1]);
                max_weight = max_weight < value ? value : max_weight;
            }
            
            for (int r1 = 0; r1 < prev_neurons.r; r1++) {
                Vector2 this_pos = get_node_position(layer_index, r2, this_neurons);
                Vector2 prev_pos = get_node_position(layer_index-1, r1, prev_neurons);

                this_pos.x -= node_radius;
                prev_pos.x += node_radius;
                float ratio = 0.5;
                if (max_weight != 0) {
                    ratio = fabs(weights.matrix[r2][r1]) / max_weight;
                }
                
                int cval = (int) (255 * ratio);
                Color color = {
                    .a = 255,
                    .r = cval,
                    .g = cval,
                    .b = cval,
                };
                DrawLineV(this_pos, prev_pos, color);
                if (CheckCollisionPointLine(GetMousePosition(), prev_pos, this_pos, mouse_hover_distance_threshold)) { 
                    // display information about the weight
                    char weight[weight_value_precision];
                    snprintf(weight, weight_value_precision, "%f", weights.matrix[r2][r1]);
                    OpenTooltip(weight);
                }
            }
        }
    } else { // Activation or Output, one to one connections
        assert(this_neurons.r == prev_neurons.r);
        for (int r = 0; r < prev_neurons.r; r++) {
            Vector2 this_pos = get_node_position(layer_index, r, this_neurons);
            Vector2 prev_pos = get_node_position(layer_index-1, r, prev_neurons);

            this_pos.x -= node_radius;
            prev_pos.x += node_radius;
            
            const int num_dots = 11;
            for (int dot = 1; dot <= num_dots; dot++) {
                if (dot % 2 == 0) {
                    continue;
                }

                Vector2 draw_end;
                draw_end.x = prev_pos.x + (int)((this_pos.x - prev_pos.x) * (dot / (float) num_dots));
                draw_end.y = prev_pos.y + (int)((this_pos.y - prev_pos.y) * (dot / (float) num_dots));
                Vector2 draw_start;
                draw_start.x = prev_pos.x + (int)((this_pos.x - prev_pos.x) * ((dot-1) / (float) num_dots));
                draw_start.y = prev_pos.y + (int)((this_pos.y - prev_pos.y) * ((dot-1) / (float) num_dots));
                DrawLineV(draw_start, draw_end, BLACK);
            }

            // todo draw tooltip info about layer?
        }
    }
}

void DrawLayer(int layer_index, layer_t *layer) {
    mymatrix_t nodes = layer_get_neurons(layer);

    // calculate values for color scaling
    float max_node_value = -1;
    float min_node_value = -1;
    for (int r = 0; r < nodes.r; r++) {
        float value = fabs(tanh(nodes.matrix[r][0]));
        if (max_node_value < 0 || min_node_value < 0) {
            max_node_value = value;
            min_node_value = value;
        } else {
            max_node_value = max_node_value < value ? value : max_node_value;
            min_node_value = min_node_value > value ? value : min_node_value;
        }
    }

    // draw each neuron
    for (int r = 0; r < nodes.r; r++) {
        // draw node with color respective to its value
        float ratio = .5;
        if (max_node_value != min_node_value) {
            ratio = (fabs(tanh(nodes.matrix[r][0])) - min_node_value) / (max_node_value - min_node_value);
        }
        Color target = nodes.matrix[r][0] < 0 ? node_negative_color : node_positive_color;        
        Color shade = {
            .a = (char) round(255 * ratio),
            .r = target.r,
            .g = target.g,
            .b = target.b
        };
        Vector2 pos = get_node_position(layer_index, r, nodes);
        DrawCircleV(pos, node_radius, WHITE);
        DrawCircleV(pos, node_radius, shade);
        DrawCircleLinesV(pos, node_radius, BLACK); // outline
        
        // edit input node values
        if (playground_state && layer->type == INPUT && CheckCollisionPointCircle(GetMousePosition(), pos, node_radius + node_gap/2)) {
            DrawRectangle(pos.x - node_radius - 6, pos.y + node_value_font_size - 4, node_radius * 2 + 12, node_value_font_size + 8, tooltip_background_color);
            DrawRectangleLines(pos.x - node_radius - 6, pos.y + node_value_font_size - 4, node_radius * 2 + 12, node_value_font_size + 8, BLACK);
            GuiSlider((Rectangle) {.x = pos.x - 3 * node_radius / 4, .y = pos.y + node_value_font_size, .width = 3 * node_radius / 2, .height = node_value_font_size},
                    "0", "1", &nodes.matrix[r][0], 0, 1);

            // todo this should most likely be ran on a separate thread, perhaps just have one thread always running
            // that detects changes to the model's inputs when not currently training or testing and recalculates the corresponding output
            mymatrix_t output = model_calculate(training_info.model);
        }

        // todo maybe in future allow user to alter value in node maybe
        // clicking it opens a small gui that has a slider for the node's value
        
        // draw value of node
        char node_value[node_value_precision];
        snprintf(node_value, node_value_precision, "%f", (float) nodes.matrix[r][0]);
        DrawCenteredText(node_value, pos.x, pos.y, node_value_font_size, BLACK);
    }

    // draw layer name and information about it
    int layer_height = nodes.r * (node_gap + 2 * node_radius) - node_gap;
    int layer_start_y = model_start_y + (modelHeight - layer_height) / 2;
    int layer_name_y = layer_start_y + layer_height + layer_name_offset_y + layer_font_size / 2;
    int layer_function_name_y = layer_name_y - layer_font_size / 2 + layer_font_size * 2;
    int layer_x = model_start_x + layer_index * (layer_gap + 2 * node_radius) + node_radius;
    int layer_info_font_size = layer_font_size - 4;
    DrawOutlinedCenteredText(get_layer_name(layer), layer_x, layer_name_y, layer_font_size, WHITE, 1, BLACK);
    if (layer->type == ACTIVATION) {
        DrawOutlinedCenteredText(get_activation_function_name(&layer->layer.activation), layer_x, 
                layer_function_name_y, layer_info_font_size, WHITE, 1, BLACK);
    } else if (layer->type == OUTPUT) {
        DrawOutlinedCenteredText(get_output_function_name(&layer->layer.output), layer_x, 
                layer_function_name_y, layer_info_font_size, WHITE, 1, BLACK);
    }
}

void DrawNeuralNetwork(neural_network_model_t *model) {
    // calculate center x for the model
    int model_width = model->num_layers * (layer_gap + 2 * node_radius) - layer_gap;
    model_start_x = modelX + (modelWidth - model_width) / 2;
    model_start_y = modelY;

    // draw model background
    Color color = DARKBLUE;
    color.a = 170;
    DrawRectangle(modelX, modelY, modelWidth, modelHeight, color);

    // draw each layer
    layer_t *cur = model->input_layer;
    for (int layer_i = 0; layer_i < model->num_layers; layer_i++) {
        DrawLayer(layer_i, cur);
        if (layer_i != 0) {
            DrawLayerEdges(layer_i, cur, cur->prev);
        }
        cur = cur->next;
    }
}

/**
 * Ran as a separate thread
 */
void* train_run(void *vargp) {
    if (!is_training && !is_testing) {
        is_training = true;
        CLOCK_MARK
        model_train_info((training_info_t*) vargp);
        CLOCK_MARK_ENTRY("TRAINING")
        is_training = false;
    }
}

/**
 * Ran as a separate thread
 */
void* test_run(void *vargp) {
    if (!is_testing && !is_training) {
        is_testing = true;
        CLOCK_MARK
        // model_test_info((training_info_t*) vargp); // TODO
        CLOCK_MARK_ENTRY("TESTING")
        is_testing = false;
    }
}

void DrawWindow(neural_network_model_t *model) {
    BeginDrawing();
    {
        ClearBackground(RAYWHITE);
        DrawText(TextFormat("%d FPS", GetFPS()), 5, 5, 10, BLACK);

        show_tooltip = false;        
        // maybe have a better way to signify if a model is built or not
        if (model && model->input_layer != NULL) {
            DrawNeuralNetwork(model);
        }

        // draw model's training_info
        // TODO

        // some model control buttons
        if (GuiButton((Rectangle) {.x = 50, .y = 50, .height = 20, .width = 110}, "Start Training")) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, train_run, &training_info);
            pthread_detach(thread_id);
        }

        if (GuiButton((Rectangle) {.x = 170, .y = 50, .height = 20, .width = 80}, "Start Test")) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, test_run, &training_info);
            pthread_detach(thread_id);
        }

        GuiToggle((Rectangle) {.x = 260, .y = 50, .height = 20, .width = 80}, "Playground", &playground_state);
        if (is_testing || is_training) { // dont mess with training
            playground_state = false;
        }

        // draw tooltip
        if (show_tooltip) {
            Vector2 mouse_pos = GetMousePosition();
            int inner_width = tooltip_width - tooltip_border * 2;
            DrawRectangleLines(mouse_pos.x - 1, mouse_pos.y - tooltip_height - 1, tooltip_width + 2, tooltip_height+2, BLACK);
            DrawRectangle(mouse_pos.x, mouse_pos.y - tooltip_height, tooltip_width, tooltip_height, tooltip_background_color);
            DrawCenteredText(tooltip_msg, mouse_pos.x + tooltip_width / 2, mouse_pos.y - tooltip_height / 2, tooltip_fontsize, tooltip_fontcolor);
        }
    }
    EndDrawing();
}


void window_keep_open(neural_network_model_t *model, unsigned int num_seconds) {
    if (num_seconds == 0) {
        num_seconds = ~0;
    }

    time_t now = clock();
    unsigned long long num_ms = num_seconds * 1000L;
    while (!WindowShouldClose() && clock() - now < num_ms) {
        DrawWindow(model);
    }
    is_window_open = false;
}

