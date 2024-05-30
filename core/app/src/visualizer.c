#include <app/visualizer.h>

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
static const int layer_font_size = 14;
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

void draw_centered_text(const char* text, int centerx, int centery, int fontsize, Color fontcolor) {
    int width = MeasureText(text, fontsize);
    DrawText(text, centerx - width / 2, centery - fontsize/2, fontsize, fontcolor);
}

Vector2 get_node_position(int layer_index, int r, mymatrix_t nodes) {
    int layer_height = nodes.r * (node_gap + 2 * node_radius) - node_gap;
    int layer_start_y = model_start_y + (modelHeight - layer_height) / 2;

    int x = model_start_x + layer_index * (layer_gap + 2 * node_radius) + node_radius;
    int y = layer_start_y + r * (node_gap + 2 * node_radius) + node_radius;

    Vector2 vec = {
        .x = x,
        .y = y,
    };
    return vec;
}

void open_tooltip(const char* msg) {
    if (show_tooltip) {
        return;
    }

    int size = strlen(msg);
    memcpy(tooltip_msg, msg, size);
    tooltip_msg[tooltip_buffer_size-1] = '\0'; // safety null character
    show_tooltip = true;
}

void draw_edges(int layer_index, layer_t *layer, layer_t *prev) {
    mymatrix_t this_neurons = layer_get_neurons(layer);
    mymatrix_t prev_neurons = layer_get_neurons(prev);
    if (layer->type == DENSE) { // fully connected
        mymatrix_t weights = layer->layer.dense.weights;
        for (int r1 = 0; r1 < prev_neurons.r; r1++) {
            for (int r2 = 0; r2 < this_neurons.r; r2++) {
                Vector2 this_pos = get_node_position(layer_index, r2, this_neurons);
                Vector2 prev_pos = get_node_position(layer_index-1, r1, prev_neurons);

                this_pos.x -= node_radius;
                prev_pos.x += node_radius;

                DrawLineV(this_pos, prev_pos, BLACK);
                if (CheckCollisionPointLine(GetMousePosition(), prev_pos, this_pos, mouse_hover_distance_threshold)) { 
                    // display information about the weight
                    char weight[weight_value_precision];
                    snprintf(weight, weight_value_precision, "%f", weights.matrix[r2][r1]);
                    open_tooltip(weight);
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

            DrawLineV(this_pos, prev_pos, BLACK);

            // todo draw tooltip info like (activation function type or output guess function)
        }
    }
}

void draw_layer(int layer_index, layer_t *layer) {
    mymatrix_t nodes = layer_get_neurons(layer);

    double max_node_value = -1;
    double min_node_value = -1;
    for (int r = 0; r < nodes.r; r++) {
        double value = fabs(tanh(nodes.matrix[r][0]));
        if (max_node_value < 0 || min_node_value < 0) {
            max_node_value = value;
            min_node_value = value;
        } else {
            max_node_value = max_node_value < value ? value : max_node_value;
            min_node_value = min_node_value > value ? value : min_node_value;
        }
    }

    // center
    for (int i = 0; i < nodes.r; i++) {
        // draw node outline
        double ratio = .5;
        if (max_node_value != min_node_value) {
            ratio = (fabs(tanh(nodes.matrix[i][0])) - min_node_value) / (max_node_value - min_node_value);
        }
        Color target = nodes.matrix[i][0] < 0 ? node_negative_color : node_positive_color;        
        Color shade = {
            .a = (char) round(255 * ratio),
            .r = target.r,
            .g = target.g,
            .b = target.b
        };
        Vector2 pos = get_node_position(layer_index, i, nodes);
        DrawCircleV(pos, node_radius, WHITE);
        DrawCircleV(pos, node_radius, shade);
        DrawCircleLinesV(pos, node_radius, BLACK);
        
        // draw value of node
        char node_value[node_value_precision];
        snprintf(node_value, node_value_precision, "%f", (float) nodes.matrix[i][0]);
        draw_centered_text(node_value, pos.x, pos.y, node_value_font_size, BLACK);
    }

    // draw layer name
    int layer_height = nodes.r * (node_gap + 2 * node_radius) - node_gap;
    int layer_start_y = model_start_y + (modelHeight - layer_height) / 2;
    int layer_x = model_start_x + layer_index * (layer_gap + 2 * node_radius) + node_radius;

    
    draw_centered_text(get_layer_name(layer), layer_x, layer_start_y + layer_height + layer_name_offset_y + layer_font_size / 2, layer_font_size, BLACK);
    if (layer->type == ACTIVATION) {
        draw_centered_text(get_activation_function_name(&layer->layer.activation), layer_x, layer_start_y + layer_height + layer_name_offset_y + 2 * layer_font_size, layer_font_size-2, BLACK);
    } else if (layer->type == OUTPUT) {
        draw_centered_text(get_output_function_name(&layer->layer.output), layer_x, layer_start_y + layer_height + layer_name_offset_y + 2 * layer_font_size, layer_font_size-2, BLACK);
    }
}

void draw_model(neural_network_model_t *model) {
    // calculate center x for the model
    int model_width = model->num_layers * (layer_gap + 2 * node_radius) - layer_gap;
    model_start_x = modelX + (modelWidth - model_width) / 2;
    model_start_y = modelY;

    // draw model background
    DrawRectangle(modelX, modelY, modelWidth, modelHeight, LIGHTGRAY);

    layer_t *cur = model->input_layer;
    for (int layer_i = 0; layer_i < model->num_layers; layer_i++) {
        draw_layer(layer_i, cur);
        if (layer_i != 0) {
            draw_edges(layer_i, cur, cur->prev);
        }
        cur = cur->next;
    }
}

void* train_run(void *vargp) {
    if (!is_training) {
        is_training = true;
        CLOCK_MARK
        model_train_info((training_info_t*) vargp);
        CLOCK_MARK_ENTRY("TRAINING")
    }
}

void window_draw(neural_network_model_t *model) {
    BeginDrawing();
    {
        ClearBackground(RAYWHITE);

        show_tooltip = false;        
        // maybe have a better way to signify if a model is built or not
        if (model && model->input_layer != NULL) {
            draw_model(model);
        }
        
        DrawText(TextFormat("%d FPS", GetFPS()), 5, 5, 10, BLACK);

        if (GuiButton((Rectangle) {.x = 50, .y = 50, .height = 20, .width = 140}, "Start Training")) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, train_run, &training_info);
        }

        // draw tooltip
        if (show_tooltip) {
            Vector2 mouse_pos = GetMousePosition();
            int inner_width = tooltip_width - tooltip_border * 2;
            DrawRectangleLines(mouse_pos.x - 1, mouse_pos.y - tooltip_height - 1, tooltip_width + 2, tooltip_height+2, BLACK);
            DrawRectangle(mouse_pos.x, mouse_pos.y - tooltip_height, tooltip_width, tooltip_height, tooltip_background_color);
            draw_centered_text(tooltip_msg, mouse_pos.x + tooltip_width / 2, mouse_pos.y - tooltip_height / 2, tooltip_fontsize, tooltip_fontcolor);
        }
    }
    EndDrawing();
}

void window_close() {
    CloseWindow();
}

void window_keep_open(neural_network_model_t *model, unsigned int num_seconds) {
    if (num_seconds == 0) {
        num_seconds = ~0;
    }

    time_t now = clock();
    unsigned long long num_ms = num_seconds * 1000L;
    while (!WindowShouldClose() && clock() - now < num_ms) {
        window_draw(model);
    }
    is_window_open = false;
}

