#include <app/visualizer.h>

#include <raylib.h>
#include <time.h>
#include <stdio.h>

static bool is_window_open = false;
static const int screenWidth = 800;
static const int screenHeight = 450;
static const int modelWidth = 760;
static const int modelHeight = 300;
static const int modelX = 20;
static const int modelY = 20;

static int model_start_x = modelX;
static int model_start_y = modelY;

static const int node_value_precision = 6;
static const int node_value_font_size = 10;
static const int node_radius = 25;
static const int node_gap = 20;
static const int layer_font_size = 14;
static const int layer_name_offset_y = 20;
static const int layer_gap = 60;

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

void draw_edges(int layer_index, layer_t *layer, layer_t *prev) {

}

void draw_layer(int layer_index, layer_t *layer) {
    mymatrix_t nodes = layer_get_neurons(layer);

    int layer_height = nodes.r * (node_gap + 2 * node_radius) - node_gap;
    // center
    int layer_start_y = model_start_y + (modelHeight - layer_height) / 2;
    int x = model_start_x + layer_index * (layer_gap + 2 * node_radius) + node_radius;
    for (int i = 0; i < nodes.r; i++) {
        int y = layer_start_y + i * (node_gap + 2 * node_radius) + node_radius;

        // draw node outline
        DrawCircleLines(x, y, node_radius, BLACK);
        
        // draw value of node
        char node_value[node_value_precision];
        snprintf(node_value, node_value_precision, "%f", (float) nodes.matrix[i][0]);
        draw_centered_text(node_value, x, y, node_value_font_size, BLACK);
    }

    // draw layer name
    draw_centered_text(get_layer_name(layer), x, layer_start_y + layer_height + layer_name_offset_y + layer_font_size / 2, layer_font_size, BLACK);
}

void draw_model(neural_network_model_t *model) {
    // calculate center x for the model
    int model_width = model->num_layers * (layer_gap + 2 * node_radius) - layer_gap;
    model_start_x = modelX + (modelWidth - model_width) / 2;
    model_start_y = modelY;

    // draw model background
    DrawRectangle(modelX, modelY, modelWidth, modelHeight, LIGHTGRAY);

    layer_t* cur = model->input_layer;
    for (int layer_i = 0; layer_i < model->num_layers; layer_i++) {
        draw_layer(layer_i, cur);
        if (layer_i != 0) {
            draw_edges(layer_i, cur, cur->prev);
        }
        cur = cur->next;
    }
}

void window_draw(neural_network_model_t *model) {
    BeginDrawing();
    {
        ClearBackground(RAYWHITE);

        // maybe have a better way to signify if a model is built or not
        if (model && model->input_layer != NULL) {
            draw_model(model);
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

