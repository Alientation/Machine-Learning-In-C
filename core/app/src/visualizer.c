#include <app/visualizer.h>
#include <app/visutil.h>
#include <app/drawpanel.h>

#define RAYGUI_IMPLEMENTATION
#include <app/raygui.h>

#include <pthread.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

#define PROFILER_DISABLE_FUNCTION_RETURN
#include <util/profiler.h>
#include <util/debug_memory.h>

/*
GOALS

selection of train/test data, graph of train/test accuracy and output loss

MAYBE BUT NOT NECESSARY
split rendering from logic (like the draw panel) so we can run each on its own thread and mouse inputs are registered faster
*/

static visualizer_state_t vis_state = {
    .is_training = false,
    .is_testing = false,
    .playground_state = false,

    .show_tooltip = false,
    .tooltip_priority = 0,
    .tooltip_weight_value = NULL,

    .is_window_open = false,
}; 

static const Color NODE_POS_COLOR = {
    .a = 128,
    .r = 150, .g = 255, .b = 173
};
static const Color NODE_NEG_COLOR = {
    .a = 128,
    .r = 255, .g = 178, .b = 150
};

static const Color TOOLTIP_FONTCOLOR = BLACK;
static const Color TOOLTIP_BACKGROUND_COLOR = {
    .a = 255,
    .r = 227, .g = 215, .b = 252
};


//===========================================================================

void* window_run(void *vargp) {    
    assert(!vis_state.is_window_open);
    SetTraceLogLevel(LOG_ERROR); 

    visualizer_argument_t *args = (visualizer_argument_t*) vargp;
    vis_state.vis_args = *args;
    vis_state.is_window_open = true;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, TextFormat("%s Visualizer", args->model_name));
    SetTargetFPS(60);    

    vis_state.draw_args = (drawing_panel_args_t) {
        .is_open = false,
        .brush_size = 10,
        .brush_color = ColorToHSV(BLACK),
        
        .prev_draw_pos = (Vector2) {.x = -1, .y = -1},
        .is_dragged = false,
        .is_drawing = false,
        .draw_texture = LoadRenderTexture(400, 400),
        
        .segments_list_head = NULL,
        .segments_list_cur = NULL,
        .segments_list_size = 0,

        .updated = false,
        .update_frames = 3,
        .cur_frames = 0,
        .input_texture = LoadRenderTexture(28, 28),
        .gray_scale = true,
        .buffer_width = 28,
        .buffer_height = 28,
        .output_buffer = malloc(sizeof(float) * 28 * 28)
    };
    SetTextureFilter(vis_state.draw_args.draw_texture.texture, TEXTURE_FILTER_TRILINEAR);

    BeginTextureMode(vis_state.draw_args.draw_texture);
    {
        ClearBackground(WHITE);
    }
    EndTextureMode();
    DrawingPanelAdd(&vis_state.draw_args);

    window_keep_open(args->model, 0);

    DrawingPanelFreeHistory(&vis_state.draw_args);
    UnloadRenderTexture(vis_state.draw_args.draw_texture);
    free(vis_state.draw_args.output_buffer);
}

void* train_run(void *vargp) {
    if (!vis_state.is_training && !vis_state.is_testing) {
        vis_state.is_training = true;
        CLOCK_MARK
        model_train_info((training_info_t*) vargp);
        CLOCK_MARK_ENTRY("TRAINING")
        vis_state.is_training = false;
    }
}

void* test_run(void *vargp) {
    if (!vis_state.is_testing && !vis_state.is_training) {
        vis_state.is_testing = true;
        CLOCK_MARK
        // model_test_info((training_info_t*) vargp); // TODO
        CLOCK_MARK_ENTRY("TESTING")
        vis_state.is_testing = false;
    }
}


//===================================================================


Vector2 get_node_position(int layer_index, int r, mymatrix_t nodes) {
    int layer_height = nodes.r * (NODE_GAP + 2 * NODE_RADIUS) - NODE_GAP;
    int model_width = vis_state.vis_args.model->num_layers * (LAYER_GAP + 2 * NODE_RADIUS) - LAYER_GAP;
    int layer_start_y = MODEL_Y + (MODEL_HEIGHT - layer_height) / 2;

    int x = (MODEL_WIDTH - model_width) / 2 + layer_index * (LAYER_GAP + 2 * NODE_RADIUS) + NODE_RADIUS;
    int y = layer_start_y + r * (NODE_GAP + 2 * NODE_RADIUS) + NODE_RADIUS;
    
    if (nodes.r > MAX_LAYER_NODES) {
        // used for drawing weights since those can be squished together
        y = (((float) r) / nodes.r) * HIDDEN_LAYER_HEIGHT;
    }
    return (Vector2) {.x = x, .y = y};
}

// displays the highest priority tooltip
void OpenTooltip(const char* msg, float priority, float *weight_value) {
    if (vis_state.show_tooltip && priority <= vis_state.tooltip_priority) {
        return;
    }

    int size = strlen(msg);
    memcpy(vis_state.tooltip_msg, msg, size);
    vis_state.tooltip_msg[TOOLTIP_BUFFER_SIZE-1] = '\0'; // safety null character
    vis_state.show_tooltip = true;
    vis_state.tooltip_priority = priority;
    vis_state.tooltip_weight_value = weight_value;
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

                this_pos.x -= NODE_RADIUS;
                prev_pos.x += NODE_RADIUS;
                float ratio = 0.5;
                if (max_weight != 0) {
                    ratio = 1 - fabs(weights.matrix[r2][r1]) / max_weight;
                }
                
                // todo future, draw lines with thickness relative to its weight instead of color
                // it is kind of hard to distinguish between colors especially when weight lines are extremely thin
                int cval = (int) (255 * ratio);
                Color color = {
                    .a = 255,
                    .r = cval,
                    .g = cval,
                    .b = cval,
                };
                DrawLineEx(this_pos, prev_pos, 3, BLACK);
                DrawLineV(this_pos, prev_pos, color);
                if (CheckCollisionPointLine(GetMousePosition(), prev_pos, this_pos, MOUSE_HOVER_DISTANCE_TO_WEIGHT)) { 
                    // display information about the weight
                    char weight[WEIGHT_DISPLAY_PRECISION];
                    snprintf(weight, WEIGHT_DISPLAY_PRECISION, "%f", weights.matrix[r2][r1]);
                    OpenTooltip(weight, sqrt(pow(prev_pos.x - this_pos.x, 2) + pow(prev_pos.y - this_pos.y, 2)), &weights.matrix[r2][r1]);
                }
            }
        }
    } else { // Activation or Output, one to one connections
        assert(this_neurons.r == prev_neurons.r);
        for (int r = 0; r < prev_neurons.r; r++) {
            Vector2 this_pos = get_node_position(layer_index, r, this_neurons);
            Vector2 prev_pos = get_node_position(layer_index-1, r, prev_neurons);

            this_pos.x -= NODE_RADIUS;
            prev_pos.x += NODE_RADIUS;
            
            for (int dot = 1; dot <= WEIGHT_DOTTED_LINES; dot++) {
                if (dot % 2 == 0) {
                    continue;
                }

                Vector2 draw_end;
                draw_end.x = prev_pos.x + (int)((this_pos.x - prev_pos.x) * (dot / (float) WEIGHT_DOTTED_LINES));
                draw_end.y = prev_pos.y + (int)((this_pos.y - prev_pos.y) * (dot / (float) WEIGHT_DOTTED_LINES));
                Vector2 draw_start;
                draw_start.x = prev_pos.x + (int)((this_pos.x - prev_pos.x) * ((dot-1) / (float) WEIGHT_DOTTED_LINES));
                draw_start.y = prev_pos.y + (int)((this_pos.y - prev_pos.y) * ((dot-1) / (float) WEIGHT_DOTTED_LINES));
                DrawLineV(draw_start, draw_end, BLACK);
            }

            // todo draw tooltip info about layer?
        }
    }
}

void DrawLayerInformation(int layer_index, layer_t *layer) {
    mymatrix_t nodes = layer_get_neurons(layer);

    // draw layer name and information about it
    // number of nodes * node diameter + node_gap between each node (subtract one because fence post problem)
    int layer_height = nodes.r * (NODE_GAP + 2 * NODE_RADIUS) - NODE_GAP;
    // offset the layer so that it is in the center vertically
    int layer_start_y = MODEL_Y + (MODEL_HEIGHT - layer_height) / 2;
    int layer_name_y = layer_start_y + layer_height + LAYER_NAME_OFFSET_Y + LAYER_DISPLAY_FONTSIZE / 2;
    int layer_function_name_y = layer_name_y + (3 * LAYER_DISPLAY_FONTSIZE) / 2;

    int model_width = vis_state.vis_args.model->num_layers * (LAYER_GAP + 2 * NODE_RADIUS) - LAYER_GAP;
    int layer_x = (MODEL_WIDTH - model_width) / 2 + layer_index * (LAYER_GAP + 2 * NODE_RADIUS) + NODE_RADIUS;
    int layer_info_font_size = LAYER_DISPLAY_FONTSIZE - 4;
    DrawOutlinedCenteredText(get_layer_name(layer), layer_x, layer_name_y, LAYER_DISPLAY_FONTSIZE, BLACK, 0, BLACK);
    if (layer->type == ACTIVATION) {
        DrawOutlinedCenteredText(get_activation_function_name(&layer->layer.activation), layer_x, 
                layer_function_name_y, layer_info_font_size, BLACK, 0, BLACK);
    } else if (layer->type == OUTPUT) {
        DrawOutlinedCenteredText(get_output_function_name(&layer->layer.output), layer_x, 
                layer_function_name_y, layer_info_font_size, BLACK, 0, BLACK);
        DrawOutlinedCenteredText(get_output_guess_function_name(&layer->layer.output), layer_x, 
                layer_function_name_y + (3 * LAYER_DISPLAY_FONTSIZE) / 2, layer_info_font_size, BLACK, 0, BLACK);
    }
}

void DrawLayer(int layer_index, layer_t *layer) {
    mymatrix_t nodes = layer_get_neurons(layer);

    if (nodes.r > MAX_LAYER_NODES) {
        DrawRectangleLines(get_node_position(layer_index, 0, nodes).x - NODE_RADIUS, get_node_position(layer_index, 0, nodes).y, 
                HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_HEIGHT, BLACK);

        DrawLayerInformation(layer_index, layer);
        return;
    }

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
        Color target = nodes.matrix[r][0] < 0 ? NODE_NEG_COLOR : NODE_POS_COLOR;        
        Color shade = {
            .a = (char) round(255 * ratio),
            .r = target.r,
            .g = target.g,
            .b = target.b
        };
        Vector2 pos = get_node_position(layer_index, r, nodes);
        DrawCircleV(pos, NODE_RADIUS, WHITE);
        DrawCircleV(pos, NODE_RADIUS, shade);
        DrawCircleLinesV(pos, NODE_RADIUS, BLACK); // outline
        
        // edit input node values
        if (vis_state.playground_state && layer->type == INPUT && CheckCollisionPointCircle(GetMousePosition(), pos, MOUSE_HOVER_DISTANCE_TO_NODE)
                && !vis_state.draw_args.is_open) {
            // check if too small
            if (NODE_RADIUS < MIN_NODE_RADIUS_FOR_SLIDER_BAR) {
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    // show node's details in a pop up window with a slider
                }
            } else {
                int rec_x = pos.x - NODE_RADIUS - 6;
                int rec_y = pos.y + NODE_DISPLAY_FONTSIZE - 4;
                int rec_width = NODE_RADIUS * 2 + 12;
                int rec_height = NODE_DISPLAY_FONTSIZE + 8;

                DrawRectangle(rec_x, rec_y, rec_width, rec_height, TOOLTIP_BACKGROUND_COLOR);
                DrawRectangleLines(rec_x, rec_y, rec_width, rec_height, BLACK);
                GuiSlider((Rectangle) {.x = pos.x - 3 * NODE_RADIUS / 4, .y = pos.y + NODE_DISPLAY_FONTSIZE, .width = 3 * NODE_RADIUS / 2, .height = NODE_DISPLAY_FONTSIZE},
                        "0", "1", &nodes.matrix[r][0], 0, 1);

                // todo this should most likely be ran on a separate thread, perhaps just have one thread always running
                // that detects changes to the model's inputs when not currently training or testing and recalculates the corresponding output
                mymatrix_t output = model_calculate(vis_state.vis_args.training_info.model);
            }
        }

        // todo maybe in future allow user to alter value in node maybe
        // clicking it opens a small gui that has a slider for the node's value
        
        // draw value of node
        char node_value[NODE_DISPLAY_PRECISION];
        snprintf(node_value, NODE_DISPLAY_PRECISION, "%f", (float) nodes.matrix[r][0]);
        DrawCenteredText(node_value, pos.x, pos.y, NODE_DISPLAY_FONTSIZE, BLACK);
    }

    DrawLayerInformation(layer_index, layer);
}

void DrawNeuralNetwork(neural_network_model_t *model) {
    // draw model background
    Color color = GRAY;
    color.a = 170;
    DrawRectangle(MODEL_X, MODEL_Y, MODEL_WIDTH, MODEL_HEIGHT, color);
    DrawCenteredText(vis_state.vis_args.model_name, MODEL_WIDTH/2, 35, 20, BLACK);

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

void DrawWindow(neural_network_model_t *model) {
    BeginDrawing();
    {
        ClearBackground(RAYWHITE);
        DrawText(TextFormat("%d FPS", GetFPS()), 5, 5, 10, BLACK);

        vis_state.show_tooltip = false;        
        // maybe have a better way to signify if a model is built or not
        if (model && model->input_layer != NULL) {
            DrawNeuralNetwork(model);
        }

        // draw model's training_info
        // TODO

        // some model control buttons
        if (GuiButton((Rectangle) {.x = 50, .y = 60, .height = 30, .width = 130}, "Start Training") && !vis_state.draw_args.is_open) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, train_run, &vis_state.vis_args.training_info);
            pthread_detach(thread_id);
        }

        if (GuiButton((Rectangle) {.x = 190, .y = 60, .height = 30, .width = 100}, "Start Test") && !vis_state.draw_args.is_open) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, test_run, &vis_state.vis_args.training_info);
            pthread_detach(thread_id);
        }

        GuiToggle((Rectangle) {.x = 300, .y = 60, .height = 30, .width = 100}, "Playground", &vis_state.playground_state);
        if (vis_state.is_testing || vis_state.is_training) { // dont mess with training
            vis_state.playground_state = false;
        }

        // TODO dont add drawing panel if model does not accept drawings as input
        if (GuiButton((Rectangle) {.x = 410, .y = 60, .height = 30, .width = 120}, "Drawing Panel") && !vis_state.draw_args.is_open) {
            vis_state.draw_args.is_open = true;
        }
        GuiDrawingPanelPopup(&vis_state.draw_args);

        // draw tooltip
        if (vis_state.show_tooltip && !vis_state.draw_args.is_open) {
            Vector2 mouse_pos = GetMousePosition();
            int rec_x = mouse_pos.x;
            int rec_y = mouse_pos.y - TOOLTIP_HEIGHT;
            DrawRectangleLines(rec_x, rec_y, TOOLTIP_WIDTH, TOOLTIP_HEIGHT, BLACK);
            DrawRectangle(rec_x, rec_y, TOOLTIP_WIDTH, TOOLTIP_HEIGHT, TOOLTIP_BACKGROUND_COLOR);
            DrawCenteredText(vis_state.tooltip_msg, mouse_pos.x + TOOLTIP_WIDTH / 2, mouse_pos.y - TOOLTIP_HEIGHT / 2, TOOLTIP_FONTSIZE, TOOLTIP_FONTCOLOR);

            if (vis_state.tooltip_weight_value && vis_state.playground_state && !vis_state.is_testing && !vis_state.is_training) { // todo replace with model.isLocked instead
                *vis_state.tooltip_weight_value += TOOLTIP_WEIGHT_VALUE_SCALE * GetMouseWheelMove();
                mymatrix_t output = model_calculate(vis_state.vis_args.training_info.model); // todo preferrably run on separate thread
            }
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
    vis_state.is_window_open = false;
}

