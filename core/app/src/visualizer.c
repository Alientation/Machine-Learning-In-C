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
#define TIME
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

    .node_positions = NULL,

    .show_tooltip = false,
    .tooltip_priority = 0,
    .tooltip_weight_value = NULL,

    .is_window_open = false,
};

//===========================================================================
static int compute_network_width(void) {
    layer_t *cur_layer = vis_state.vis_args.model->input_layer;
    int network_width = LAYER_GAP * (vis_state.vis_args.model->num_layers - 1);
    for (int i = 0; i < vis_state.vis_args.model->num_layers; i++) {
        nmatrix_t neurons = layer_get_neurons(cur_layer);
        int num_nodes = neurons.n_elements;
        vis_state.node_positions[i] = malloc(num_nodes * sizeof(Vector2));

        int nodes_per_section = HIDDEN_LAYER_NODES_HEIGHT;
        if (cur_layer->type == INPUT) {
            nodes_per_section = INPUT_LAYER_NODES_HEIGHT;
        }

        int num_sections = (num_nodes + nodes_per_section - 1) / nodes_per_section;
        network_width += (NODE_RADIUS * 2) * num_sections + NODE_GAP * (num_sections - 1);
        cur_layer = cur_layer->next;
    }
    return network_width;
}

static void construct_node_positions(void) {
    // TODO in future move all information about layer drawing to separate file
    vis_state.node_positions = malloc(vis_state.vis_args.model->num_layers * sizeof(Vector2*));

    // compute the width of the whole neural network to center it in the screen
    layer_t *cur_layer = vis_state.vis_args.model->input_layer;
    int network_width = compute_network_width();

    // center
    int network_x = MODEL_X + MODEL_WIDTH/2 - network_width/2;

    int layer_x = network_x;
    cur_layer = vis_state.vis_args.model->input_layer;
    for (int i = 0 ; i < vis_state.vis_args.model->num_layers; i++) {
        nmatrix_t neurons = layer_get_neurons(cur_layer);
        int num_nodes = neurons.n_elements;
        int nodes_per_section = cur_layer->type == INPUT ? INPUT_LAYER_NODES_HEIGHT : HIDDEN_LAYER_NODES_HEIGHT;
        if (nodes_per_section > num_nodes) {
            nodes_per_section = num_nodes;
        }

        // round up how many sections are needed with nodes_per_section nodes vertically
        int num_sections = (num_nodes + nodes_per_section - 1) / nodes_per_section;

        // center
        int layer_height = (NODE_RADIUS * 2) * nodes_per_section + (NODE_GAP) * (nodes_per_section - 1);
        int layer_y = MODEL_Y + MODEL_HEIGHT/2 - layer_height/2;
        for (int sec = 0; sec < num_sections; sec++) {
            for (int node = 0; node < nodes_per_section; node++) {
                // todo comment
                int node_index = sec * nodes_per_section + node;
                vis_state.node_positions[i][node_index].x = layer_x + sec * (NODE_RADIUS*2 + NODE_GAP) + NODE_RADIUS;
                vis_state.node_positions[i][node_index].y = layer_y + node * (NODE_RADIUS*2 + NODE_GAP) + NODE_RADIUS;
            }
        }

        // width of layer + gap between layers
        layer_x += (NODE_RADIUS * 2) * num_sections + NODE_GAP * (num_sections - 1);
        layer_x += LAYER_GAP;

        cur_layer = cur_layer->next;
    }
}

static void initialize_visualizer(visualizer_argument_t *vis_args) {
    vis_state.vis_args = *vis_args;
    vis_state.is_window_open = true;
    vis_state.draw_args = (drawing_panel_args_t) {
        .vis_args = vis_args,

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

        .is_save_popup_open = false,
        .is_dataset_viewer_open = false,

        .dataset_directory = vis_args->default_dataset_directory,
        .sel_dataset_index = -1,
        .sel_label_index = -1,
        .current_dataset = {0},
        .img_dataset_vis = {0},
        .dataset_list_scroll_index = 0,
        .sel_dataset_image_index = -1,

        .num_transformations = 0,
        .is_transformations_active = false,
        .max_rotation_degree = 0,
        .is_rotations_active = false,
        .max_translations_pixels_x = 0,
        .is_translations_x_active = false,
        .max_translations_pixels_y = 0,
        .is_translations_y_active = false,
        .max_artifacts = 0,
        .is_artifacts_active = false,
        .train_test_split = 0.8,
        .is_train_test_split_active = false,

        .add_dataset_file_name = malloc((FILE_NAME_BUFFER_SIZE + 1) * sizeof(char)),
        .add_dataset_type = 0,

        .images_dataset_width_option_active = false,
        .images_dataset_width_input = malloc((NUMBER_INPUT_BUFFER_SIZE + 1) * sizeof(char)),
        .images_dataset_height_option_active = false,
        .images_dataset_height_input = malloc((NUMBER_INPUT_BUFFER_SIZE + 1) * sizeof(char)),
        .is_editing_dataset_file_name = false,

        .num_labels = vis_args->num_labels,
        .label_names = vis_args->output_labels,

        .updated = false,
        .update_frames = 3,
        .cur_frames = 0,
        .input_texture = LoadRenderTexture(28, 28),
        .gray_scale = true,
        .buffer_width = 28,
        .buffer_height = 28,
        .output_buffer = malloc(sizeof(float) * 28 * 28),
    };

    // text box input buffers
    memset(vis_state.draw_args.add_dataset_file_name, 0, FILE_NAME_BUFFER_SIZE * sizeof(char));
    memset(vis_state.draw_args.images_dataset_width_input, 0, NUMBER_INPUT_BUFFER_SIZE * sizeof(char));
    vis_state.draw_args.images_dataset_width_input[0] = '2';
    vis_state.draw_args.images_dataset_width_input[1] = '8';
    memset(vis_state.draw_args.images_dataset_height_input, 0, NUMBER_INPUT_BUFFER_SIZE * sizeof(char));
    vis_state.draw_args.images_dataset_height_input[0] = '2';
    vis_state.draw_args.images_dataset_height_input[1] = '8';
}

static void end_visualizer(void) {
    // save the currently open dataset
    if (vis_state.draw_args.sel_dataset_index != -1) {
        WriteDataSet(vis_state.draw_args.current_dataset);

        if (vis_state.draw_args.current_dataset.type == DATASET_IMAGES) {
            UnloadImageDataSetVisualizer(vis_state.draw_args.img_dataset_vis);
        }
        UnloadDataSet(vis_state.draw_args.current_dataset);
    }

    DrawingPanelFreeHistory(&vis_state.draw_args);
    UnloadRenderTexture(vis_state.draw_args.draw_texture);
    UnloadRenderTexture(vis_state.node_texture);
    UnloadRenderTexture(vis_state.node_outline_texture);
    free(vis_state.draw_args.output_buffer);
    free(vis_state.draw_args.add_dataset_file_name);
    free(vis_state.draw_args.images_dataset_width_input);
    free(vis_state.draw_args.images_dataset_height_input);

    for (int i = 0; i < vis_state.vis_args.model->num_layers; i++) {
        free(vis_state.node_positions[i]);
    }
    free(vis_state.node_positions);
}

void* window_run(void *vargp) {
    assert(!vis_state.is_window_open);
    SetTraceLogLevel(LOG_ERROR);

    visualizer_argument_t *vis_args = (visualizer_argument_t*) vargp;
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, TextFormat("%s Visualizer", vis_args->model_name));
    SetTargetFPS(60);

    initialize_visualizer(vis_args);

    // smoother rescaling textures for the drawing panel images to input into the model
    SetTextureFilter(vis_state.draw_args.draw_texture.texture, TEXTURE_FILTER_TRILINEAR);

    // set initial background of drawing to white
    BeginTextureMode(vis_state.draw_args.draw_texture);
    {
        ClearBackground(WHITE);
    }
    EndTextureMode();
    DrawingPanelAdd(&vis_state.draw_args);

    // preload circle textures for nodes
    vis_state.node_texture = LoadRenderTexture(2*NODE_RADIUS, 2*NODE_RADIUS);
    BeginTextureMode(vis_state.node_texture);
        DrawCircle(NODE_RADIUS, NODE_RADIUS, NODE_RADIUS, WHITE);
    EndTextureMode();
    vis_state.node_outline_texture = LoadRenderTexture(2*NODE_RADIUS, 2*NODE_RADIUS);
    BeginTextureMode(vis_state.node_outline_texture);
        DrawCircleLines(NODE_RADIUS, NODE_RADIUS, NODE_RADIUS, BLACK);
    EndTextureMode();

    construct_node_positions();

    // RUNNER
    window_keep_open(vis_args->model, 0);

    // CLEAN UP
    end_visualizer();
}

static void* train_run(void *vargp) {
    if (!vis_state.is_training && !vis_state.is_testing) {
        vis_state.is_training = true;
        CLOCK_MARK
        model_train_info((training_info_t*) vargp);
        CLOCK_MARK_ENTRY("TRAINING")
        vis_state.is_training = false;
    }
}

static void* test_run(void *vargp) {
    if (!vis_state.is_testing && !vis_state.is_training) {
        vis_state.is_testing = true;
        CLOCK_MARK
        model_test_info((training_info_t*) vargp);
        CLOCK_MARK_ENTRY("TESTING")
        vis_state.is_testing = false;
    }
}


//===================================================================
static layer_t *get_layer(int layer_index) {
    assert(layer_index < vis_state.vis_args.model->num_layers);

    layer_t *cur = vis_state.vis_args.model->input_layer;
    for (int i = 0; i < layer_index; i++) {
        cur = cur->next;
    }
    return cur;
}

static Vector2 get_node_position(int layer_index, int r) {
    return vis_state.node_positions[layer_index][r];
}

static Vector2 get_layer_topleft(int layer_index) {
    return vis_state.node_positions[layer_index][0];
}

static Vector2 get_layer_bottomright(int layer_index) {
    nmatrix_t nodes = layer_get_neurons(get_layer(layer_index));

    return vis_state.node_positions[layer_index][nodes.n_elements - 1]; // TODO bottom right position is not accurate if last section of layer has less nodes
    // instead store more info in node_positions (maybe make it an array of layer_info which contains node_positions)
}

// displays the highest priority tooltip
static void OpenTooltip(const char* msg, float priority, float *weight_value) {
    if (vis_state.show_tooltip && priority <= vis_state.tooltip_priority) {
        return;
    }

    int size = strlen(msg);
    memcpy(vis_state.tooltip_msg, msg, size+1);
    vis_state.tooltip_msg[TOOLTIP_BUFFER_SIZE-1] = '\0'; // safety null character
    vis_state.show_tooltip = true;
    vis_state.tooltip_priority = priority;
    vis_state.tooltip_weight_value = weight_value;
}

static void DrawLayerEdges(int layer_index, layer_t *layer, layer_t *prev) {
    nmatrix_t this_neurons = layer_get_neurons(layer);
    nmatrix_t prev_neurons = layer_get_neurons(prev);

    if (layer->type == DENSE) { // fully connected
        nmatrix_t weights = layer->layer.dense.weights;

        int skip = 1;
        if (weights.n_elements >= MAX_WEIGHTS_DRAWN_PER_LAYER) {
            skip = 1 + (weights.n_elements + MAX_WEIGHTS_DRAWN_PER_LAYER - 1) / MAX_WEIGHTS_DRAWN_PER_LAYER;
        }

        for (int r2 = 0; r2 < this_neurons.dims[0]; r2++) {
            // so we can color each weight based on its respective value to other weights
            // connecting to the same output neuron
            float max_weight = -1;
            for (int r1 = 0; r1 < prev_neurons.dims[0]; r1++) {
                float value = fabs(weights.matrix[r2 * prev_neurons.dims[0] + r1]);
                max_weight = max_weight < value ? value : max_weight;
            }

            for (int r1 = 0; r1 < prev_neurons.dims[0]; r1++) {
                if ((r2 * weights.dims[0] + r1) % skip != 0) {
                    continue;
                }

                Vector2 this_pos = get_node_position(layer_index, r2);
                Vector2 prev_pos = get_node_position(layer_index-1, r1);

                this_pos.x -= NODE_RADIUS;
                prev_pos.x += NODE_RADIUS;
                float ratio = max_weight != 0 ? 1 - fabs(weights.matrix[r2 * prev_neurons.dims[0] + r1]) / max_weight : 0.5;

                int cval = (int) (255 * ratio);
                Color color = {
                    .a = 255,
                    .r = cval, .g = cval, .b = cval,
                };
                DrawLineEx(this_pos, prev_pos, 3, BLACK);
                DrawLineV(this_pos, prev_pos, color);
                if (CheckCollisionPointLine(GetMousePosition(), prev_pos, this_pos, MOUSE_HOVER_DISTANCE_TO_WEIGHT)) {
                    // display information about the weight
                    char weight[WEIGHT_DISPLAY_PRECISION];
                    snprintf(weight, WEIGHT_DISPLAY_PRECISION, "%f", weights.matrix[r2 * prev_neurons.dims[0] + r1]);
                    OpenTooltip(weight, 1 / (0.1 + sqrt(pow(prev_pos.x - this_pos.x, 2) + pow(prev_pos.y - this_pos.y, 2))), &weights.matrix[r2 * prev_neurons.dims[0] + r1]);
                }
            }
        }
    } else { // Activation or Output, one to one connections
        assert(this_neurons.n_elements == prev_neurons.n_elements);
        for (int i = 0; i < prev_neurons.n_elements; i++) {
            Vector2 this_pos = get_node_position(layer_index, i);
            Vector2 prev_pos = get_node_position(layer_index-1, i);

            this_pos.x -= NODE_RADIUS;
            prev_pos.x += NODE_RADIUS;

            for (int dot = 1; dot <= WEIGHT_DOTTED_LINES; dot+=2) {
                Vector2 draw_end = {
                    .x = prev_pos.x + (int)((this_pos.x - prev_pos.x) * (dot / (float) WEIGHT_DOTTED_LINES)),
                    .y = prev_pos.y + (int)((this_pos.y - prev_pos.y) * (dot / (float) WEIGHT_DOTTED_LINES)),
                };

                Vector2 draw_start = {
                    .x = prev_pos.x + (int)((this_pos.x - prev_pos.x) * ((dot-1) / (float) WEIGHT_DOTTED_LINES)),
                    .y = prev_pos.y + (int)((this_pos.y - prev_pos.y) * ((dot-1) / (float) WEIGHT_DOTTED_LINES)),
                };

                DrawLineV(draw_start, draw_end, BLACK);
            }

            if (CheckCollisionPointLine(GetMousePosition(), prev_pos, this_pos, MOUSE_HOVER_DISTANCE_TO_WEIGHT)) {
                if (layer->type == ACTIVATION) {
                    // draw activated values
                    char activated_values[NODE_DISPLAY_PRECISION];
                    snprintf(activated_values, NODE_DISPLAY_PRECISION, "%f", layer->layer.activation.activated_values.matrix[i]);
                    OpenTooltip(activated_values, 1 / (0.1 + sqrt(pow(prev_pos.x - this_pos.x, 2) + pow(prev_pos.y - this_pos.y, 2))), NULL);
                } else if (layer->type == OUTPUT) {
                    // draw activated values
                    char output_values[NODE_DISPLAY_PRECISION];
                    snprintf(output_values, NODE_DISPLAY_PRECISION, "%f", layer->layer.output.guess.matrix[i]);
                    OpenTooltip(output_values, 1 / (0.1 + sqrt(pow(prev_pos.x - this_pos.x, 2) + pow(prev_pos.y - this_pos.y, 2))), NULL);
                }
            }
        }
    }
}

static void DrawLayerInformation(int layer_index, layer_t *layer) {
    nmatrix_t nodes = layer_get_neurons(layer);

    // draw layer name and information about it
    Vector2 layer_topleft = get_layer_topleft(layer_index);
    Vector2 layer_bottomright = get_layer_bottomright(layer_index);

    int layer_height = layer_bottomright.y - layer_topleft.y + 2*NODE_RADIUS;
    // offset the layer so that it is in the center vertically
    int layer_start_y = MODEL_Y + (MODEL_HEIGHT - layer_height) / 2;
    int layer_name_y = layer_start_y + layer_height + LAYER_NAME_OFFSET_Y + LAYER_DISPLAY_FONTSIZE / 2;
    int layer_function_name_y = layer_name_y + (3 * LAYER_DISPLAY_FONTSIZE) / 2;

    int layer_x = layer_topleft.x + (layer_bottomright.x - layer_topleft.x)/2;
    int layer_info_font_size = LAYER_DISPLAY_FONTSIZE - 4;
    DrawOutlinedCenteredText(get_layer_name(layer), layer_x, layer_name_y, LAYER_DISPLAY_FONTSIZE, BLACK, 0, BLACK);
    if (layer->type == ACTIVATION) {
        DrawOutlinedCenteredText(get_activation_function_name(&layer->layer.activation), layer_x, layer_function_name_y, layer_info_font_size, BLACK, 0, BLACK);
    } else if (layer->type == OUTPUT) {
        DrawOutlinedCenteredText(get_output_function_name(&layer->layer.output), layer_x, layer_function_name_y, layer_info_font_size, BLACK, 0, BLACK);
        int gap = (3 * LAYER_DISPLAY_FONTSIZE) / 2;
        DrawOutlinedCenteredText(get_output_guess_function_name(&layer->layer.output), layer_x, layer_function_name_y + gap, layer_info_font_size, BLACK, 0, BLACK);
    }
}

static void DrawLayer(int layer_index, layer_t *layer) {
    nmatrix_t nodes = layer_get_neurons(layer);

    // calculate values for color scaling
    float max_node_value = -1;
    float min_node_value = -1;
    for (int r = 0; r < nodes.dims[0]; r++) {
        float value = fabs(tanh(nodes.matrix[r]));
        if (max_node_value < 0 || min_node_value < 0) {
            max_node_value = value;
            min_node_value = value;
        } else {
            max_node_value = max_node_value < value ? value : max_node_value;
            min_node_value = min_node_value > value ? value : min_node_value;
        }
    }

    // draw each neuron
    for (int r = 0; r < nodes.dims[0]; r++) {
        // draw node with color respective to its value
        float ratio = .5;
        if (max_node_value != min_node_value) {
            ratio = (fabs(tanh(nodes.matrix[r])) - min_node_value) / (max_node_value - min_node_value);
        }
        Color target = nodes.matrix[r] < 0 ? NODE_NEG_COLOR : NODE_POS_COLOR;
        Vector3 HSV = ColorToHSV(target);
        Color shade = ColorFromHSV(HSV.x, HSV.y * ratio, HSV.z);
        Vector2 pos = get_node_position(layer_index, r);

        DrawTexture(vis_state.node_texture.texture, pos.x - NODE_RADIUS, pos.y - NODE_RADIUS, shade);
        DrawTexture(vis_state.node_outline_texture.texture, pos.x - NODE_RADIUS, pos.y - NODE_RADIUS, WHITE);

        if (layer->type == OUTPUT) {
            DrawCenteredText(vis_state.vis_args.output_labels[r], pos.x + NODE_RADIUS + NODE_GAP + 5, pos.y, 10, BLACK);
        }

        // edit input node values
        if (vis_state.playground_state && layer->type == INPUT && CheckCollisionPointCircle(GetMousePosition(), pos, MOUSE_HOVER_DISTANCE_TO_NODE)
                && !vis_state.draw_args.is_open) {
            // check if too small
            bool model_needs_update = false;
            if (NODE_RADIUS < MIN_NODE_RADIUS_FOR_SLIDER_BAR) {
                // shortcut to set node values
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                    nodes.matrix[r] = IsKeyDown(KEY_LEFT_SHIFT) ? -1 : 1;
                    model_needs_update = true;
                } else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
                    nodes.matrix[r] = 0;
                    model_needs_update = true;
                }
            } else {
                int rec_x = pos.x - NODE_RADIUS - 6;
                int rec_y = pos.y + NODE_DISPLAY_FONTSIZE - 4;
                int rec_width = NODE_RADIUS * 2 + 12;
                int rec_height = NODE_DISPLAY_FONTSIZE + 8;

                DrawRectangle(rec_x, rec_y, rec_width, rec_height, TOOLTIP_BACKGROUND_COLOR);
                DrawRectangleLines(rec_x, rec_y, rec_width, rec_height, BLACK);
                float prevValue = nodes.matrix[r];
                GuiSlider((Rectangle) {.x = pos.x - 3 * NODE_RADIUS / 4, .y = pos.y + NODE_DISPLAY_FONTSIZE, .width = 3 * NODE_RADIUS / 2, .height = NODE_DISPLAY_FONTSIZE},
                        "0", "1", &nodes.matrix[r], 0, 1);

                // run model on the changed inputs
                model_needs_update = prevValue != nodes.matrix[r];
            }

            if (model_needs_update) {
                nmatrix_t output = model_calculate(vis_state.vis_args.training_info->model);
            }
        }

        // draw value of node
        if (NODE_DISPLAY_PRECISION > MIN_NODE_PRECISION_FOR_DISPLAY) {
            char node_value[NODE_DISPLAY_PRECISION];
            snprintf(node_value, NODE_DISPLAY_PRECISION, "%f", (float) nodes.matrix[r]);
            DrawCenteredText(node_value, pos.x, pos.y, NODE_DISPLAY_FONTSIZE, BLACK);
        } else if (CheckCollisionPointCircle(GetMousePosition(), pos, NODE_RADIUS)) { // display tooltip
            // display information about the weight
            char weight[WEIGHT_DISPLAY_PRECISION];
            snprintf(weight, WEIGHT_DISPLAY_PRECISION, "%f", nodes.matrix[r]);
            OpenTooltip(weight, 1000000, &nodes.matrix[r]);
        }
    }

    DrawLayerInformation(layer_index, layer);
}

static void DrawNeuralNetwork(neural_network_model_t *model) {
    // draw model background
    Color color = GRAY;
    color.a = 170;
    DrawRectangle(MODEL_X, MODEL_Y, MODEL_WIDTH, MODEL_HEIGHT, color);
    DrawCenteredText(vis_state.vis_args.model_name, MODEL_WIDTH/2, 35, 20, BLACK);

    // draw each layer
    layer_t *cur = model->input_layer;
    for (int layer_i = 0; layer_i < model->num_layers; layer_i++) {
        if (layer_i != model->num_layers-1) {
            DrawLayerEdges(layer_i+1, cur->next, cur);
        }
        DrawLayer(layer_i, cur);
        cur = cur->next;
    }
}

static void DrawTrainingInfo(void) {
    training_info_t *t_info = vis_state.vis_args.training_info;
    DrawText("# Train:\n# Test:\nTrain Acc:\nAvg Train Err:\nTest Acc:\nAvg Test Err:\n\nEpoch:\nTrain Index:\nTest Index:",
            MODEL_X + 20, MODEL_Y + 100, 16, DARKGRAY);
    DrawText(TextFormat("%d\n%d\n%.2f\n%.3f\n%.2f\n%.3f\n\n%d\n%d\n%d",
            t_info->train_size, t_info->test_size, t_info->train_accuracy, t_info->avg_train_error, t_info->test_accuracy,
            t_info->avg_test_error, t_info->epoch, t_info->train_index, t_info->test_index), MODEL_X + 160, MODEL_Y + 100, 16, BLACK);

    DrawText(TextFormat("Batch: %d\nLearning: %.3f\nEpochs: %d\nTrgt Acc: %.2f",
            t_info->batch_size, t_info->learning_rate, t_info->target_epochs, t_info->target_accuracy),
            MODEL_X + 20, MODEL_Y + 350, 12, BLACK);

    if (!vis_state.is_training && !vis_state.is_testing) {
        Rectangle batch_size_r = {
            .x = MODEL_X + 160,
            .y = MODEL_Y + 350,
            .width = 60,
            .height = 16
        };
        float value = t_info->batch_size;
        if (GuiSlider(batch_size_r, "", "64", &value, 1, 64)) {
            _TOGGLE_BOOL(&vis_state.vis_args.is_batch_size_active);
        }
        t_info->batch_size = value;

        if (GuiSlider(RecShift(batch_size_r, 0, 20), "", "1", &t_info->learning_rate, 0.001, 0.1)) {
            _TOGGLE_BOOL(&vis_state.vis_args.is_learning_rate_active);
        }

        value = t_info->target_epochs;
        if (GuiSlider(RecShift(batch_size_r, 0, 40), "", "50", &value, 1, 50)) {
            _TOGGLE_BOOL(&vis_state.vis_args.is_target_epochs_active);
        }
        t_info->target_epochs = value;

        if (GuiSlider(RecShift(batch_size_r, 0, 60), "", "1.1", &t_info->target_accuracy, 0.5, 1.1)) {
            _TOGGLE_BOOL(&vis_state.vis_args.is_target_accuracy_active);
        }
    }
}

static nmatrix_t set_training_set_display(bool is_train, int loc) {
    training_info_t *t_info = vis_state.vis_args.training_info;
    assert(loc >= 0 && loc < is_train ? t_info->train_size : t_info->test_size);

    neural_network_model_t *model = vis_state.vis_args.training_info->model;
    int *cur = &vis_state.current_example;
    *cur = loc;
    nmatrix_memcpy(&model->input_layer->layer.input.input_values, is_train ? &t_info->train_x[*cur] : &t_info->test_x[*cur]);
    return model_calculate(model);
}

static nmatrix_t move_training_set_display(bool is_train, int move) {
    training_info_t *t_info = vis_state.vis_args.training_info;
    int *cur = &vis_state.current_example;
    int max = is_train ? t_info->train_size : t_info->test_size;

    if ((*cur) + move < 0) {
        *cur = 0;
    } else if ((*cur) + move >= max) {
        *cur = max - 1;
    } else {
        *cur = (*cur) + move;
    }

    return set_training_set_display(is_train, *cur);
}

static void DrawTrainingExamplesDisplay(void) {
    if (vis_state.vis_args.training_info->train_size == 0 || vis_state.vis_args.training_info->test_size == 0) {
        return;
    }

    Rectangle display_train_examples_r = {
        .x = MODEL_X + 30,
        .y = MODEL_Y + 450,
        .width = 80,
        .height = 30,
    };
    Rectangle display_test_examples_r = RecShift(display_train_examples_r, 100, 0);

    bool prev_state = vis_state.show_training;
    GuiToggle(display_train_examples_r, "See Train", &vis_state.show_training);
    if (!prev_state && vis_state.show_training) {
        vis_state.show_testing = false;
        set_training_set_display(true, 0);
    }

    prev_state = vis_state.show_testing;
    GuiToggle(display_test_examples_r, "See Test", &vis_state.show_testing);
    if (!prev_state && vis_state.show_testing) {
        vis_state.show_training = false;
        set_training_set_display(false, 0);
    }

    if (vis_state.show_training || vis_state.show_testing) {
        Rectangle prev_incorrect_button_r = {
            .x = display_train_examples_r.x + display_test_examples_r.width + 10 - 63,
            .y = display_train_examples_r.y + display_train_examples_r.height + 15,
            .width = 35,
            .height = 15,
        };

        Rectangle prev_button_r = {
            .x = prev_incorrect_button_r.x + prev_incorrect_button_r.width + 5,
            .y = prev_incorrect_button_r.y,
            .width = 20,
            .height = 15,
        };

        Rectangle next_button_r = {
            .x = prev_button_r.x + prev_button_r.width + 6,
            .y = prev_button_r.y,
            .width = 20,
            .height = 15,
        };

        Rectangle next_incorrect_button_r = {
            .x = next_button_r.x + next_button_r.width + 5,
            .y = next_button_r.y,
            .width = 35,
            .height = 15,
        };

        training_info_t *t_info = vis_state.vis_args.training_info;
        bool is_train = vis_state.show_training;
        nmatrix_t *correct = is_train ? t_info->train_y : t_info->test_y;
        int *cur = &vis_state.current_example;
        int max = is_train ? t_info->train_size : t_info->test_size;
        const char* display_name = is_train ? "Train" : "Test";

        if (GuiButton(prev_incorrect_button_r, "<<<")) {
            nmatrix_t output;
            do {
                output = move_training_set_display(is_train, -1);
            } while (nmatrix_equal(&output, &correct[*cur]) && *cur > 0 && *cur < max - 1);
        }
        if (GuiButton(prev_button_r, "<")) {
            move_training_set_display(is_train, -1);
        }

        if (GuiButton(next_button_r, ">")) {
            move_training_set_display(is_train, 1);
        }

        if (GuiButton(next_incorrect_button_r, ">>>")) {
            nmatrix_t output;
            do {
                output = move_training_set_display(is_train, 1);
            } while (nmatrix_equal(&output, &correct[*cur]) && *cur > 0 && *cur < max - 1);
        }

        DrawText(TextFormat("Displaying Example: %d", *cur), prev_incorrect_button_r.x, prev_incorrect_button_r.y + 20, 16, BLACK);
        DrawText(TextFormat("Expected Label: %s", vis_state.vis_args.output_labels[unpack_one_hot_encoded(correct[*cur])]),
                prev_incorrect_button_r.x, prev_incorrect_button_r.y + 40, 16, BLACK);
    }
}

static void DrawWindow(neural_network_model_t *model) {
    SetTextLineSpacing(20);
    BeginDrawing();
    {
        ClearBackground(RAYWHITE);
        DrawText(TextFormat("%d FPS", GetFPS()), 5, 5, 10, BLACK);

        vis_state.show_tooltip = false;
        if (vis_state.draw_args.updated_training_info) {
            vis_state.draw_args.updated_training_info = false;
            vis_state.current_example = 0;
            vis_state.has_trained = false;
            vis_state.show_training = false;
        }

        // maybe have a better way to signify if a model is built or not
        if (model && model->input_layer != NULL) {
            DrawNeuralNetwork(model);
        }

        DrawTrainingInfo();

        if (!vis_state.is_training && !vis_state.is_testing && vis_state.has_trained) {
            DrawTrainingExamplesDisplay();
        }

        // some model control buttons
        int control_y = MODEL_Y + 40;
        if (GuiButton((Rectangle) {.x = MODEL_X + 30, .y = control_y, .height = 30, .width = 130}, "Start Training") && !vis_state.draw_args.is_open) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, train_run, vis_state.vis_args.training_info);
            pthread_detach(thread_id);
            vis_state.has_trained = true;
        }

        if (GuiButton((Rectangle) {.x = MODEL_X + 170, .y = control_y, .height = 30, .width = 100}, "Start Test") && !vis_state.draw_args.is_open) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, test_run, vis_state.vis_args.training_info);
            pthread_detach(thread_id);
        }

        GuiToggle((Rectangle) {.x = MODEL_X + 280, .y = control_y, .height = 30, .width = 100}, "Playground", &vis_state.playground_state);
        if (vis_state.is_testing || vis_state.is_training) { // dont mess with training
            vis_state.playground_state = false;
        }

        if (vis_state.vis_args.allow_drawing_panel_as_model_input) {
            if (GuiButton((Rectangle) {.x = MODEL_X + 390, .y = control_y, .height = 30, .width = 120}, "Drawing Panel") && !vis_state.draw_args.is_open) {
                vis_state.draw_args.is_open = true;
            }
            GuiDrawingPanelPopup(&vis_state.draw_args);
        }

        // draw tooltip
        if (vis_state.show_tooltip && !vis_state.draw_args.is_open) {
            Vector2 mouse_pos = GetMousePosition();
            int rec_x = mouse_pos.x;
            int rec_y = mouse_pos.y - TOOLTIP_HEIGHT;
            DrawRectangleLines(rec_x, rec_y, TOOLTIP_WIDTH, TOOLTIP_HEIGHT, BLACK);
            DrawRectangle(rec_x, rec_y, TOOLTIP_WIDTH, TOOLTIP_HEIGHT, TOOLTIP_BACKGROUND_COLOR);
            DrawCenteredText(vis_state.tooltip_msg, mouse_pos.x + TOOLTIP_WIDTH / 2, mouse_pos.y - TOOLTIP_HEIGHT / 2, TOOLTIP_FONTSIZE, TOOLTIP_FONTCOLOR);

            if (vis_state.tooltip_weight_value && vis_state.playground_state && !vis_state.is_testing && !vis_state.is_training) { // todo replace with model.isLocked instead
                *vis_state.tooltip_weight_value += WEIGHT_VALUE_MOUSEWHEEL_SCALE * GetMouseWheelMove();
                nmatrix_t output = model_calculate(vis_state.vis_args.training_info->model); // todo preferrably run on separate thread
            }

            vis_state.show_tooltip = false;
            vis_state.tooltip_priority = 0;
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
