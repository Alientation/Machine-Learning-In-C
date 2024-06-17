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
    
    .node_positions = NULL,

    .show_tooltip = false,
    .tooltip_priority = 0,
    .tooltip_weight_value = NULL,

    .is_window_open = false,
}; 

//===========================================================================
int compute_network_width() {
    layer_t *cur_layer = vis_state.vis_args.model->input_layer;
    int network_width = LAYER_GAP * (vis_state.vis_args.model->num_layers - 1);
    for (int i = 0; i < vis_state.vis_args.model->num_layers; i++) {
        mymatrix_t neurons = layer_get_neurons(cur_layer);
        int num_nodes = neurons.r * neurons.c;
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

void construct_node_positions() {
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
        mymatrix_t neurons = layer_get_neurons(cur_layer);
        int num_nodes = neurons.r * neurons.c;
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

void initialize_visualizer(visualizer_argument_t *vis_args) {
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
        .image_dataset_visualizer = {0},
        .dataset_list_scroll_index = 0, 
        .sel_dataset_image_index = -1,
        
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
    memset(vis_state.draw_args.add_dataset_file_name, 0, (FILE_NAME_BUFFER_SIZE + 1) * sizeof(char));
    memset(vis_state.draw_args.images_dataset_width_input, 0, (NUMBER_INPUT_BUFFER_SIZE + 1) * sizeof(char));
    vis_state.draw_args.images_dataset_width_input[0] = '2';
    vis_state.draw_args.images_dataset_width_input[1] = '8';
    memset(vis_state.draw_args.images_dataset_height_input, 0, (NUMBER_INPUT_BUFFER_SIZE + 1) * sizeof(char));
    vis_state.draw_args.images_dataset_height_input[0] = '2';
    vis_state.draw_args.images_dataset_height_input[1] = '8';    
}

void end_visualizer() {
    // save the currently open dataset
    if (vis_state.draw_args.sel_dataset_index != -1) {
        WriteDataSet(vis_state.draw_args.current_dataset);

        if (vis_state.draw_args.current_dataset.type == DATASET_IMAGES) {
            UnloadImageDataSetVisualizer(vis_state.draw_args.image_dataset_visualizer);
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
    // SetTraceLogLevel(LOG_ERROR);

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
        model_test_info((training_info_t*) vargp);
        CLOCK_MARK_ENTRY("TESTING")
        vis_state.is_testing = false;
    }
}


//===================================================================
layer_t *get_layer(int layer_index) {
    assert(layer_index < vis_state.vis_args.model->num_layers);

    layer_t *cur = vis_state.vis_args.model->input_layer;
    for (int i = 0; i < layer_index; i++) {
        cur = cur->next;
    }
    return cur;
}

Vector2 get_node_position(int layer_index, int r) {
    return vis_state.node_positions[layer_index][r];    
}

Vector2 get_layer_topleft(int layer_index) {
    return vis_state.node_positions[layer_index][0];
}

Vector2 get_layer_bottomright(int layer_index) {
    mymatrix_t nodes = layer_get_neurons(get_layer(layer_index));
    
    return vis_state.node_positions[layer_index][nodes.r * nodes.c - 1]; // TODO bottom right position is not accurate if last section of layer has less nodes
    // instead store more info in node_positions (maybe make it an array of layer_info which contains node_positions)
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

void DrawLayerEdges(int layer_index, layer_t *layer, layer_t *prev) {
    mymatrix_t this_neurons = layer_get_neurons(layer);
    mymatrix_t prev_neurons = layer_get_neurons(prev);
    
    if (layer->type == DENSE) { // fully connected
        mymatrix_t weights = layer->layer.dense.weights;

        int skip = 1;
        if (weights.r * weights.c >= MAX_WEIGHTS_DRAWN_PER_LAYER) {
            skip = 1 + (weights.r * weights.c + MAX_WEIGHTS_DRAWN_PER_LAYER - 1) / MAX_WEIGHTS_DRAWN_PER_LAYER;
        }

        for (int r2 = 0; r2 < this_neurons.r; r2++) {
            // so we can color each weight based on its respective value to other weights
            // connecting to the same output neuron
            float max_weight = -1;
            for (int r1 = 0; r1 < prev_neurons.r; r1++) {
                float value = fabs(weights.matrix[r2][r1]);
                max_weight = max_weight < value ? value : max_weight;
            }
            
            for (int r1 = 0; r1 < prev_neurons.r; r1++) {
                if ((r2 * weights.r + r1) % skip != 0) {
                    continue;
                }

                Vector2 this_pos = get_node_position(layer_index, r2);
                Vector2 prev_pos = get_node_position(layer_index-1, r1);

                this_pos.x -= NODE_RADIUS;
                prev_pos.x += NODE_RADIUS;
                float ratio = max_weight != 0 ? 1 - fabs(weights.matrix[r2][r1]) / max_weight : 0.5;
                
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
                    snprintf(weight, WEIGHT_DISPLAY_PRECISION, "%f", weights.matrix[r2][r1]);
                    OpenTooltip(weight, 1 / (0.1 + sqrt(pow(prev_pos.x - this_pos.x, 2) + pow(prev_pos.y - this_pos.y, 2))), &weights.matrix[r2][r1]);
                }
            }
        }
    } else { // Activation or Output, one to one connections
        assert(this_neurons.r == prev_neurons.r);
        for (int r = 0; r < prev_neurons.r; r++) {
            Vector2 this_pos = get_node_position(layer_index, r);
            Vector2 prev_pos = get_node_position(layer_index-1, r);

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
        }
    }
}

void DrawLayerInformation(int layer_index, layer_t *layer) {
    mymatrix_t nodes = layer_get_neurons(layer);

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
        Color target = nodes.matrix[r][0] < 0 ? NODE_NEG_COLOR : NODE_POS_COLOR;
        Vector3 HSV = ColorToHSV(target);
        Color shade = ColorFromHSV(HSV.x, HSV.y * ratio, HSV.z);
        Vector2 pos = get_node_position(layer_index, r);
        
        DrawTexture(vis_state.node_texture.texture, pos.x - NODE_RADIUS, pos.y - NODE_RADIUS, shade);
        DrawTexture(vis_state.node_outline_texture.texture, pos.x - NODE_RADIUS, pos.y - NODE_RADIUS, WHITE);
        
        // edit input node values
        if (vis_state.playground_state && layer->type == INPUT && CheckCollisionPointCircle(GetMousePosition(), pos, MOUSE_HOVER_DISTANCE_TO_NODE)
                && !vis_state.draw_args.is_open) {
            // check if too small
            bool model_needs_update = false;
            if (NODE_RADIUS < MIN_NODE_RADIUS_FOR_SLIDER_BAR) {
                // shortcut to set node values
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                    nodes.matrix[r][0] = 1;
                    model_needs_update = true;
                } else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
                    nodes.matrix[r][0] = 0;
                    model_needs_update = true;
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

                // run model on the changed inputs
                model_needs_update = true;
            }

            if (model_needs_update) {
                mymatrix_t output = model_calculate(vis_state.vis_args.training_info->model);
            }
        }
        
        // draw value of node
        if (NODE_DISPLAY_PRECISION > MIN_NODE_PRECISION_FOR_DISPLAY) {
            char node_value[NODE_DISPLAY_PRECISION];
            snprintf(node_value, NODE_DISPLAY_PRECISION, "%f", (float) nodes.matrix[r][0]);
            DrawCenteredText(node_value, pos.x, pos.y, NODE_DISPLAY_FONTSIZE, BLACK);
        } else if (CheckCollisionPointCircle(GetMousePosition(), pos, NODE_RADIUS)) { // display tooltip
            // display information about the weight
            char weight[WEIGHT_DISPLAY_PRECISION];
            snprintf(weight, WEIGHT_DISPLAY_PRECISION, "%f", nodes.matrix[r][0]);
            OpenTooltip(weight, 1000000, &nodes.matrix[r][0]);
        }
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
        if (layer_i != model->num_layers-1) {
            DrawLayerEdges(layer_i+1, cur->next, cur);
        }
        DrawLayer(layer_i, cur);
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
        int control_y = MODEL_Y + 40;
        if (GuiButton((Rectangle) {.x = MODEL_X + 30, .y = control_y, .height = 30, .width = 130}, "Start Training") && !vis_state.draw_args.is_open) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, train_run, vis_state.vis_args.training_info);
            pthread_detach(thread_id);
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
                mymatrix_t output = model_calculate(vis_state.vis_args.training_info->model); // todo preferrably run on separate thread
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

