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
visualizer_argument_t vis_args;
extern training_info_t training_info;
bool is_training = false;
bool is_testing = false;
bool playground_state = false;

static bool is_window_open = false;
static const int SCREEN_WIDTH = 1200;
static const int SCREEN_HEIGHT = 700;
static const int MODEL_WIDTH = 1160;
static const int MODEL_HEIGHT = 450;
static const int MODEL_X = 20;
static const int MODEL_Y = 20;

static int model_start_x = MODEL_X;
static int model_start_y = MODEL_Y;

static const Color NODE_POS_COLOR = {
    .a = 128,
    .r = 150, .g = 255, .b = 173
};
static const Color NODE_NEG_COLOR = {
    .a = 128,
    .r = 255, .g = 178, .b = 150
};

#define SCALING_FACTOR 0.7

static const int WEIGHT_DISPLAY_PRECISION = ceil(9 * SCALING_FACTOR);
static const int NODE_DISPLAY_PRECISION = ceil(7 * SCALING_FACTOR);
static const int WEIGHT_DOTTED_LINES = 2 * (((int)ceil((11 * SCALING_FACTOR))) / 2) + 1;

static const int NODE_DISPLAY_FONTSIZE = 15 * SCALING_FACTOR;
static const int NODE_RADIUS = 35 * SCALING_FACTOR;
static const int NODE_GAP = 20 * SCALING_FACTOR;
static const int LAYER_DISPLAY_FONTSIZE = 16 * SCALING_FACTOR;
static const int LAYER_NAME_OFFSET_Y = 20 * SCALING_FACTOR;
static const int LAYER_GAP = 60 * SCALING_FACTOR;
static const int MAX_LAYER_NODES = 3 / SCALING_FACTOR;
static const int HIDDEN_LAYER_WIDTH = NODE_RADIUS * 2;
static const int HIDDEN_LAYER_HEIGHT = MAX_LAYER_NODES * (NODE_GAP + NODE_RADIUS * 2) - NODE_GAP;

static const int MIN_NODE_RADIUS_FOR_SLIDER_BAR = 20;

static const int MOUSE_HOVER_DISTANCE_TO_NODE = NODE_RADIUS + NODE_GAP/2;
static const int MOUSE_HOVER_DISTANCE_TO_WEIGHT = ceil(10 * SCALING_FACTOR);

static const int TOOLTIP_BORDER = 10;
static const int TOOLTIP_HEIGHT = 30;
static const int TOOLTIP_WIDTH = 100;
static const int TOOLTIP_FONTSIZE = 16;
static const Color TOOLTIP_FONTCOLOR = BLACK;
static const Color TOOLTIP_BACKGROUND_COLOR = {
    .a = 255,
    .r = 227,
    .g = 215,
    .b = 252
};

static const int TOOLTIP_BUFFER_SIZE = 100;
static bool show_tooltip = false;
static char tooltip_msg[100];
static float tooltip_priority = 0;
static float *tooltip_weight_value = NULL;
static float TOOLTIP_WEIGHT_VALUE_SCALE = 0.05;


struct DrawingPanelArgs {
    bool isOpen;
    float brush_size;
    Vector2 prev_draw_pos;

    RenderTexture2D *drawnImage;

    // scaled down
    int buffer_width;
    int buffer_height;
    float *output_buffer;
    bool updated;
} drawing_panel_args; 

//===========================================================================

void* window_run(void *vargp) {
    assert(!is_window_open);

    visualizer_argument_t *args = (visualizer_argument_t*) vargp;
    vis_args = *args;    
    is_window_open = true;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, TextFormat("%s Visualizer", args->model_name));
    SetTargetFPS(60);    

    drawing_panel_args = (struct DrawingPanelArgs) {
        .isOpen = false,
        .brush_size = 4,
        .prev_draw_pos = (Vector2) {.x = -1, .y = -1},
        .drawnImage = malloc(sizeof(RenderTexture2D)),
        .updated = false,
        .buffer_width = 10,
        .buffer_height = 10,
        .output_buffer = malloc(sizeof(float) * 100)
    };
    *drawing_panel_args.drawnImage = LoadRenderTexture(400, 400);

    BeginTextureMode(*drawing_panel_args.drawnImage);
    {
        ClearBackground(WHITE);
    }
    EndTextureMode();


    window_keep_open(args->model, 0);

    UnloadRenderTexture(*drawing_panel_args.drawnImage);
    free(drawing_panel_args.drawnImage);
    free(drawing_panel_args.output_buffer);
}

void* train_run(void *vargp) {
    if (!is_training && !is_testing) {
        is_training = true;
        CLOCK_MARK
        model_train_info((training_info_t*) vargp);
        CLOCK_MARK_ENTRY("TRAINING")
        is_training = false;
    }
}

void* test_run(void *vargp) {
    if (!is_testing && !is_training) {
        is_testing = true;
        CLOCK_MARK
        // model_test_info((training_info_t*) vargp); // TODO
        CLOCK_MARK_ENTRY("TESTING")
        is_testing = false;
    }
}


//===================================================================

void GuiDrawingPanelPopup(struct DrawingPanelArgs *args) {
    if (!args->isOpen) {
        return;
    }
    
    // darken background
    DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, ColorAlpha(BLACK, 0.5));

    // drawing panel
    RenderTexture2D *draw_image = args->drawnImage;
    
    int draw_window_x = 50;
    int draw_window_y = 50;
    int draw_window_width = SCREEN_WIDTH - 2 * draw_window_x;
    int draw_window_height = SCREEN_HEIGHT - 2 * draw_window_y;

    int draw_panel_x = draw_window_x + draw_window_width/2 - draw_image->texture.width/2;
    int draw_panel_y = draw_window_y + draw_window_height/2 - draw_image->texture.height/2;

    args->isOpen = !GuiWindowBox((Rectangle) {.x = draw_window_x, .y = draw_window_y, .width = draw_window_width, .height = draw_window_height}, "Drawing Panel");
    
    Rectangle draw_panel_rec = {.x = draw_panel_x, .y = draw_panel_y, .width = draw_image->texture.width, .height = draw_image->texture.height};
    DrawRectangleRoundedLines(draw_panel_rec, .02, 10, 5, BLACK);

    bool is_draw = IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsGestureDetected(GESTURE_DRAG);
    bool is_erase = IsMouseButtonDown(MOUSE_BUTTON_RIGHT);
    if ((is_draw || is_erase) && CheckCollisionPointRec(GetMousePosition(), draw_panel_rec)) {
        Color draw_color = {
            .a = 255,
            .r = is_erase * 255,
            .g = is_erase * 255,
            .b = is_erase * 255,
        };

        Vector2 size = {
            .x = args->brush_size,
            .y = args->brush_size,
        };
        Vector2 pos = GetMousePosition();
        pos.x -= size.x / 2.0;
        pos.y -= size.y / 2.0;

        pos.x -= draw_panel_x;
        pos.y -= draw_panel_y;

        pos.y = draw_image->texture.height - pos.y; // texture drawing flips vertically because of opengl or smthn

        BeginTextureMode(*draw_image);
        {
            DrawCircleV(pos, args->brush_size, draw_color);
            
            // to connect points since the refresh rate is 60 fps which causes gaps
            if (args->prev_draw_pos.x != -1 && args->prev_draw_pos.y != -1) {
                DrawLineEx(args->prev_draw_pos, pos, args->brush_size * 2, draw_color);
            }

            // TODO try to fix
            // when dragging mouse past border of draw panel, there is a small gap between the border and the last drawn
            // position. Perhaps track whenever the mouse drag JUST leaves the area, and still draw that line segment
            
            args->prev_draw_pos = pos;
        }
        EndTextureMode();

    } else {
        args->prev_draw_pos = (Vector2) {.x = GetMousePosition().x - (args->brush_size/2) - draw_panel_x, 
                .y = draw_image->texture.height - (GetMousePosition().y - (args->brush_size/2) - draw_panel_y)};
    }

    DrawTexture(draw_image->texture, draw_panel_x, draw_panel_y, WHITE);
}

Vector2 get_node_position(int layer_index, int r, mymatrix_t nodes) {
    int layer_height = nodes.r * (NODE_GAP + 2 * NODE_RADIUS) - NODE_GAP;
    int layer_start_y = model_start_y + (MODEL_HEIGHT - layer_height) / 2;

    int x = model_start_x + layer_index * (LAYER_GAP + 2 * NODE_RADIUS) + NODE_RADIUS;
    int y = layer_start_y + r * (NODE_GAP + 2 * NODE_RADIUS) + NODE_RADIUS;
    
    if (nodes.r > MAX_LAYER_NODES) {
        // used for drawing weights since those can be squished together
        y = (((float) r) / nodes.r) * HIDDEN_LAYER_HEIGHT;
    }
    return (Vector2) {.x = x, .y = y};
}

// displays the highest priority tooltip
void OpenTooltip(const char* msg, float priority, float *weight_value) {
    if (show_tooltip && priority <= tooltip_priority) {
        return;
    }

    int size = strlen(msg);
    memcpy(tooltip_msg, msg, size);
    tooltip_msg[TOOLTIP_BUFFER_SIZE-1] = '\0'; // safety null character
    show_tooltip = true;
    tooltip_priority = priority;
    tooltip_weight_value = weight_value;
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
    int layer_height = nodes.r * (NODE_GAP + 2 * NODE_RADIUS) - NODE_GAP;
    int layer_start_y = model_start_y + (MODEL_HEIGHT - layer_height) / 2;
    int layer_name_y = layer_start_y + layer_height + LAYER_NAME_OFFSET_Y + LAYER_DISPLAY_FONTSIZE / 2;
    int layer_function_name_y = layer_name_y - LAYER_DISPLAY_FONTSIZE / 2 + LAYER_DISPLAY_FONTSIZE * 2;
    int layer_x = model_start_x + layer_index * (LAYER_GAP + 2 * NODE_RADIUS) + NODE_RADIUS;
    int layer_info_font_size = LAYER_DISPLAY_FONTSIZE - 4;
    DrawOutlinedCenteredText(get_layer_name(layer), layer_x, layer_name_y, LAYER_DISPLAY_FONTSIZE, WHITE, 1, BLACK);
    if (layer->type == ACTIVATION) {
        DrawOutlinedCenteredText(get_activation_function_name(&layer->layer.activation), layer_x, 
                layer_function_name_y, layer_info_font_size, WHITE, 1, BLACK);
    } else if (layer->type == OUTPUT) {
        DrawOutlinedCenteredText(get_output_function_name(&layer->layer.output), layer_x, 
                layer_function_name_y, layer_info_font_size, WHITE, 1, BLACK);
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
        if (playground_state && layer->type == INPUT && CheckCollisionPointCircle(GetMousePosition(), pos, MOUSE_HOVER_DISTANCE_TO_NODE)
                && !drawing_panel_args.isOpen) {
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
                mymatrix_t output = model_calculate(training_info.model);
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
    // calculate center x for the model
    int model_width = model->num_layers * (LAYER_GAP + 2 * NODE_RADIUS) - LAYER_GAP;
    model_start_x = MODEL_X + (MODEL_WIDTH - model_width) / 2;
    model_start_y = MODEL_Y;

    // draw model background
    Color color = DARKBLUE;
    color.a = 170;
    DrawRectangle(MODEL_X, MODEL_Y, MODEL_WIDTH, MODEL_HEIGHT, color);
    DrawCenteredText(vis_args.model_name, MODEL_WIDTH/2, 35, 20, BLACK);

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

        show_tooltip = false;        
        // maybe have a better way to signify if a model is built or not
        if (model && model->input_layer != NULL) {
            DrawNeuralNetwork(model);
        }

        // draw model's training_info
        // TODO

        // some model control buttons
        if (GuiButton((Rectangle) {.x = 50, .y = 60, .height = 30, .width = 130}, "Start Training") && !drawing_panel_args.isOpen) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, train_run, &training_info);
            pthread_detach(thread_id);
        }

        if (GuiButton((Rectangle) {.x = 190, .y = 60, .height = 30, .width = 100}, "Start Test") && !drawing_panel_args.isOpen) {
            pthread_t thread_id;
            pthread_create(&thread_id, NULL, test_run, &training_info);
            pthread_detach(thread_id);
        }

        GuiToggle((Rectangle) {.x = 300, .y = 60, .height = 30, .width = 100}, "Playground", &playground_state);
        if (is_testing || is_training) { // dont mess with training
            playground_state = false;
        }

        if (GuiButton((Rectangle) {.x = 410, .y = 60, .height = 30, .width = 120}, "Drawing Panel") && !drawing_panel_args.isOpen) {
            drawing_panel_args.isOpen = true;
        }
        GuiDrawingPanelPopup(&drawing_panel_args);

        // draw tooltip
        if (show_tooltip && !drawing_panel_args.isOpen) {
            Vector2 mouse_pos = GetMousePosition();
            int rec_x = mouse_pos.x;
            int rec_y = mouse_pos.y - TOOLTIP_HEIGHT;
            DrawRectangleLines(rec_x, rec_y, TOOLTIP_WIDTH, TOOLTIP_HEIGHT, BLACK);
            DrawRectangle(rec_x, rec_y, TOOLTIP_WIDTH, TOOLTIP_HEIGHT, TOOLTIP_BACKGROUND_COLOR);
            DrawCenteredText(tooltip_msg, mouse_pos.x + TOOLTIP_WIDTH / 2, mouse_pos.y - TOOLTIP_HEIGHT / 2, TOOLTIP_FONTSIZE, TOOLTIP_FONTCOLOR);

            if (tooltip_weight_value && playground_state && !is_testing && !is_training) { // todo replace with model.isLocked instead
                *tooltip_weight_value += TOOLTIP_WEIGHT_VALUE_SCALE * GetMouseWheelMove();
                mymatrix_t output = model_calculate(training_info.model); // todo preferrably run on separate thread
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
    is_window_open = false;
}

