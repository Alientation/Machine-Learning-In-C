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

selection of train/test data, graph of train/test accuracy and output loss

MAYBE BUT NOT NECESSARY
split rendering from logic (like the draw panel) so we can run each on its own thread and mouse inputs are registered faster
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


typedef struct SegmentListNode {
    RenderTexture2D saved_image;
    struct SegmentListNode *next;
    struct SegmentListNode *prev;
} segment_list_node_t;

struct DrawingPanelArgs {
    bool isOpen;
    float brush_size;
    Vector3 brush_color; // in HSV

    Vector2 prev_draw_pos;
    bool is_dragged;
    bool is_drawing;
    RenderTexture2D drawn_image;

    segment_list_node_t *segments_list_head;
    segment_list_node_t *segments_list_cur;
    int segments_queue_size;

    // scaled down
    bool updated;
    RenderTexture2D model_input_image;
    int buffer_width;
    int buffer_height;
    float *output_buffer;
} drawing_panel_args; 

void DrawingPanelFreeHistory(struct DrawingPanelArgs *args);
void DrawingPanelAdd(struct DrawingPanelArgs *args);

//===========================================================================

void* window_run(void *vargp) {
    assert(!is_window_open);
    SetTraceLogLevel(LOG_ERROR); 

    visualizer_argument_t *args = (visualizer_argument_t*) vargp;
    vis_args = *args;    
    is_window_open = true;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, TextFormat("%s Visualizer", args->model_name));
    SetTargetFPS(60);    

    drawing_panel_args = (struct DrawingPanelArgs) {
        .isOpen = false,
        .brush_size = 4,
        .brush_color = ColorToHSV(BLACK),
        
        .prev_draw_pos = (Vector2) {.x = -1, .y = -1},
        .is_dragged = false,
        .is_drawing = false,
        .drawn_image = LoadRenderTexture(400, 400),
        
        .segments_list_head = NULL,
        .segments_list_cur = NULL,
        .segments_queue_size = 0,

        .updated = false,
        .model_input_image = LoadRenderTexture(10, 10),
        .buffer_width = 10,
        .buffer_height = 10,
        .output_buffer = malloc(sizeof(float) * 100)
    };

    BeginTextureMode(drawing_panel_args.drawn_image);
    {
        ClearBackground(WHITE);
    }
    EndTextureMode();
    DrawingPanelAdd(&drawing_panel_args);

    window_keep_open(args->model, 0);

    DrawingPanelFreeHistory(&drawing_panel_args);
    UnloadRenderTexture(drawing_panel_args.drawn_image);
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

// TODO perhaps for optimization, but not really critical, instead of storing these saved
// drawing panel images as render textures which would be put in VRAM, store as images on RAM and
// write to the drawing panel's render texture when needed (saves VRAM space and time spent sending these textures to it after each draw segment)
void DrawingPanelFreeHistory(struct DrawingPanelArgs *args) {
    segment_list_node_t *cur = args->segments_list_head;
    while (cur != NULL) {
        segment_list_node_t *prev = cur;
        cur = cur->next;
        UnloadRenderTexture(prev->saved_image);
        free(prev);
    }
    args->segments_list_cur = NULL;
    args->segments_list_head = NULL;
    args->segments_queue_size = 0;
}

void DrawingPanelUndo(struct DrawingPanelArgs *args) {
    if (args->segments_list_cur == NULL || args->segments_list_cur->prev == NULL) {
        return;
    }

    args->segments_list_cur = args->segments_list_cur->prev;
    BeginTextureMode(args->drawn_image);
    {
        DrawTexture(args->segments_list_cur->saved_image.texture, 0, 0, WHITE);
    }
    EndTextureMode();

    args->updated = true;
}

void DrawingPanelRedo(struct DrawingPanelArgs *args) {
    if (args->segments_list_cur == NULL || args->segments_list_cur->next == NULL) {
        return;
    }

    args->segments_list_cur = args->segments_list_cur->next;
    BeginTextureMode(args->drawn_image);
    {
        DrawTexture(args->segments_list_cur->saved_image.texture, 0, 0, WHITE);
    }
    EndTextureMode();

    args->updated = true;
}

void DrawingPanelAdd(struct DrawingPanelArgs *args) {
    segment_list_node_t *new_node = malloc(sizeof(segment_list_node_t));
    new_node->prev = NULL;
    new_node->next = NULL;
    new_node->saved_image = LoadRenderTexture(args->drawn_image.texture.width, args->drawn_image.texture.height);
    BeginTextureMode(new_node->saved_image);
    {
        ClearBackground(WHITE);
        DrawTexture(args->drawn_image.texture, 0, 0, WHITE);
    }
    EndTextureMode();

    if (args->segments_list_cur == NULL) {
        args->segments_list_cur = new_node;
        args->segments_list_head = new_node;
        args->segments_queue_size = 1;
    } else {
        segment_list_node_t *delete = args->segments_list_cur->next;
        while (delete != NULL) {
            segment_list_node_t *prev = delete;
            delete = delete->next;
            args->segments_queue_size--;
            
            UnloadRenderTexture(prev->saved_image);
            free(prev);
        }

        new_node->prev = args->segments_list_cur;
        args->segments_list_cur->next = new_node;
        args->segments_queue_size++;
        args->segments_list_cur = new_node;
    }

    args->updated = true;
}

void DrawingPanelClear(struct DrawingPanelArgs *args) {
    BeginTextureMode(args->drawn_image);
    {
        ClearBackground(WHITE);
    }
    EndTextureMode();

    DrawingPanelAdd(args);

    args->updated = true;
}


// GOALS
// Set Brush Mode
// Save image
// Load image
// Show Buffered view that will be sent to the model
void GuiDrawingPanelPopup(struct DrawingPanelArgs *args) {
    if (!args->isOpen) {
        return;
    }
    
    // darken background
    DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, ColorAlpha(BLACK, 0.5));

    // drawing panel
    RenderTexture2D draw_image = args->drawn_image;
    
    Rectangle draw_window_rec = {
        .x = 50,
        .y = 50,
        .width = SCREEN_WIDTH - 2 * 50,
        .height = SCREEN_HEIGHT - 2 * 50,
    };    

    
    Rectangle draw_panel_rec = {
        .x = draw_window_rec.x + draw_window_rec.width/2 - draw_image.texture.width/2 - 3,
        .y = draw_window_rec.y + draw_window_rec.height/2 - draw_image.texture.height/2 - 3,
        .width = draw_image.texture.width + 6,
        .height = draw_image.texture.height + 6,
    };

    args->isOpen = !GuiWindowBox(draw_window_rec, "Drawing Panel");
    
    DrawRectangleRec(draw_panel_rec, WHITE);
    DrawRectangleRoundedLines(draw_panel_rec, .02, 10, 5, BLACK);


    // Brush color picker
    Rectangle color_picker_rec = {
        .x = draw_panel_rec.x + draw_panel_rec.width + 40,
        .y = draw_window_rec.y + draw_window_rec.height/2 - 120/2,
        .width = 120,
        .height = 120,
    };
    GuiColorPickerHSV(color_picker_rec, "Brush Color", &args->brush_color);

    Rectangle brush_color_rec = {
        .x = color_picker_rec.x,
        .y = color_picker_rec.y - 40,
        .width = 30,
        .height = 30, 
    };

    // Brush size picker
    Rectangle brush_size_picker_rec = {
        .x = draw_panel_rec.x + draw_panel_rec.width + 30,
        .y = brush_color_rec.y - 40,
        .width = 100,
        .height = 30,
    };

    float max_brush_size = 10;
    float min_brush_size = 0.5;
    GuiSliderBar(brush_size_picker_rec, TextFormat("%.2f", min_brush_size), TextFormat("%.2f", max_brush_size), &args->brush_size, min_brush_size, max_brush_size);
    DrawText("Brush Size", brush_size_picker_rec.x, brush_size_picker_rec.y - 20, 12, BLACK);

    Vector2 brush_size_display_rec = {
        .x = brush_size_picker_rec.x + brush_size_picker_rec.width + 45,
        .y = brush_size_picker_rec.y + brush_size_picker_rec.height/2
    };
    DrawRectangleRoundedLines((Rectangle) {.x = brush_size_display_rec.x - max_brush_size, .y = brush_size_display_rec.y - max_brush_size, .width = max_brush_size*2, .height = max_brush_size*2}, 
            0.02, 4, 2, BLACK);
    DrawCircleV(brush_size_display_rec , args->brush_size, ColorFromHSV(args->brush_color.x, args->brush_color.y, args->brush_color.z));


    // Drawing Panel Options
    Rectangle draw_panel_clear_rec = {
        .x = draw_panel_rec.x,
        .y = draw_panel_rec.y + draw_panel_rec.height + 10,
        .width = 40,
        .height = 20,
    };
    if (GuiButton(draw_panel_clear_rec, "Clear")) {
        DrawingPanelClear(args);
    }

    Rectangle draw_panel_undo_rec = {
        .x = draw_panel_clear_rec.x + draw_panel_clear_rec.width + 10,
        .y = draw_panel_clear_rec.y,
        .width = 40,
        .height = 20,
    };
    if (GuiButton(draw_panel_undo_rec, "Undo")) {
        DrawingPanelUndo(args);
    }

    Rectangle draw_panel_redo_rec = {
        .x = draw_panel_undo_rec.x + draw_panel_undo_rec.width + 10,
        .y = draw_panel_undo_rec.y,
        .width = 40,
        .height = 20,
    };
    if (GuiButton(draw_panel_redo_rec, "Redo")) {
        DrawingPanelRedo(args);
    }


    bool is_draw = IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsGestureDetected(GESTURE_DRAG);
    bool is_erase = IsMouseButtonDown(MOUSE_BUTTON_RIGHT);
    bool just_off_panel = args->is_dragged && !CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);
    if (((is_draw || is_erase) && CheckCollisionPointRec(GetMousePosition(), draw_panel_rec)) || just_off_panel) {
        Color draw_color = ColorFromHSV(args->brush_color.x, args->brush_color.y, args->brush_color.z);
        if (is_erase) {
            draw_color = WHITE;
        }

        Vector2 size = {
            .x = args->brush_size,
            .y = args->brush_size,
        };
        Vector2 pos = GetMousePosition();
        pos.x -= size.x / 2.0;
        pos.y -= size.y / 2.0;

        pos.x -= draw_panel_rec.x;
        pos.y -= draw_panel_rec.y;

        pos.y = draw_image.texture.height - pos.y; // texture drawing flips vertically because of opengl or smthn

        BeginTextureMode(draw_image);
        {
            
            DrawCircleV(pos, args->brush_size, draw_color);
            
            // to connect points since the refresh rate is 60 fps which causes gaps
            if (args->prev_draw_pos.x != -1 && args->prev_draw_pos.y != -1) {
                DrawLineEx(args->prev_draw_pos, pos, args->brush_size * 2, draw_color);
                DrawCircleV(args->prev_draw_pos, args->brush_size, draw_color);
            }

            // TODO try to fix
            // when dragging mouse past border of draw panel, there is a small gap between the border and the last drawn
            // position. Perhaps track whenever the mouse drag JUST leaves the area, and still draw that line segment
            
            args->prev_draw_pos = pos;
        }
        EndTextureMode();

        args->is_drawing = true;
        args->updated = true;
    } else {
        // is not drawing
        args->prev_draw_pos = (Vector2) {.x = GetMousePosition().x - (args->brush_size/2) - draw_panel_rec.x, 
                .y = draw_image.texture.height - (GetMousePosition().y - (args->brush_size/2) - draw_panel_rec.y)};
        

        // check if just stopped drawing, register segment
        if (args->is_drawing) {
            printf("DETECTED Line Segment !!\n");
            DrawingPanelAdd(args);
        }
        args->is_drawing = false;     
    }

    args->is_dragged = IsGestureDetected(GESTURE_DRAG) && CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);

    DrawTexture(draw_image.texture, draw_panel_rec.x, draw_panel_rec.y, WHITE);


    // Draw what the model will see
    Rectangle model_input_rec = {
        .x = draw_panel_rec.x - 10 - 50,
        .y = draw_panel_rec.y,
        .width = 50,
        .height = 50
    };

    RenderTexture2D model_image = args->model_input_image;

    if (args->updated) {
        // scale down drawn image
        args->updated = false;
        int scale_ceil = (draw_image.texture.height + model_image.texture.height - 1) / model_image.texture.height;
        int scale_floor = draw_image.texture.height / model_image.texture.height;
        Image draw_image_converted = LoadImageFromTexture(draw_image.texture);
        
        BeginTextureMode(model_image);
        {
            for (int r = 0; r < model_image.texture.height; r++) {
                for (int c = 0; c < model_image.texture.width; c++) {
                    int actualR = r * scale_floor;
                    int actualC = c * scale_floor;

                    int radius = (scale_ceil + 1) / 2 - 1;
                    double sum = 0;
                    double total_distance = 0;
                    int n_pixels = 0;
                    for (int d_r = -radius; d_r <= radius; d_r++) {
                        for (int d_c = -radius; d_c <= radius; d_c++) {
                            int new_r = actualR + d_r;
                            int new_c = actualC + d_c;
                            if (new_r < 0 || new_c < 0 || new_r >= draw_image.texture.height || new_c >= draw_image.texture.width) {
                                continue;
                            }
                            double distance = sqrt(d_r * d_r + d_c * d_c);
                            total_distance += distance;
                            n_pixels++;
                            Color pixel = GetImageColor(draw_image_converted, new_c, new_r);
                            
                            Color new_pixel = gray_scale(pixel.r, pixel.g, pixel.b);
                            sum += new_pixel.r * distance;
                        }
                    }

                    double new_color = sum / total_distance;
                    
                    DrawPixel(c, model_image.texture.height - r, (Color) {.a = 255, .r = new_color, .g = new_color, .b = new_color});

                    assert(r * args->buffer_width + c < args->buffer_width * args->buffer_height);
                    args->output_buffer[r * args->buffer_width + c] = new_color / 255.0;
                }
            }
        }
        EndTextureMode();

        UnloadImage(draw_image_converted);
    }

    DrawTexturePro(model_image.texture, (Rectangle) {.x = 0, .y = 0, .width = model_image.texture.width, .height = model_image.texture.height},
            model_input_rec, (Vector2) {.x = 0, .y = 0}, 0, WHITE);


    // TODO draw the model's output
    
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
    int layer_height = nodes.r * (NODE_GAP + 2 * NODE_RADIUS) - NODE_GAP;
    int layer_start_y = model_start_y + (MODEL_HEIGHT - layer_height) / 2;
    int layer_name_y = layer_start_y + layer_height + LAYER_NAME_OFFSET_Y + LAYER_DISPLAY_FONTSIZE / 2;
    int layer_function_name_y = layer_name_y - LAYER_DISPLAY_FONTSIZE / 2 + LAYER_DISPLAY_FONTSIZE * 2;
    int layer_x = model_start_x + layer_index * (LAYER_GAP + 2 * NODE_RADIUS) + NODE_RADIUS;
    int layer_info_font_size = LAYER_DISPLAY_FONTSIZE - 4;
    DrawOutlinedCenteredText(get_layer_name(layer), layer_x, layer_name_y, LAYER_DISPLAY_FONTSIZE, BLACK, 0, BLACK);
    if (layer->type == ACTIVATION) {
        DrawOutlinedCenteredText(get_activation_function_name(&layer->layer.activation), layer_x, 
                layer_function_name_y, layer_info_font_size, BLACK, 0, BLACK);
    } else if (layer->type == OUTPUT) {
        DrawOutlinedCenteredText(get_output_function_name(&layer->layer.output), layer_x, 
                layer_function_name_y, layer_info_font_size, BLACK, 0, BLACK);
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
    Color color = GRAY;
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

        // TODO dont add drawing panel if model does not accept drawings as input
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

