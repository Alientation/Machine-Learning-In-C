#pragma once
#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <model/model.h>

#include <raylib.h>
#include <math.h>

static const int SCREEN_WIDTH = 1200;
static const int SCREEN_HEIGHT = 700;
static const int MODEL_WIDTH = 1160;
static const int MODEL_HEIGHT = 450;
static const int MODEL_X = 20;
static const int MODEL_Y = 20;

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

static const float TOOLTIP_WEIGHT_VALUE_SCALE = 0.05;
#define TOOLTIP_BUFFER_SIZE 100

typedef struct VisualizerArgument {
    neural_network_model_t *model;
    training_info_t training_info;
    char *model_name;
    bool allow_drawing_panel_as_model_input;

    char** (*label_guess)(mymatrix_t model_guess);
} visualizer_argument_t;

typedef struct SegmentListNode {
    RenderTexture2D saved_image;
    struct SegmentListNode *next;
    struct SegmentListNode *prev;
} segment_list_node_t;

typedef struct DrawingPanelArgs {
    bool is_open;
    float brush_size;
    Vector3 brush_color; // in HSV

    Vector2 prev_draw_pos;
    bool is_dragged;
    bool is_drawing;
    RenderTexture2D draw_texture;

    segment_list_node_t *segments_list_head;
    segment_list_node_t *segments_list_cur;
    int segments_list_size;

    // scaled down
    bool updated;
    int update_frames; // how many frames before each update
    int cur_frames;
    RenderTexture2D input_texture;
    bool gray_scale;
    int buffer_width;
    int buffer_height;
    float *output_buffer;
} drawing_panel_args_t; 

typedef struct VisualizerState {
    visualizer_argument_t vis_args;
    drawing_panel_args_t draw_args;
    bool is_training;
    bool is_testing;
    bool playground_state;

    bool show_tooltip;
    char tooltip_msg[TOOLTIP_BUFFER_SIZE];
    float tooltip_priority;
    float *tooltip_weight_value;

    bool is_window_open;
} visualizer_state_t;

/**
 * thread for the window, supplied a VisualizerArgument object pointer
 */
void* window_run(void *vargp);

/**
 * Keeps the window open for a specific number of seconds. If num_seconds is 0, then the window is kept open
 * for a very, very long time (UINT32_MAX seconds)
 */
void window_keep_open(neural_network_model_t *model, unsigned int num_seconds);



#endif // VISUALIZER_H