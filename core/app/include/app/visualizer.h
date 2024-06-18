#pragma once
#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <model/model.h>
#include <app/dataset.h>

#include <raylib.h>
#include <math.h>

static const int SCREEN_WIDTH = 1600;
static const int SCREEN_HEIGHT = 900;
static const int MODEL_WIDTH = 1560;
static const int MODEL_HEIGHT = 700;
static const int MODEL_X = 20;
static const int MODEL_Y = 20;

#define SCALING_FACTOR 0.2

// how many digits (including '-' sign for negative numbers)
// that can be displayed for weights and nodes
static const int WEIGHT_DISPLAY_PRECISION = 9; // displayed in a tooltip
static const int NODE_DISPLAY_PRECISION = ceil(7 * SCALING_FACTOR); // displayed inside the node, so has to be scaled accordingly

// number of dotted lines connecting nodes in non-dense layers
static const int WEIGHT_DOTTED_LINES =  11;
// limit how many weights can be drawn per layer, skipping some weights in between to avoid clutter
static const int MAX_WEIGHTS_DRAWN_PER_LAYER = 150;

// fontsize of the value of node
static const int NODE_DISPLAY_FONTSIZE = 15 * SCALING_FACTOR;
// if node is too small, don't bother drawing the value
static const int MIN_NODE_PRECISION_FOR_DISPLAY = 2;
static const int NODE_RADIUS = 35 * SCALING_FACTOR;
static const int NODE_GAP = 20 * SCALING_FACTOR; // gap between consecutive nodes vertically and horizontally between stacks

// Type of layer font size
static const int LAYER_DISPLAY_FONTSIZE = 12;
// margin from bottom of layer's nodes to the layer's name
static const int LAYER_NAME_OFFSET_Y = 20;
// gap between consecutive layers
static const int LAYER_GAP = 65;

// how many nodes can be stacked vertically before having multiple stacks of nodes 
// side by side in a layer
static const int INPUT_LAYER_NODES_HEIGHT = 28; // should prolly be passed into the visualizer as an argument
static const int HIDDEN_LAYER_NODES_HEIGHT = 32;

// displayed when hovering over input node in 'playground' mode to allow editing inputs manually
static const int MIN_NODE_RADIUS_FOR_SLIDER_BAR = 20;

// how far the mouse can be from target object to display a tooltip
static const int MOUSE_HOVER_DISTANCE_TO_NODE = NODE_RADIUS + NODE_GAP/2;
static const int MOUSE_HOVER_DISTANCE_TO_WEIGHT = ceil(14 * SCALING_FACTOR);

// Tooltip information
static const int TOOLTIP_BORDER = 10;
static const int TOOLTIP_HEIGHT = 30;
static const int TOOLTIP_WIDTH = 100;
static const int TOOLTIP_FONTSIZE = 16;

// when 'playground' mode is enabled, using the mousewheel while hovering over 
// weights and input nodes will alter the value accordingly
static const float WEIGHT_VALUE_MOUSEWHEEL_SCALE = 0.05;
static const float NODE_VALUE_MOUSEWHEEL_SCALE = 0.625; // TODO this is not used since the display tooltip function does not take in a custom mousewheel scale

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

// max characters that can be written in a tooltip
#define TOOLTIP_BUFFER_SIZE 100
static const int FILE_NAME_BUFFER_SIZE = 40;
static const int NUMBER_INPUT_BUFFER_SIZE = 10;

typedef struct LabelGuesses {
    float *confidences;
} label_guesses_t;

typedef struct VisualizerArgument {
    neural_network_model_t *model;
    training_info_t *training_info;
    char *model_name;
    bool allow_drawing_panel_as_model_input;
    
    int num_labels;
    const char **output_labels; // one hot encoded
    char *default_dataset_directory; // input data
} visualizer_argument_t;

typedef struct SegmentListNode {
    RenderTexture2D saved_image;
    struct SegmentListNode *next;
    struct SegmentListNode *prev;
} segment_list_node_t;

// TODO MOVE MOST TO SEPARATE DRAWING PANEL STATE STRUCT TO DECLUTTER
typedef struct DrawingPanelArgs {
    visualizer_argument_t *vis_args;

    bool is_open;           // Is drawing panel window open
    float brush_size;       // radius of drawn circle
    Vector3 brush_color;    // drawing color stored in HSV

    Vector2 prev_draw_pos;          // position of the last circle drawn, used to connect the two positions to smooth the drawing out
    bool is_dragged;                // whether the mouse is being dragged, when the mouse just recently left the drawing window continue to draw the line segment out
    bool is_drawing;                // whether stuff was drawn in the last frame to detect when drawing stops and saving the current state to the history buffer
    RenderTexture2D draw_texture;   // stores the drawing

    // TODO perhaps limit the size of history stored
    segment_list_node_t *segments_list_head;    // history buffer to allow undoing/redoing, stored as a linked list
    segment_list_node_t *segments_list_cur;     // points to the current state in the list
    int segments_list_size;

    // TODO SAVE THESE IN A SEPARATE STRUCT FOR THE SAVE WINDOW
    // save popup
    bool is_save_popup_open;    
    bool is_dataset_viewer_open;
    
    // dataset picker
    char *dataset_directory;    // the relative path to the directory storing the datasets
    int sel_dataset_index;      // index of the file in the directory, -1 if nothing is selected
    int sel_label_index;        // index of the label for the current image, -1 if nothing is selected
    dataset_t current_dataset;  // information about the currently selected dataset
    image_dataset_visualizer_t img_dataset_vis; // visualizer for the dataset
    int dataset_list_scroll_index; //
    int sel_dataset_image_index;

    // convert to training data options
    int num_transformations;        // number of times each of the below transformations are applied on a single dataset image, ie there will be num_transformations * num_images extra examples in training/testing dataset
    bool is_transformations_active;
    float max_rotation_degree;      // rotate the image clockwise or counterclockwise a random degree, the maximum magnitude of which is specified here
    bool is_rotations_active;
    float max_translations_pixels_x;    // maximum movement horizontally // TODO translations should be limited to not cause the drawing to off the image bounds
    bool is_translations_x_active;
    float max_translations_pixels_y;    // maximum movement vertically
    bool is_translations_y_active;
    float max_artifacts;            // percentage of the image that will have a random artifact (ie, a random pixel shaded in with a random value)
    bool is_artifacts_active;

    // create a new dataset
    char *add_dataset_file_name;
    int add_dataset_type;

    // create a new image dataset
    bool images_dataset_width_option_active;
    char *images_dataset_width_input;
    bool images_dataset_height_option_active;
    char *images_dataset_height_input;
    bool is_editing_dataset_file_name;

    int num_labels;
    const char** label_names;

    bool updated;       // whether the drawing was updated
    int update_frames;  // how many frames before each update
    int cur_frames;     // frames since last update
    RenderTexture2D input_texture;  // scaled down texture of the drawn image, to be input into the model
    bool gray_scale;    // whether the model takes in a gray scaled image
    int buffer_width;   // width dim of model's input
    int buffer_height;  // height dim of model's input
    float *output_buffer; // store the model's input in a linear array
} drawing_panel_args_t; 

typedef struct VisualizerState {
    visualizer_argument_t vis_args;
    drawing_panel_args_t draw_args;
    bool is_training;
    bool is_testing;
    bool playground_state;

    // draw info
    Vector2 **node_positions;
    RenderTexture2D node_texture;
    RenderTexture2D node_outline_texture;

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