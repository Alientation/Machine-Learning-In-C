#pragma once
#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <model/model.h>

typedef struct VisualizerArgument {
    neural_network_model_t *model;
    char *model_name;
    bool allow_drawing_panel_as_model_input;

    char** (*label_guess)(mymatrix_t model_guess);
} visualizer_argument_t;


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