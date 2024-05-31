#pragma once
#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <model/model.h>

/**
 * thread for the window, supplied a neural_network_model_t object pointer
 */
void* window_run(void *vargp);

/**
 * Keeps the window open for a specific number of seconds. If num_seconds is 0, then the window is kept open
 * for a very, very long time (UINT32_MAX seconds)
 */
void window_keep_open(neural_network_model_t *model, unsigned int num_seconds);


#endif // VISUALIZER_H