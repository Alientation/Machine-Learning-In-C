#pragma once
#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <model/model.h>

void* window_run(void *vargp);
void DrawWindow(neural_network_model_t *model);
void window_close();
void window_keep_open(neural_network_model_t *model, unsigned int num_seconds);


#endif // VISUALIZER_H