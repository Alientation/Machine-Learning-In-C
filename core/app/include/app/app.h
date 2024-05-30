#pragma once
#ifndef APP_H
#define APP_H

#include <model/model.h>

training_info_t nn_binary_digit_recognizer(neural_network_model_t *model);
training_info_t nn_AND(neural_network_model_t *model);
training_info_t nn_XOR(neural_network_model_t *model);


#endif // APP_H