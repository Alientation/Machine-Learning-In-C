#pragma once
#ifndef DATASET_H
#define DATASET_H

#include <raylib.h>

typedef struct DataSet {
    enum DataSetType {
        DATASET_INVALID,
        DATASET_IMAGES,
    } type;

    union DataSetData {
        struct DataSetData_Images {
            int count;

            int uniform_width;
            int uniform_height;
            struct ImageListNode {
                Image image;
                struct ImageListNode *next;
                struct ImageListNode *prev;
            };
            struct ImageListNode *image_list_head;
        } image_dataset;
    } data;

    char* file_path;
} dataset_t;

#define NUMBER_DISPLAYED_IMAGES 5
typedef struct ImageDataSetVisualizer {
    dataset_t dataset;
    int left_image_index;
    Texture2D displayed_images[NUMBER_DISPLAYED_IMAGES];
    
} image_dataset_visualizer_t;

image_dataset_visualizer_t LoadImageDataSetVisualizer(dataset_t dataset);
void UnloadImageDataSetVisualizer(image_dataset_visualizer_t dataset_vis);


dataset_t ConstructImageDataSet(const char* file_path, int width, int height);
dataset_t LoadDataSet(const char* file_path);
void WriteDataSet(dataset_t dataset);

void DataSetAddImage(dataset_t dataset, Image image);
void DataSetRemoveImage(dataset_t dataset, int index);
void DataSetRemoveImages(dataset_t dataset, int from_index, int to_index);

void UnloadDataSet(dataset_t dataset);

#endif // DATASET_H