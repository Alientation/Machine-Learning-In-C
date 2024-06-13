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
            unsigned long long count;
            unsigned long long num_labels;
            char** label_names;

            unsigned int uniform_width;
            unsigned int uniform_height;
            struct ImageListNode {
                Image image;
                long long label;
                struct ImageListNode *next;
                struct ImageListNode *prev;
            };
            struct ImageListNode *image_list_head;
            struct ImageListNode *image_list_tail;
        } image_dataset;
    } data;

    char* file_path;
} dataset_t;

#define NUMBER_DISPLAYED_IMAGES 5
typedef struct ImageDataSetVisualizer {
    dataset_t *dataset;
    int left_image_index;
    struct ImageListNode *left_image_node;
    int number_displayed;
    Texture2D displayed_images[NUMBER_DISPLAYED_IMAGES];
    struct ImageListNode *displayed_images_nodes[NUMBER_DISPLAYED_IMAGES];
    
} image_dataset_visualizer_t;

image_dataset_visualizer_t LoadImageDataSetVisualizer(dataset_t *dataset);
void UpdateImageDataSetVisualizer(image_dataset_visualizer_t *dataset_vis);
void UnloadImageDataSetVisualizer(image_dataset_visualizer_t dataset_vis);

void MoveDisplayImageDataSetVisualizer(image_dataset_visualizer_t *dataset_vis, int move_count);
void SetDisplayImageDataSetVisualizer(image_dataset_visualizer_t *dataset_vis, int pos);


dataset_t ConstructImageDataSet(const char* file_path, int width, int height, int num_labels, const char** label_names);
dataset_t LoadDataSet(const char* file_path);
void WriteDataSet(dataset_t dataset);

void DataSetAddImage(dataset_t *dataset, Image image, long label);
void DataSetRemoveImage(dataset_t *dataset, int index);
void DataSetRemoveImages(dataset_t *dataset, int from_index, int to_index);

void UnloadDataSet(dataset_t dataset);


#endif // DATASET_H