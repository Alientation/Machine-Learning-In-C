#include <app/dataset.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define IMAGE_DATASET_HEADER 0xFF0000DDLU

long read_bytes(unsigned char *data, int start_byte, int bytes) {
    long value = 0;
    for (int i = start_byte + bytes - 1; i >= start_byte; i--) {
        value <<= 8;
        value += data[i];
    }
    return value;
}

void write_bytes(unsigned char *data, int start_byte, unsigned long value, int bytes) {
    for (int i = start_byte; i < start_byte + bytes; i++) {
        data[i] = value & (0xFF);
        value >>= 8;
    }
}

image_dataset_visualizer_t LoadImageDataSetVisualizer(dataset_t dataset) {
    image_dataset_visualizer_t dataset_vis = {
        .dataset = dataset,
        .left_image_index = 0
    };

    struct ImageListNode *cur = dataset.data.image_dataset.image_list_head;
    for (int i = 0; i < NUMBER_DISPLAYED_IMAGES && i < dataset.data.image_dataset.count; i++) {
        dataset_vis.displayed_images[i] = LoadTextureFromImage(cur->image);
        cur = cur->next;
    }
}

void UnloadImageDataSetVisualizer(image_dataset_visualizer_t dataset_vis) {
    for (int i = 0; i < NUMBER_DISPLAYED_IMAGES && i < dataset_vis.dataset.data.image_dataset.count; i++) {
        UnloadTexture(dataset_vis.displayed_images[i]);
    }
}



dataset_t ConstructImageDataSet(const char* file_path, int width, int height) {
    return (dataset_t) {
        .type = DATASET_IMAGES,
        .file_path = strdup(file_path),
        .data.image_dataset.count = 0,
        .data.image_dataset.image_list_head = NULL,
        .data.image_dataset.uniform_width = width,
        .data.image_dataset.uniform_height = height,
    };
}

dataset_t LoadDataSet(const char* file_path) {
    int file_size = 0;
    unsigned char* data = LoadFileData(file_path, &file_size);
    
    long header = read_bytes(data, 0, 8);
    if (header == IMAGE_DATASET_HEADER) {
        int width = (int) read_bytes(data, 8, 4);
        int height = (int) read_bytes(data, 12, 4);
        long num_images = read_bytes(data, 16, 8);

        dataset_t dataset = ConstructImageDataSet(file_path, width, height);

        int byte = 24;
        for (int i = 0; i < num_images; i++) {
            long image_bytes = read_bytes(data, byte, 8);
            byte += 8;

            DataSetAddImage(dataset, LoadImageFromMemory(".png", data + byte, image_bytes));
            byte += image_bytes;
        }
        return dataset;
    }

    printf("Failed to load dataset from %s with header %llx\n", file_path, header);
    return (dataset_t) {.type = DATASET_INVALID};
}

void WriteDataSet(dataset_t dataset) {

    // HEADER:  8 BYTES
    // IMAGE_WIDTH: 4 BYTES
    // IMAGE_HEIGHT: 4 BYTES
    // NUM_IMAGES: 8 BYTES
    // NUM_IMAGE_BYTES: 8 BYTES
    // IMAGE_DATA: NUM_IMAGE_BYTES BYTES
    if (dataset.type == DATASET_IMAGES) {
        struct DataSetData_Images images = dataset.data.image_dataset;
        int data_size = 24; // HEADER + NUM_IMAGES
        unsigned char* images_data[images.count];
        int images_size[images.count];

        struct ImageListNode *cur_image = images.image_list_head;        
        for (int i = 0; i < images.count; i++) {
            images_data[i] = ExportImageToMemory(images.image_list_head->image, ".png", &images_size[i]);
            data_size += 8 + images_size[i]; // NUM_IMAGE_BYTES + IMAGE_DATA
            cur_image = cur_image->next;
        }

        unsigned char *data = malloc(sizeof(char) * data_size);
        write_bytes(data, 0, IMAGE_DATASET_HEADER, 8);
        write_bytes(data, 8, images.uniform_width, 4);
        write_bytes(data, 12, images.uniform_height, 4);
        write_bytes(data, 16, images.count, 8);
        int byte_index = 24;
        for (int i = 0; i < images.count; i++) {
            write_bytes(data, byte_index, images_size[i], 8);
            byte_index += 8;
            for (int j = i; j < images_size[i]; j++) {
                data[j + byte_index] = images_data[i][j];
            }
            byte_index += images_size[i];
        }

        SaveFileData(dataset.file_path, data, data_size);
    } else {
        assert(0);
    }
}

void DataSetAddImage(dataset_t dataset, Image image) {
    assert(dataset.type == DATASET_IMAGES);
    
    struct DataSetData_Images images = dataset.data.image_dataset;
    if (images.image_list_head == NULL) {
        images.image_list_head = malloc(sizeof(struct ImageListNode));
        images.image_list_head->image = image;
        images.image_list_head->next = NULL;
        images.image_list_head->prev = NULL;
    } else {
        images.image_list_head->next = malloc(sizeof(struct ImageListNode));
        images.image_list_head->next->image = image;
        images.image_list_head->next->prev = images.image_list_head;
        images.image_list_head = images.image_list_head->next;
    }
    images.count++;
}

void DataSetRemoveImage(dataset_t dataset, int index) {
    DataSetRemoveImages(dataset, index, index+1);
}

void DataSetRemoveImages(dataset_t dataset, int from_index, int to_index) {
    assert(from_index >= 0);
    assert(from_index <= to_index);
    assert(to_index <= dataset.data.image_dataset.count);

    struct DataSetData_Images images = dataset.data.image_dataset;
    
    struct ImageListNode *remove_start = images.image_list_head;
    for (int i = 0; i < from_index; i++) {
        remove_start = remove_start->next;
    }

    struct ImageListNode *remove_end = remove_start;
    for (int i = from_index; i < to_index - 1; i++) {
        UnloadImage(remove_end->image);
        remove_end = remove_end->next;
    }
    UnloadImage(remove_end->image);

    remove_start->prev->next = remove_end->next;
    if (remove_end->next) {
        remove_end->prev = remove_start->prev;
    }

    images.count -= to_index - from_index;
}


void UnloadDataSet(dataset_t dataset) {
    free(dataset.file_path);
    switch (dataset.type) {
        case DATASET_IMAGES:
            struct DataSetData_Images images = dataset.data.image_dataset;
            struct ImageListNode *cur_image = images.image_list_head;
            for (int i = 0; i < images.count; i++) {
                UnloadImage(cur_image->image);
                struct ImageListNode *next = cur_image->next;
                free(cur_image);
                cur_image = next;
            }
            break;
        default:
            assert(0);
    }
}