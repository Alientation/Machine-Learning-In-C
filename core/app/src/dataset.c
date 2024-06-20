#include <app/dataset.h>
#include <util/math.h>

#define PROFILER_DISABLE_FUNCTION_RETURN
#include <util/profiler.h>
#include <app/visutil.h>

#include <pthread.h>
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

image_dataset_visualizer_t LoadImageDataSetVisualizer(dataset_t *dataset) {
    assert(dataset->type == DATASET_IMAGES);
    
    image_dataset_visualizer_t dataset_vis = {
        .dataset = dataset,
        .left_image_index = 0,
        .left_image_node = dataset->data.image_dataset.image_list_head,
    };
    struct DataSetData_Images images = dataset->data.image_dataset;

    struct ImageListNode *cur = dataset->data.image_dataset.image_list_head;
    for (int i = 0; i < NUMBER_DISPLAYED_IMAGES && i < dataset->data.image_dataset.count; i++) {
        dataset_vis.displayed_images[i] = LoadTextureFromImage(cur->image);
        dataset_vis.displayed_images_nodes[i] = cur;
        cur = cur->next;
    }

    dataset_vis.number_displayed = NUMBER_DISPLAYED_IMAGES >= dataset->data.image_dataset.count ? dataset->data.image_dataset.count : NUMBER_DISPLAYED_IMAGES;

    Image img = GenImageColor(images.uniform_width, images.uniform_height, WHITE);
    for (int i = dataset->data.image_dataset.count; i < NUMBER_DISPLAYED_IMAGES; i++) {
        dataset_vis.displayed_images[i] = LoadTextureFromImage(img);
        dataset_vis.displayed_images_nodes[i] = NULL;
    }
    UnloadImage(img);
    return dataset_vis;
}

void UpdateImageDataSetVisualizer(image_dataset_visualizer_t *dataset_vis) {
    for (int i = 0; i < NUMBER_DISPLAYED_IMAGES; i++) {
        UnloadTexture(dataset_vis->displayed_images[i]);
    }

    int num_images = dataset_vis->dataset->data.image_dataset.count;

    dataset_vis->left_image_node = dataset_vis->dataset->data.image_dataset.image_list_head;
    dataset_vis->number_displayed = num_images > NUMBER_DISPLAYED_IMAGES ? NUMBER_DISPLAYED_IMAGES : num_images;
    printf("updating dataset vis with %d images showcasing images %d-%d\n", num_images, dataset_vis->left_image_index, dataset_vis->left_image_index + dataset_vis->number_displayed);
    for (int i = 0; i < dataset_vis->left_image_index; i++) {
        dataset_vis->left_image_node = dataset_vis->left_image_node->next;
    }
    
    struct ImageListNode *cur = dataset_vis->left_image_node;
    Image img = GenImageColor(dataset_vis->dataset->data.image_dataset.uniform_width, dataset_vis->dataset->data.image_dataset.uniform_height, WHITE);
    for (int i = 0; i < NUMBER_DISPLAYED_IMAGES; i++) {
        if (cur != NULL) {
            dataset_vis->displayed_images[i] = LoadTextureFromImage(cur->image);
            dataset_vis->displayed_images_nodes[i] = cur;
        } else {
            dataset_vis->displayed_images[i] = LoadTextureFromImage(img);
            dataset_vis->displayed_images_nodes[i] = NULL;
        }

        if (cur != NULL) {
            cur = cur->next;
        }
    }
    UnloadImage(img);
}

void UnloadImageDataSetVisualizer(image_dataset_visualizer_t dataset_vis) {
    for (int i = 0; i < NUMBER_DISPLAYED_IMAGES; i++) {
        UnloadTexture(dataset_vis.displayed_images[i]);
    }
}

// TODO OPTIMIZE THIS
void MoveDisplayImageDataSetVisualizer(image_dataset_visualizer_t *dataset_vis, int move_count) {
    int target = move_count + dataset_vis->left_image_index;
    if (target < 0) {
        target = 0;
    } else if (target > dataset_vis->dataset->data.image_dataset.count - NUMBER_DISPLAYED_IMAGES) {
        target = dataset_vis->dataset->data.image_dataset.count - NUMBER_DISPLAYED_IMAGES;
    }

    if (target == dataset_vis->left_image_index) {
        return; // no need to update anything
    }

    dataset_vis->left_image_index = target;
    UpdateImageDataSetVisualizer(dataset_vis);
}

void SetDisplayImageDataSetVisualizer(image_dataset_visualizer_t *dataset_vis, int pos) {
    MoveDisplayImageDataSetVisualizer(dataset_vis, pos - dataset_vis->left_image_index);
}


dataset_t ConstructImageDataSet(const char* file_path, int width, int height, int num_labels, const char** label_names) {
     dataset_t dataset = {
        .type = DATASET_IMAGES,
        .file_path = strdup(file_path),
        .data.image_dataset.count = 0,
        .data.image_dataset.image_list_head = NULL,
        .data.image_dataset.image_list_tail = NULL,
        .data.image_dataset.uniform_width = width,
        .data.image_dataset.uniform_height = height,
        .data.image_dataset.num_labels = num_labels,
        .data.image_dataset.label_names = malloc(sizeof(char*) * num_labels),
    };
    
    for (int i = 0; i < num_labels; i++) {
        dataset.data.image_dataset.label_names[i] = strdup(label_names[i]);
    }

    return dataset;
}

dataset_t LoadDataSet(const char* file_path) {
    int file_size = 0;
    unsigned char* data = LoadFileData(file_path, &file_size);
    
    long header = read_bytes(data, 0, 8);
    if (header == IMAGE_DATASET_HEADER) {
        int width = (int) read_bytes(data, 8, 4);
        int height = (int) read_bytes(data, 12, 4);
        long num_images = read_bytes(data, 16, 8);
        long num_labels = read_bytes(data, 24, 8);

        const char* labels[num_labels];
        int byte = 32;
        for (int i = 0; i < num_labels; i++) {
            int len = strlen(data + byte);
            labels[i] = data+byte;
            byte += len + 1; // including NULL TERMINATED CHARACTER
        }

        dataset_t dataset = ConstructImageDataSet(file_path, width, height, num_labels, labels);

        for (int i = 0; i < num_images; i++) {
            long image_bytes = read_bytes(data, byte, 8);
            byte += 8;
            long image_label = read_bytes(data, byte, 8);
            byte += 8;

            // printf("loading image: image_bytes=%d, label=%d\n", image_bytes, image_label);
            Image image = LoadImageFromMemory(".png", data + byte, image_bytes);
            // // TODO THIS IS TEMPORARY TO REDUCE SIZE OF OLD DATASETS
            // ImageColorGrayscale(&image);
            DataSetAddImage(&dataset, image, image_label);
            
            byte += image_bytes;
        }
        
        UnloadFileData(data);
        return dataset;
    }

    UnloadFileData(data);
    printf("Failed to load dataset from %s with header %llx\n", file_path, header);
    return (dataset_t) {.type = DATASET_INVALID};
}

void WriteDataSet(dataset_t dataset) {

    // HEADER:  8 BYTES
    // IMAGE_WIDTH: 4 BYTES
    // IMAGE_HEIGHT: 4 BYTES
    // NUM_IMAGES: 8 BYTES
    // NUM_LABELS: 8 BYTES
    // LABELS: X NULL TERMINATED STRINGS
    // NUM_IMAGE_BYTES: 8 BYTES
    // IMAGE_DATA: NUM_IMAGE_BYTES BYTES
    if (dataset.type == DATASET_IMAGES) {
        struct DataSetData_Images images = dataset.data.image_dataset;
        int data_size = 32; // HEADER + IMAGES DIMENSIONS + NUM_IMAGES + NUM_LABELS
        unsigned char* images_data[images.count];
        int images_size[images.count];

        for (int i = 0; i < images.num_labels; i++) {
            data_size += strlen(images.label_names[i]) + 1; // NULL TERMINATED CHARACTER
        }

        printf("saving dataset: number of images %d\n", images.count);

        struct ImageListNode *cur_image = images.image_list_head;
        for (int i = 0; i < images.count; i++) {
            images_data[i] = ExportImageToMemory(cur_image->image, ".png", &images_size[i]);
            data_size += 16 + images_size[i]; // NUM_IMAGE_BYTES + IMAGE_LABEL + IMAGE_DATA
            cur_image = cur_image->next;
        }

        unsigned char *data = malloc(sizeof(char) * data_size);
        write_bytes(data, 0, IMAGE_DATASET_HEADER, 8);
        write_bytes(data, 8, images.uniform_width, 4);
        write_bytes(data, 12, images.uniform_height, 4);
        write_bytes(data, 16, images.count, 8);
        write_bytes(data, 24, images.num_labels, 8);
        int byte_index = 32;

        for (int i = 0; i < images.num_labels; i++) {
            strcpy(data + byte_index, images.label_names[i]);
            byte_index += strlen(images.label_names[i]) + 1; // NULL TERMINATED CHARACTER
        }

        cur_image = images.image_list_head;
        for (int i = 0; i < images.count; i++) {
            write_bytes(data, byte_index, images_size[i], 8);
            byte_index += 8;
            write_bytes(data, byte_index, cur_image->label, 8);
            byte_index += 8;
            for (int j = 0; j < images_size[i]; j++) {
                data[j + byte_index] = images_data[i][j];
            }
            byte_index += images_size[i];
            cur_image = cur_image->next;
        }

        SaveFileData(dataset.file_path, data, data_size);

        free(data);
        for (int i = 0; i < images.count; i++) {
            free(images_data[i]);
        }
    } else {
        assert(0);
    }
}

void DataSetAddImage(dataset_t *dataset, Image image, long label) {
    assert(dataset->type == DATASET_IMAGES);
    
    struct DataSetData_Images *images = &dataset->data.image_dataset;
    if (images->image_list_head == NULL) {
        images->image_list_head = malloc(sizeof(struct ImageListNode));
        images->image_list_head->image = image;
        images->image_list_head->label = label;
        images->image_list_head->next = NULL;
        images->image_list_head->prev = NULL;

        images->image_list_tail = images->image_list_head;
    } else {
        images->image_list_tail->next = malloc(sizeof(struct ImageListNode));
        images->image_list_tail->next->image = image;
        images->image_list_tail->next->label = label;
        images->image_list_tail->next->prev = images->image_list_tail;
        images->image_list_tail = images->image_list_tail->next;
        images->image_list_tail->next = NULL;
    }
    images->count++;
}

void DataSetRemoveImage(dataset_t *dataset, int index) {
    DataSetRemoveImages(dataset, index, index+1);
}

void DataSetRemoveImages(dataset_t *dataset, int from_index, int to_index) {
    assert(from_index >= 0);
    assert(from_index <= to_index);
    assert(to_index <= dataset->data.image_dataset.count);

    struct DataSetData_Images *images = &dataset->data.image_dataset;
    
    struct ImageListNode *remove_start = images->image_list_head;
    for (int i = 0; i < from_index; i++) {
        remove_start = remove_start->next;
    } 

    struct ImageListNode *remove_end = remove_start;
    for (int i = from_index; i < to_index - 1; i++) {
        UnloadImage(remove_end->image);
        struct ImageListNode *delete = remove_end;
        remove_end = remove_end->next;
        free(delete);
    }
    UnloadImage(remove_end->image);

    // TODO implement linked list as circular with header node to clean it up
    if (from_index == 0 && to_index == dataset->data.image_dataset.count) {
        images->image_list_head = NULL;
        images->image_list_tail = NULL;
    } else if (from_index == 0) {
        images->image_list_head = remove_end->next;
    } else if (to_index == dataset->data.image_dataset.count) {
        images->image_list_tail = remove_start->prev;
    } else {
        remove_start->prev->next = remove_end->next;
        if (remove_end->next) {
            remove_end->prev = remove_start->prev;
        }
    }

    images->count -= to_index - from_index;
}

void convert_image_to_mymatrix(mymatrix_t* mymatrix, Image image) {
    for (int i = 0; i < image.height; i++) {
        for (int j = 0; j < image.width; j++) {
            // mymatrix->matrix[i * image.width + j][0] = 1 - GetImageColor(image, j, i).r / 256.0;
            assert(image.height * image.width == mymatrix->r);
            mymatrix->matrix[j * image.height + i][0] = 1 - GetImageColor(image, j, i).r / 256.0;
        }
    }
}

void one_hot_encode_matrix(mymatrix_t *mymatrix, int label) {
    mymatrix->matrix[label][0] = 1;
}

// TODO load on separate thread
void ImageDataSetConvertToTraining(training_info_t *training_info, dataset_t *dataset, int num_transformations, int max_rot_deg, int max_transl_x, int max_transl_y, float max_artifacts) {
    assert(dataset->type == DATASET_IMAGES);

    // TODO add option to select training/testing split, batch size, learning rate, target epochs, target accuracy
    const float train_test_split = 0.8;
    struct DataSetData_Images data = dataset->data.image_dataset;

    training_info->in_progress = false;
    training_info->train_size = 0;
    training_info->train_x = NULL;
    training_info->train_y = NULL;
    training_info->test_size = 0;
    training_info->test_x = NULL;
    training_info->test_y = NULL;

    training_info->batch_size = 1;
    training_info->learning_rate = 0.01;
    training_info->target_epochs = 20;
    training_info->target_accuracy = 1.5;
    
    const int num_examples_per_image = num_transformations + 1;
    
    training_info->train_size = data.count * num_examples_per_image * train_test_split;
    training_info->test_size = data.count * num_examples_per_image - training_info->train_size;

    training_info->train_x = malloc(training_info->train_size * sizeof(mymatrix_t));
    training_info->train_y = malloc(training_info->train_size * sizeof(mymatrix_t));
    training_info->test_x = malloc(training_info->test_size * sizeof(mymatrix_t));
    training_info->test_y = malloc(training_info->test_size * sizeof(mymatrix_t));

    struct Example {
        Image image;
        int label;
    };

    // struct Example shuffler[data.count * num_examples_per_image];
    struct Example *shuffler = malloc(sizeof(struct Example) * data.count * num_examples_per_image);
    const int input_size = data.uniform_width * data.uniform_width;
    const int output_size = data.num_labels;
    
    struct ImageListNode *cur = data.image_list_head;
    RenderTexture2D cur_tex = LoadRenderTexture(data.uniform_width, data.uniform_height);

    for (int i = 0; i < data.count; i++) {
        shuffler[i * num_examples_per_image] = (struct Example) {
            .image = ImageCopy(cur->image),
            .label = cur->label,
        };

        Image cur_image = ImageCopy(cur->image);
        ImageFlipVertical(&cur_image);

        Texture2D tex = LoadTextureFromImage(cur_image);

        for (int j = 1; j < num_examples_per_image; j++) {
            int rand_deg = GetRandomValue(-max_rot_deg, max_rot_deg);
            int rand_transl_x = GetRandomValue(-max_transl_x, max_transl_x);
            int rand_transl_y = GetRandomValue(-max_transl_y, max_transl_y);
            float rand_artifacts = GetRandomValue(0, max_artifacts * 100) / 100.0;

            int bufferx = 2 * abs(rand_transl_x);
            int buffery = 2 * abs(rand_transl_y);

            int x = -rand_transl_x;
            int y = -rand_transl_y;
            int width = data.uniform_width;
            int height = data.uniform_height;

            Rectangle img_src_r = {
                .x = x, .y = y, .width = width, .height = height
            };
            Rectangle img_dst_r = {
                .x = data.uniform_width/2, .y = data.uniform_height/2, .width = data.uniform_width, .height = data.uniform_height
            };

            BeginTextureMode(cur_tex);
            {
                ClearBackground(WHITE);
                DrawTexturePro(tex, img_src_r, img_dst_r, (Vector2) {data.uniform_width/2, data.uniform_height/2}, rand_deg, WHITE);
            }
            EndTextureMode();

            // TODO ADD RANDOM NOISE TO IMAGE

            shuffler[i * num_examples_per_image + j] = (struct Example) {
                .image = LoadImageFromTexture(cur_tex.texture),
                .label = cur->label,
            };
        }
        UnloadImage(cur_image);
        UnloadTexture(tex);

        cur = cur->next;
    }
    UnloadRenderTexture(cur_tex);

    srand(2304093940);
    for (int i = 0; i < data.count * num_examples_per_image; i++) {
        // int s1 = random_uniform_range(data.count * num_examples_per_image);
        int s1 = i;
        int s2 = i + rand() / (RAND_MAX / (data.count * num_examples_per_image - i) + 1);
        if (s1 >= data.count * num_examples_per_image) {
            s1 = data.count * num_examples_per_image - 1;
        }
        if (s2 >= data.count * num_examples_per_image) {
            s2 = data.count * num_examples_per_image - 1;
        }

        struct Example temp = shuffler[s1];
        shuffler[s1] = shuffler[s2];
        shuffler[s2] = temp; 
    }

    for (int i = 0; i < training_info->train_size; i++) {
        training_info->train_x[i] = matrix_allocator(input_size, 1);
        training_info->train_y[i] = matrix_allocator(output_size, 1);

        convert_image_to_mymatrix(&training_info->train_x[i], shuffler[i].image);
        one_hot_encode_matrix(&training_info->train_y[i], shuffler[i].label);
    }

    for (int i = 0; i < training_info->test_size; i++) {
        training_info->test_x[i] = matrix_allocator(input_size, 1);
        training_info->test_y[i] = matrix_allocator(output_size, 1);

        convert_image_to_mymatrix(&training_info->test_x[i], shuffler[i + training_info->train_size].image);
        one_hot_encode_matrix(&training_info->test_y[i], shuffler[i + training_info->train_size].label);
    }

    for (int i = 0; i < data.count * num_examples_per_image; i++) {
        UnloadImage(shuffler[i].image);
    }
    free(shuffler);

    printf("Created training set with %d examples (%d training, %d testing)\n", data.count * num_examples_per_image, training_info->train_size, training_info->test_size);
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

            for (int i = 0; i < images.num_labels; i++) {
                free(images.label_names[i]);
            }
            free(images.label_names);
            break;
        default:
            assert(0);
    }
}