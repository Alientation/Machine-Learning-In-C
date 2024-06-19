#include <app/drawpanel.h>
#include <app/visutil.h>
#include <app/dataset.h>

#include <stdio.h>
#include <string.h>

// TODO perhaps for optimization, but not really critical, instead of storing these saved
// drawing panel images as render textures which would be put in VRAM, store as images on RAM and
// write to the drawing panel's render texture when needed (saves VRAM space and time spent sending these textures to it after each draw segment)
void DrawingPanelFreeHistory(drawing_panel_args_t *draw_args) {
    segment_list_node_t *cur = draw_args->segments_list_head;
    while (cur != NULL) {
        segment_list_node_t *prev = cur;
        cur = cur->next;
        UnloadRenderTexture(prev->saved_image);
        free(prev);
    }
    draw_args->segments_list_cur = NULL;
    draw_args->segments_list_head = NULL;
    draw_args->segments_list_size = 0;
}

void DrawingPanelUndo(drawing_panel_args_t *draw_args) {
    if (draw_args->segments_list_cur == NULL || draw_args->segments_list_cur->prev == NULL) {
        return;
    }

    draw_args->segments_list_cur = draw_args->segments_list_cur->prev;
    BeginTextureMode(draw_args->draw_texture);
    {
        DrawTexture(draw_args->segments_list_cur->saved_image.texture, 0, 0, WHITE);
    }
    EndTextureMode();

    draw_args->updated = true;
}

void DrawingPanelRedo(drawing_panel_args_t *draw_args) {
    if (draw_args->segments_list_cur == NULL || draw_args->segments_list_cur->next == NULL) {
        return;
    }

    draw_args->segments_list_cur = draw_args->segments_list_cur->next;
    BeginTextureMode(draw_args->draw_texture);
    {
        DrawTexture(draw_args->segments_list_cur->saved_image.texture, 0, 0, WHITE);
    }
    EndTextureMode();

    draw_args->updated = true;
}

void DrawingPanelAdd(drawing_panel_args_t *draw_args) {
    segment_list_node_t *new_node = malloc(sizeof(segment_list_node_t));
    new_node->prev = NULL;
    new_node->next = NULL;
    new_node->saved_image = LoadRenderTexture(draw_args->draw_texture.texture.width, draw_args->draw_texture.texture.height);
    BeginTextureMode(new_node->saved_image);
    {
        ClearBackground(WHITE);
        DrawTexture(draw_args->draw_texture.texture, 0, 0, WHITE);
    }
    EndTextureMode();

    if (draw_args->segments_list_cur == NULL) {
        draw_args->segments_list_cur = new_node;
        draw_args->segments_list_head = new_node;
        draw_args->segments_list_size = 1;
    } else {
        segment_list_node_t *delete = draw_args->segments_list_cur->next;
        while (delete != NULL) {
            segment_list_node_t *prev = delete;
            delete = delete->next;
            draw_args->segments_list_size--;
            
            UnloadRenderTexture(prev->saved_image);
            free(prev);
        }

        new_node->prev = draw_args->segments_list_cur;
        draw_args->segments_list_cur->next = new_node;
        draw_args->segments_list_size++;
        draw_args->segments_list_cur = new_node;
    }

    draw_args->updated = true;
}

void DrawingPanelClear(drawing_panel_args_t *draw_args) {
    BeginTextureMode(draw_args->draw_texture);
    {
        ClearBackground(WHITE);
    }
    EndTextureMode();

    DrawingPanelAdd(draw_args);

    draw_args->updated = true;
}

void GuiFileViewer(drawing_panel_args_t *draw_args, Rectangle file_viewer_r) {
    // files can be dropped into the current directory
    char *dir_path = strdup(concat(3, GetWorkingDirectory(), "\\", draw_args->dataset_directory));
    if (IsFileDropped()) {
        FilePathList dropped_files = LoadDroppedFiles();
        for (int i = 0; i < dropped_files.count; i++) {
            const char *file_path = concat(3, dir_path, "\\", GetFileName(dropped_files.paths[i]));
            printf("loading file %s\n", file_path);
            char *text = LoadFileText(dropped_files.paths[i]);
            SaveFileText(file_path, text);
            UnloadFileText(text);
        }
        UnloadDroppedFiles(dropped_files);
    }

    // display all dataset files in directory
    FilePathList data_files = LoadDirectoryFiles(dir_path);
    free(dir_path);
    char* file_paths[data_files.count];
    for (int i = 0; i < data_files.count; i++) {
        const char* file = GetFileName(data_files.paths[i]);
        file_paths[i] = malloc(sizeof(char) * (strlen(file) + 1));
        strcpy(file_paths[i], file);
    }

    const char* text = data_files.count > 0 ? TextJoin((const char**)file_paths, data_files.count, ";") : NULL;
    for (int i = 0; i < data_files.count; i++) {
        free(file_paths[i]);
    }
    
    int prev_selected_file = draw_args->sel_dataset_index;
    GuiListView(file_viewer_r, text, &draw_args->dataset_list_scroll_index, &draw_args->sel_dataset_index);
    if (draw_args->sel_dataset_index != prev_selected_file) {
        // unload the previous dataset
        if (prev_selected_file != -1) {
            WriteDataSet(draw_args->current_dataset);
            UnloadDataSet(draw_args->current_dataset);
            UnloadImageDataSetVisualizer(draw_args->img_dataset_vis);
        }

        // select and load the new dataset
        draw_args->sel_dataset_image_index = -1;
        draw_args->sel_label_index = -1;
        if (draw_args->sel_dataset_index != -1) {
            draw_args->current_dataset = LoadDataSet(data_files.paths[draw_args->sel_dataset_index]);
            
            // handle if selected dataset is invalid
            if (draw_args->current_dataset.type == DATASET_INVALID) {
                printf("Invalid dataset selected \'%s\'\n", data_files.paths[draw_args->sel_dataset_index]);
                draw_args->sel_dataset_index = -1;
            } else {
                draw_args->img_dataset_vis = LoadImageDataSetVisualizer(&draw_args->current_dataset);
            }
        }
    }
    UnloadDirectoryFiles(data_files);
}

void GuiAddDataset(drawing_panel_args_t *draw_args, Rectangle file_viewer_r) {
    // name of new dataset
    Rectangle name_ds_r = {
        .x = file_viewer_r.x,
        .y = file_viewer_r.y + file_viewer_r.height + 40,
        .width = file_viewer_r.width - 60,
        .height = 30,
    };
    DrawCenteredText("New Dataset", file_viewer_r.x + file_viewer_r.width/2, name_ds_r.y - 20, 12, BLACK);
    if (GuiTextBox(name_ds_r, draw_args->add_dataset_file_name, FILE_NAME_BUFFER_SIZE, draw_args->is_editing_dataset_file_name)) {
        draw_args->is_editing_dataset_file_name = !draw_args->is_editing_dataset_file_name;
    }

    // add new dataset button
    Rectangle add_ds_r = {
        .x = name_ds_r.x + name_ds_r.width + 10,
        .y = name_ds_r.y,
        .width = file_viewer_r.width - name_ds_r.width - 10,
        .height = 30,
    };
    if (GuiButton(add_ds_r, "Create") && strlen(draw_args->add_dataset_file_name) != 0) {
        if (draw_args->add_dataset_type == 0) {
            const char* dataset_path = concat(6, GetWorkingDirectory(), "\\", draw_args->dataset_directory, "\\", draw_args->add_dataset_file_name, ".ds");
            dataset_t dataset = ConstructImageDataSet(dataset_path, TextToInteger(draw_args->images_dataset_width_input), 
                    TextToInteger(draw_args->images_dataset_height_input), draw_args->num_labels, draw_args->label_names);
            // if there is not already a file with the same name in the directory
            if (!FileExists(dataset_path)) {
                // create an empty dataset file
                WriteDataSet(dataset);
                UnloadDataSet(dataset);
            }
        } else {
            assert(0);
        }
    }

    // new dataset options
    // type of dataset, currently only possible type are images
    const int num_dataset_types = 1;
    Rectangle ds_type_r = {
        .x = name_ds_r.x,
        .y = name_ds_r.y + name_ds_r.height + 10,
        .width = file_viewer_r.width/num_dataset_types,
        .height = 30
    };
    GuiToggleGroup(ds_type_r, "Images", &draw_args->add_dataset_type);
    
    // If selected dataset to add is IMAGE
    if (draw_args->add_dataset_type == 0) {
        // need to know the height and width of images in the dataset
        Rectangle imgs_ds_width_label_r = {
            .x = ds_type_r.x,
            .y = ds_type_r.y + ds_type_r.height + 10,
            .width = 80,
            .height = 30,
        };
        DrawCenteredText("Image Width", imgs_ds_width_label_r.x + 40, imgs_ds_width_label_r.y + 15, 12, BLACK);
        
        Rectangle imgs_ds_width_r = {
            .x = imgs_ds_width_label_r.x + imgs_ds_width_label_r.width + 10,
            .y = imgs_ds_width_label_r.y,
            .width = file_viewer_r.width - imgs_ds_width_label_r.width - 10,
            .height = 30,
        };
        if (GuiTextBox(imgs_ds_width_r, draw_args->images_dataset_width_input, NUMBER_INPUT_BUFFER_SIZE, draw_args->images_dataset_width_option_active)) {
            draw_args->images_dataset_width_option_active = !draw_args->images_dataset_width_option_active;
        }

        Rectangle images_dataset_height_label_rec = {
            .x = ds_type_r.x,
            .y = imgs_ds_width_label_r.y + imgs_ds_width_label_r.height + 10,
            .width = 80,
            .height = 30,
        };
        DrawCenteredText("Image Height", images_dataset_height_label_rec.x + 40, images_dataset_height_label_rec.y + 10, 12, BLACK);
        
        Rectangle images_dataset_height_rec = {
            .x = images_dataset_height_label_rec.x + images_dataset_height_label_rec.width + 10,
            .y = images_dataset_height_label_rec.y,
            .width = file_viewer_r.width - images_dataset_height_label_rec.width - 10,
            .height = 30,
        };
        if (GuiTextBox(images_dataset_height_rec, draw_args->images_dataset_height_input, NUMBER_INPUT_BUFFER_SIZE, draw_args->images_dataset_height_option_active)) {
            draw_args->images_dataset_height_option_active = !draw_args->images_dataset_height_option_active;
        }
    } else {
        assert(0);
    }
}

// must be an image dataset
void GuiDisplayDataset(drawing_panel_args_t *draw_args, Rectangle img_preview_r, Rectangle file_viewer_r) {
    // don't display dataset if no dataset is chosen
    if (draw_args->sel_dataset_index == -1 || draw_args->current_dataset.type != DATASET_IMAGES) {
        return;
    }
    dataset_t *dataset = &draw_args->current_dataset;
    struct DataSetData_Images *img_ds = &draw_args->current_dataset.data.image_dataset;

    Rectangle ds_display_r = {
        .x = img_preview_r.x + img_preview_r.width + 10,
        .y = img_preview_r.y + img_preview_r.height/2 - (60 * NUMBER_DISPLAYED_IMAGES - 10)/2,
        .width = 50,
        .height = 60 * NUMBER_DISPLAYED_IMAGES - 10
    };

    image_dataset_visualizer_t *vis = &draw_args->img_dataset_vis;
    for (int i = 0; i < vis->number_displayed; i++) {
        Rectangle img_display_r = {
            .x = ds_display_r.x, .y = ds_display_r.y + 60 * i, .width = 50, .height = 50
        };

        // check if user selected image from the dataset display
        bool is_mouse_over = CheckCollisionPointRec(GetMousePosition(), img_display_r);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && is_mouse_over) {
            if (draw_args->sel_dataset_image_index == i) {
                draw_args->sel_dataset_image_index = -1;
                draw_args->sel_label_index = -1;
            } else {
                draw_args->sel_label_index = vis->displayed_images_nodes[i]->label;
                draw_args->sel_dataset_image_index = i;
            }
        }

        // draw the preview images of the dataset
        DrawTexturePro(vis->displayed_images[i],
                (Rectangle) {.x = 0, .y = 0, .width = img_ds->uniform_width, .height = img_ds->uniform_height},
                img_display_r, (Vector2) {.x = 0, .y = 0}, 0, WHITE);
        DrawRectangleLinesEx((Rectangle) {.x = img_display_r.x-1, .y = img_display_r.y-1, .width = img_display_r.width, .height = img_display_r.height},
                2, (draw_args->sel_dataset_image_index == i || is_mouse_over) ? BLUE : DARKGRAY);
    } 

    // draw the arrows on the top and bottom only if there is more images than the number of displayed images
    if (img_ds->count > vis->number_displayed) {
        // move display images faster if shift is pressed
        int move_magnitude = IsKeyDown(KEY_LEFT_CONTROL) ? 100000 : IsKeyDown(KEY_LEFT_SHIFT) ? 5 : 1;
        
        // move up
        if (vis->left_image_index > 0) {
            if (GuiButton((Rectangle) {.x = ds_display_r.x + ds_display_r.width/2 - 24, .y = ds_display_r.y - 28, .width = 48, .height = 20}, "")) {
                printf("Moving Display Images Up\n");
                MoveDisplayImageDataSetVisualizer(vis, -move_magnitude);
            }
            GuiDrawIcon(ICON_ARROW_UP_FILL, ds_display_r.x + ds_display_r.width/2 - 8, ds_display_r.y - 26, 1, BLACK);
        }

        // move down
        if (vis->left_image_index < img_ds->count - NUMBER_DISPLAYED_IMAGES) {
            if (GuiButton((Rectangle) {.x = ds_display_r.x + ds_display_r.width/2 - 24, .y = ds_display_r.y + ds_display_r.height + 4, .width = 48, .height = 20}, "")) {
                printf("Moving Display Images Down\n");
                MoveDisplayImageDataSetVisualizer(vis, move_magnitude);
            }
            GuiDrawIcon(ICON_ARROW_DOWN_FILL, ds_display_r.x + ds_display_r.width/2 - 8, ds_display_r.y + ds_display_r.height + 6, 1, BLACK);
        }
        
        // update currently selected label
        if (draw_args->sel_dataset_image_index != -1) {
            draw_args->sel_label_index = vis->displayed_images_nodes[draw_args->sel_dataset_image_index]->label;
        }
    }

    // display currently selected dataset info
    Rectangle ds_info_r = {
        .x = file_viewer_r.x + file_viewer_r.width + 20,
        .y = file_viewer_r.y,
        .width = 200,
        .height = 300,
    };
    
    DrawOutlinedRectangleRec(ds_info_r, RAYWHITE, 2, BLACK);
    DrawOutlinedRectangle(ds_info_r.x, ds_info_r.y, ds_info_r.width, 40, GRAY, 2, BLACK);
    DrawOutlinedCenteredText("Image Dataset", ds_info_r.x + ds_info_r.width/2, ds_info_r.y + 20, 20, WHITE, 1, BLACK);

    DrawText(TextFormat("number of images: %d", img_ds->count), ds_info_r.x + 10, ds_info_r.y + 50, 16, BLACK);
    DrawText(TextFormat("image width: %d", img_ds->uniform_width), ds_info_r.x + 10, ds_info_r.y + 70, 16, BLACK);
    DrawText(TextFormat("image height: %d", img_ds->uniform_height), ds_info_r.x + 10, ds_info_r.y + 90, 16, BLACK);
    DrawText(TextFormat("num labels: %d", img_ds->num_labels), ds_info_r.x + 10, ds_info_r.y + 110, 16, BLACK);
    for (int i = 0; i < img_ds->num_labels; i++) {
        DrawText(TextFormat("label %d)  %s", i+1, img_ds->label_names[i]), ds_info_r.x + 20, ds_info_r.y + 130 + i * 15, 12, DARKGRAY);
    }

    // TODO add more options like selecting what kinds of image transformations to apply to
    // expand the amount of data and generate variety
    Rectangle transformation_label_r = {
        .x = ds_info_r.x, .y = ds_info_r.y + ds_info_r.height + 10, .width = 90, .height = 20
    };
    Rectangle transformation_picker_r = {
        .x = transformation_label_r.x + transformation_label_r.width + 10, .y = transformation_label_r.y, .width = ds_info_r.width - transformation_label_r.width - 10, .height = 20
    };

    const int TEXT_INPUT_BUFFER_SIZE = 10;
    char TEXT_INPUT_BUFFER[TEXT_INPUT_BUFFER_SIZE];
    strcpy(TEXT_INPUT_BUFFER, TextFormat("%d", draw_args->num_transformations));

    DrawCenteredText("# Applied:", _UNPACK_REC_CENTER(transformation_label_r), 10, BLACK);
    if (GuiTextBox(transformation_picker_r, TEXT_INPUT_BUFFER, TEXT_INPUT_BUFFER_SIZE, draw_args->is_transformations_active)) {
        draw_args->is_transformations_active = !draw_args->is_transformations_active;
    }
    draw_args->num_transformations = TextToInteger(TEXT_INPUT_BUFFER);

    Rectangle rotation_label_r = RecShift(transformation_label_r, 0, 25);
    Rectangle rotation_picker_r = RecShift(transformation_picker_r, 0, 25);
    DrawCenteredText(TextFormat("Rot: %d", (int) draw_args->max_rotation_degree), _UNPACK_REC_CENTER(rotation_label_r), 10, BLACK);
    GuiSlider(rotation_picker_r, "0", "180", &draw_args->max_rotation_degree, 0, 180);
    
    Rectangle translation_x_label_r = RecShift(rotation_label_r, 0, 25);
    Rectangle translation_x_picker_r = RecShift(rotation_picker_r, 0, 25);
    DrawCenteredText(TextFormat("Shift X: %d", (int) draw_args->max_translations_pixels_x), _UNPACK_REC_CENTER(translation_x_label_r), 10, BLACK);
    GuiSlider(translation_x_picker_r, "0", "20", &draw_args->max_translations_pixels_x, 0, 20);
    
    Rectangle translation_y_label_r = RecShift(translation_x_label_r, 0, 25);
    Rectangle translation_y_picker_r = RecShift(translation_x_picker_r, 0, 25);
    DrawCenteredText(TextFormat("Shift Y: %d", (int) draw_args->max_translations_pixels_y), _UNPACK_REC_CENTER(translation_y_label_r), 10, BLACK);
    GuiSlider(translation_y_picker_r, "0", "20", &draw_args->max_translations_pixels_y, 0, 20);

    Rectangle artifact_label_r = RecShift(translation_y_label_r, 0, 25);
    Rectangle artifact_picker_r = RecShift(translation_y_picker_r, 0, 25);
    DrawCenteredText(TextFormat("Artifacts: %.3f", draw_args->max_artifacts), _UNPACK_REC_CENTER(artifact_label_r), 10, BLACK);
    GuiSlider(artifact_picker_r, "0", "1", &draw_args->max_artifacts, 0, 1);

    // load dataset into training info for the model to run on
    if (GuiButton((Rectangle) {.x = ds_info_r.x, .y = ds_info_r.y + ds_info_r.height + 150, .width = 80, .height = 30}, "Use")) {
        training_info_free(draw_args->vis_args->training_info);
        ImageDataSetConvertToTraining(draw_args->vis_args->training_info, dataset, draw_args->num_transformations, 
                draw_args->max_rotation_degree, draw_args->max_translations_pixels_x, draw_args->max_translations_pixels_y, draw_args->max_artifacts);
        draw_args->is_dataset_viewer_open = false;
        draw_args->is_save_popup_open = false;
        draw_args->is_open = false;
    }

    // display currently selected image info
    Rectangle img_info_r = {
        .x = ds_display_r.x + ds_display_r.width + 20,
        .y = img_preview_r.y + img_preview_r.height/2 - (26 * img_ds->num_labels - 3)/2,
        .width = 20,
        .height = 26 * img_ds->num_labels - 3,
    };

    for (int i = 0; i < img_ds->num_labels; i++) {
        Rectangle label_selection_r = {
            .x = img_info_r.x,
            .y = img_info_r.y + 26 * i,
            .width = 20,
            .height = 20
        };

        bool is_mouse_over = CheckCollisionPointRec(GetMousePosition(), label_selection_r);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && is_mouse_over) {
            draw_args->sel_label_index = i;
            if (draw_args->sel_dataset_image_index != -1) {
                vis->displayed_images_nodes[draw_args->sel_dataset_image_index]->label = i;
            }
        }

        Color color = LIGHTGRAY;
        if (draw_args->sel_label_index == i) {
            color = ColorAlpha(BLUE, 0.2);
        } else if (is_mouse_over) {
            color = ColorAlpha(BLUE, 0.4);
        }
        DrawOutlinedRectangleRec(label_selection_r, color, 2, (is_mouse_over || i == draw_args->sel_label_index) ? BLUE : DARKGRAY);
        DrawCenteredText(img_ds->label_names[i], label_selection_r.x + label_selection_r.width/2, label_selection_r.y + label_selection_r.height/2, 10, BLACK);
    }


    // image options for the currently selected image from the dataset or the drawn image in the drawing panel
    if (draw_args->sel_dataset_image_index == -1) {
        Rectangle add_img_r = {
            .x = img_preview_r.x + 50,
            .y = img_preview_r.y + img_preview_r.height + 10,
            .width = img_preview_r.width - 100,
            .height = 30,
        };
        if (GuiButton(add_img_r, "Add to Dataset")) {
            printf("Saved current image to %s\n", dataset->file_path);

            DataSetAddImage(dataset, LoadImageFromTexture(draw_args->input_texture.texture), draw_args->sel_label_index);
            UpdateImageDataSetVisualizer(&draw_args->img_dataset_vis);

            draw_args->is_save_popup_open = false;
        }

        // todo add insert image option
    } else if (draw_args->sel_dataset_image_index != -1) {
        Rectangle delete_img_r = {
            .x = img_preview_r.x + 50,
            .y = img_preview_r.y + img_preview_r.height + 10,
            .width = img_preview_r.width - 100,
            .height = 30,
        };

        if (GuiButton(delete_img_r, "Delete")) {
            printf("Deleted image from %s", dataset->file_path);

            DataSetRemoveImage(dataset, vis->left_image_index + draw_args->sel_dataset_image_index);
            if (vis->left_image_index + NUMBER_DISPLAYED_IMAGES >= img_ds->count) {
                vis->left_image_index = img_ds->count - NUMBER_DISPLAYED_IMAGES;
                if (vis->left_image_index < 0) {
                    vis->left_image_index = 0;
                }
            }
            
            UpdateImageDataSetVisualizer(vis);
            draw_args->sel_dataset_image_index = -1;
        }
    }
}

void GuiSavePopup(drawing_panel_args_t *draw_args) {
    if (!draw_args->is_save_popup_open) {
        draw_args->is_dataset_viewer_open = false;
        return;
    }

    // darken background
    DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, ColorAlpha(BLACK, 0.5));

    Rectangle save_window_r = RecCenteredMargin(SCREEN_WIDTH, SCREEN_HEIGHT, 200, 150);
    draw_args->is_save_popup_open = !GuiWindowBox(save_window_r, "Save Drawing");

    // TODO, since looking up what files exist in the save folder will probably be slow, instead
    // look it up once at the start, and then keep information about the saved files in memory

    Rectangle img_preview_r = RecCenteredRecDimV(save_window_r, Vec2DExtend(200));
    image_dataset_visualizer_t *ds_vis = &draw_args->img_dataset_vis;

    // model input is either the selected image from the dataset or the drawn image in the drawing panel
    Texture2D input_texture = draw_args->sel_dataset_image_index != -1 ? ds_vis->displayed_images[draw_args->sel_dataset_image_index] : draw_args->input_texture.texture;
    DrawTexturePro(input_texture, _REC_FROM_DIM(0, 0, input_texture.width, input_texture.height), img_preview_r, Vec2DExtend(0), 0, WHITE);
    DrawRectangleLines(_UNPACK_REC(RecCenteredRecMargin(img_preview_r, -1, -1)), BLACK);

    Rectangle file_viewer_r = {
        .x = save_window_r.x + 30, 
        .y = save_window_r.y + 45, 
        .width = 200, 
        .height = 300,
    };
    DrawCenteredText(draw_args->dataset_directory, file_viewer_r.x + file_viewer_r.width/2, file_viewer_r.y - 10, 12, BLACK);
    GuiFileViewer(draw_args, file_viewer_r);

    GuiAddDataset(draw_args, file_viewer_r);

    GuiDisplayDataset(draw_args, img_preview_r, file_viewer_r);
}


// Draw brush tools to assist with drawing
void GuiDrawingTools(drawing_panel_args_t *draw_args, Rectangle draw_window_rec, Rectangle draw_panel_rec) {
    // Brush color picker
    Rectangle color_picker_rec = {
        .x = draw_panel_rec.x + draw_panel_rec.width + 40,
        .y = draw_window_rec.y + draw_window_rec.height/2 - 120/2,
        .width = 120,
        .height = 120,
    };
    if (draw_args->is_save_popup_open) {
        // don't change anything since another window is open
        Vector3 dummy = draw_args->brush_color;
        GuiColorPickerHSV(color_picker_rec, "Brush Color", &dummy);
    } else {
        GuiColorPickerHSV(color_picker_rec, "Brush Color", &draw_args->brush_color);
    }

    Rectangle brush_color_rec = {
        .x = color_picker_rec.x,
        .y = color_picker_rec.y - 40,
        .width = 30,
        .height = 30, 
    };

    // Brush size picker
    Rectangle brush_size_picker_rec = {
        .x = draw_panel_rec.x + draw_panel_rec.width + 30,
        .y = brush_color_rec.y - 40,
        .width = 100,
        .height = 30,
    };

    float max_brush_size = 18;
    float min_brush_size = 4;
    if (draw_args->is_save_popup_open) {
        float dummy = draw_args->brush_size;
        GuiSliderBar(brush_size_picker_rec, TextFormat("%.2f", min_brush_size), TextFormat("%.2f", max_brush_size), &dummy, min_brush_size, max_brush_size);
    } else {
        GuiSliderBar(brush_size_picker_rec, TextFormat("%.2f", min_brush_size), TextFormat("%.2f", max_brush_size), &draw_args->brush_size, min_brush_size, max_brush_size);
    }
    DrawText("Brush Size", brush_size_picker_rec.x, brush_size_picker_rec.y - 20, 12, BLACK);

    Rectangle brush_size_display_rec = {
        .x = brush_size_picker_rec.x + brush_size_picker_rec.width + max_brush_size + 20,
        .y = brush_size_picker_rec.y + brush_size_picker_rec.height/2 - max_brush_size,
        .width = max_brush_size * 2,
        .height = max_brush_size * 2,
    };
    DrawRectangleRoundedLines(brush_size_display_rec, 0.02, 4, 2, BLACK);
    DrawCircle(_UNPACK_REC_CENTER(brush_size_display_rec), draw_args->brush_size, ColorFromHSV(_UNPACK_VEC3(draw_args->brush_color)));


    // Drawing Panel Options
    Rectangle draw_panel_clear_rec = {
        .x = draw_panel_rec.x,
        .y = draw_panel_rec.y + draw_panel_rec.height + 10,
        .width = 40,
        .height = 20,
    };
    if ((GuiButton(draw_panel_clear_rec, "Clear") || (IsKeyDown(KEY_LEFT_CONTROL) && IsKeyPressed(KEY_C))) && !draw_args->is_save_popup_open) {
        DrawingPanelClear(draw_args);
    }

    Rectangle draw_panel_undo_rec = {
        .x = draw_panel_clear_rec.x + draw_panel_clear_rec.width + 10,
        .y = draw_panel_clear_rec.y,
        .width = 40,
        .height = 20,
    };
    if (GuiButton(draw_panel_undo_rec, "Undo") && !draw_args->is_save_popup_open) {
        DrawingPanelUndo(draw_args);
    }

    Rectangle draw_panel_redo_rec = {
        .x = draw_panel_undo_rec.x + draw_panel_undo_rec.width + 10,
        .y = draw_panel_undo_rec.y,
        .width = 40,
        .height = 20,
    };
    if (GuiButton(draw_panel_redo_rec, "Redo") && !draw_args->is_save_popup_open) {
        DrawingPanelRedo(draw_args);
    }

    // Save Image Button
    if (GuiButton((Rectangle) {.x = draw_panel_rec.x + draw_panel_rec.width - 40, .y = draw_panel_undo_rec.y, .width = 40, .height = 20}, "Save") ||
        (IsKeyDown(KEY_LEFT_CONTROL) && IsKeyPressed(KEY_S))) {
        draw_args->is_save_popup_open = true;
    }
}

void GuiDrawPanel(drawing_panel_args_t *draw_args, Rectangle draw_panel_rec) {
    bool is_draw = IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsGestureDetected(GESTURE_DRAG);
    bool is_erase = IsMouseButtonDown(MOUSE_BUTTON_RIGHT);
    bool just_off_panel = draw_args->is_dragged && !CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);
    bool should_draw = (is_draw || is_erase) && CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);
    should_draw |= just_off_panel;
    should_draw &= !draw_args->is_save_popup_open;
    if (should_draw) {
        Color draw_color = is_erase ? WHITE : ColorFromHSV(draw_args->brush_color.x, draw_args->brush_color.y, draw_args->brush_color.z);

        Vector2 size = {
            .x = draw_args->brush_size,
            .y = draw_args->brush_size,
        };
        Vector2 pos = GetMousePosition();
        pos.x -= size.x / 2.0;
        pos.y -= size.y / 2.0;

        pos.x -= draw_panel_rec.x;
        pos.y -= draw_panel_rec.y;

        pos.y = draw_args->draw_texture.texture.height - pos.y; // texture drawing flips vertically because of opengl
        BeginTextureMode(draw_args->draw_texture);
        {
            DrawCircleV(pos, draw_args->brush_size, draw_color);
            
            // to connect points since the refresh rate is 60 fps which causes gaps
            if (draw_args->prev_draw_pos.x != -1 && draw_args->prev_draw_pos.y != -1) {
                DrawLineEx(draw_args->prev_draw_pos, pos, draw_args->brush_size * 2, draw_color);
                DrawCircleV(draw_args->prev_draw_pos, draw_args->brush_size, draw_color);
            }
            
            draw_args->prev_draw_pos = pos;
        }
        EndTextureMode();

        draw_args->is_drawing = true;
        draw_args->updated = true;
    } else {
        // is not drawing
        draw_args->prev_draw_pos = (Vector2) {.x = GetMousePosition().x - (draw_args->brush_size/2) - draw_panel_rec.x, 
                .y = draw_args->draw_texture.texture.height - (GetMousePosition().y - (draw_args->brush_size/2) - draw_panel_rec.y)};
        
        // check if just stopped drawing, register segment
        if (draw_args->is_drawing) {
            DrawingPanelAdd(draw_args);
        }
        draw_args->is_drawing = false;     
    }

    draw_args->is_dragged = IsGestureDetected(GESTURE_DRAG) && CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);
}

void GuiModelInfo(drawing_panel_args_t *draw_args, Rectangle draw_panel_rec) {
    // draw image onto the texture the model will see, scaled down
    RenderTexture2D input_texture = draw_args->input_texture;
    BeginTextureMode(input_texture);
    {
        DrawTexturePro(draw_args->draw_texture.texture, (Rectangle) {.x = 0, .y = 0, .width = draw_args->draw_texture.texture.width, .height = -draw_args->draw_texture.texture.height}, 
                (Rectangle) {.x = 0, .y = 0,. width = input_texture.texture.width, .height = input_texture.texture.height}, (Vector2) {0, 0}, 0, WHITE);
    }
    EndTextureMode();

    // check if we should update the model with information about the drawn image
    draw_args->cur_frames++;
    if (draw_args->updated && draw_args->update_frames <= draw_args->cur_frames) {
        draw_args->cur_frames = 0;
        draw_args->updated = false;
        Image input_image = LoadImageFromTexture(input_texture.texture);

        if (draw_args->gray_scale) {
            ImageColorGrayscale(&input_image);
            Color *pixels = LoadImageColors(input_image);
            UpdateTexture(input_texture.texture, pixels);
            UnloadImageColors(pixels);
        }

        // read in model input into buffer
        for (int r = 0; r < input_image.height; r++) {
            for (int c = 0; c < input_image.width; c++) {
                Color color = GetImageColor(input_image, c, r);
                draw_args->output_buffer[r * input_image.height + c] = color.r;
            }
        }
        
        neural_network_model_t *model = draw_args->vis_args->training_info->model;
        convert_image_to_mymatrix(&model->input_layer->layer.input.input_values, input_image);
        mymatrix_t output = model_calculate(model);
        UnloadImage(input_image);
    }

    // Draw what the model will see, make it a scaled multiple of the model's input to prevent rescaling issues
    Rectangle model_input_rec = {
        .x = draw_panel_rec.x - 10 - (120 / draw_args->buffer_width) * draw_args->buffer_width,
        .y = draw_panel_rec.y,
        .width = (120 / draw_args->buffer_width) * draw_args->buffer_width,
        .height = (120 / draw_args->buffer_height) * draw_args->buffer_height
    };

    // display what the model with see on the side
    DrawTexturePro(input_texture.texture, (Rectangle) {.x = 0, .y = 0, .width = input_texture.texture.width, .height = input_texture.texture.height}, 
            model_input_rec, (Vector2) {0, 0}, 0, WHITE);

    // run the model
    neural_network_model_t *model = draw_args->vis_args->training_info->model;
    mymatrix_t output = model->output_layer->layer.output.output_values;
    int highest_guess = 0;
    for (int i = 0; i < draw_args->vis_args->num_labels; i++) {
        if (output.matrix[i][0] > output.matrix[highest_guess][0]) {
            highest_guess = i;
        }
    }

    // show the model's prediction and confidence
    DrawCenteredText(TextFormat("%d (%f)", highest_guess, output.matrix[highest_guess][0]), 
            model_input_rec.x + model_input_rec.width/2, model_input_rec.y + model_input_rec.height + 10, 12, BLACK);
    
    // TODO show all confidences of each label on the side
}

// GOALS
// Set Brush Mode
// Save image
// Load image
static const int line_thickness = 4;
void GuiDrawingPanelPopup(drawing_panel_args_t *draw_args) {
    if (!draw_args->is_open) {
        draw_args->is_save_popup_open = false;
        return;
    }
    
    // darken background
    DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, ColorAlpha(BLACK, 0.5));

    // drawing panel
    Rectangle draw_window_rec = RecCenteredMargin(SCREEN_WIDTH, SCREEN_HEIGHT, 50, 50);
    Rectangle draw_panel_rec = RecCenteredRecDimV(draw_window_rec, VecAddC(RenderTextureDim(draw_args->draw_texture), line_thickness/2));
    draw_args->is_open = !GuiWindowBox(draw_window_rec, "Drawing Panel");

    DrawRectangleRec(draw_panel_rec, WHITE);
    DrawRectangleRoundedLines(draw_panel_rec, .02, 10, line_thickness, BLACK);

    // brush tools
    GuiDrawingTools(draw_args, draw_window_rec, draw_panel_rec);

    // drawing panel
    GuiDrawPanel(draw_args, draw_panel_rec);
    DrawTexture(draw_args->draw_texture.texture, draw_panel_rec.x + line_thickness/2, draw_panel_rec.y + line_thickness/2, WHITE);

    // model input image and output labels
    GuiModelInfo(draw_args, draw_panel_rec);

    // draw save window if it should be shown
    GuiSavePopup(draw_args);
}