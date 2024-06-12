#include <app/drawpanel.h>
#include <app/visutil.h>

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

void GuiSavePopup(drawing_panel_args_t *draw_args) {
    if (!draw_args->is_save_popup_open) {
        draw_args->is_file_viewer_open = false;
        return;
    }

    // darken background
    DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, ColorAlpha(BLACK, 0.5));

    Rectangle save_window_rec = {
        .x = 200,
        .y = 150,
        .width = SCREEN_WIDTH - 2 * 200,
        .height = SCREEN_HEIGHT - 2 * 150,
    };   
    
    draw_args->is_save_popup_open = !GuiWindowBox(save_window_rec, "Save Drawing");

    // IDEAS
    // name of file on bottom left, bottom right 'upload' button
    // labeler on the right,

    // TODO, since looking up what files exist in the save folder will probably be slow, instead
    // look it up once at the start, and then keep information about the saved files in memory

    Rectangle image_preview = {
        .x = save_window_rec.x + save_window_rec.width/2 - 100,
        .y = save_window_rec.y + save_window_rec.height/2 - 100,
        .width = 200,
        .height = 200,
    };

    RenderTexture2D input_texture = draw_args->input_texture;
    DrawTexturePro(input_texture.texture, (Rectangle) {.x = 0, .y = 0, .width = input_texture.texture.width, .height = input_texture.texture.height}, 
            image_preview, (Vector2) {0, 0}, 0, WHITE);

    DrawText(draw_args->data_directory, save_window_rec.x + 30, save_window_rec.y + save_window_rec.height/2 - 250, 16, BLACK);

    Rectangle file_viewer_rec = {
        .x = save_window_rec.x + 30, 
        .y = save_window_rec.y + save_window_rec.height/2 - 200, 
        .width = 200, 
        .height = 400,
    };

    const char* dir_paths[3] = {
        GetWorkingDirectory(),
        "\\",
        draw_args->data_directory,
    };
    const char *dir_path = TextJoin(dir_paths, 3, "");

    if (IsFileDropped()) {
        FilePathList dropped_files = LoadDroppedFiles();
        const char* file_paths[5] = {
            GetWorkingDirectory(),
            "\\",
            draw_args->data_directory,
            "\\",
            "FILL IN",
        };
        for (int i = 0; i < dropped_files.count; i++) {
            file_paths[4] = GetFileName(dropped_files.paths[i]);
            const char *file_path = TextJoin(file_paths, 5, "");
            printf("loading file %s\n", file_path);
            char *text = LoadFileText(dropped_files.paths[i]);
            SaveFileText(file_path, text);
            UnloadFileText(text);
        }
        UnloadDroppedFiles(dropped_files);
    }

    FilePathList data_files = LoadDirectoryFiles(dir_path);
    char* file_paths[data_files.count];
    for (int i = 0; i < data_files.count; i++) {
        const char* file = GetFileName(data_files.paths[i]);
        file_paths[i] = malloc(sizeof(char) * (strlen(file) + 1));
        strcpy(file_paths[i], file);
    }

    const char* text = TextJoin((const char**)file_paths, data_files.count, ";");
    for (int i = 0; i < data_files.count; i++) {
        free(file_paths[i]);
    }
    UnloadDirectoryFiles(data_files);
    
    int prev_selected_file = draw_args->selected_file;
    GuiListView(file_viewer_rec, text, &draw_args->file_list_scroll_index, &draw_args->selected_file);
    if (draw_args->selected_file != prev_selected_file) {
        // file changed, bring up the file
    }
}




// GOALS
// Set Brush Mode
// Save image
// Load image
// Show Buffered view that will be sent to the model
void GuiDrawingPanelPopup(drawing_panel_args_t *draw_args) {
    if (!draw_args->is_open) {
        draw_args->is_save_popup_open = false;
        return;
    }
    
    // darken background
    DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, ColorAlpha(BLACK, 0.5));

    // drawing panel
    RenderTexture2D draw_texture = draw_args->draw_texture;
    
    Rectangle draw_window_rec = {
        .x = 50,
        .y = 50,
        .width = SCREEN_WIDTH - 2 * 50,
        .height = SCREEN_HEIGHT - 2 * 50,
    };    

    
    const int line_thickness = 4;
    Rectangle draw_panel_rec = {
        .x = draw_window_rec.x + draw_window_rec.width/2 - draw_texture.texture.width/2 - line_thickness/4,
        .y = draw_window_rec.y + draw_window_rec.height/2 - draw_texture.texture.height/2 - line_thickness/4,
        .width = draw_texture.texture.width + line_thickness/2,
        .height = draw_texture.texture.height + line_thickness/2,
    };

    draw_args->is_open = !GuiWindowBox(draw_window_rec, "Drawing Panel");
    
    DrawRectangleRec(draw_panel_rec, WHITE);
    DrawRectangleRoundedLines(draw_panel_rec, .02, 10, line_thickness, BLACK);


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

    Vector2 brush_size_display_rec = {
        .x = brush_size_picker_rec.x + brush_size_picker_rec.width + max_brush_size * 2 + 20,
        .y = brush_size_picker_rec.y + brush_size_picker_rec.height/2
    };
    DrawRectangleRoundedLines((Rectangle) {.x = brush_size_display_rec.x - max_brush_size, .y = brush_size_display_rec.y - max_brush_size, .width = max_brush_size*2, .height = max_brush_size*2}, 
            0.02, 4, 2, BLACK);
    DrawCircleV(brush_size_display_rec , draw_args->brush_size, ColorFromHSV(draw_args->brush_color.x, draw_args->brush_color.y, draw_args->brush_color.z));


    // Drawing Panel Options
    Rectangle draw_panel_clear_rec = {
        .x = draw_panel_rec.x,
        .y = draw_panel_rec.y + draw_panel_rec.height + 10,
        .width = 40,
        .height = 20,
    };
    if (GuiButton(draw_panel_clear_rec, "Clear") && !draw_args->is_save_popup_open) {
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


    bool is_draw = IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsGestureDetected(GESTURE_DRAG);
    bool is_erase = IsMouseButtonDown(MOUSE_BUTTON_RIGHT);
    bool just_off_panel = draw_args->is_dragged && !CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);
    bool should_draw = (is_draw || is_erase) && CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);
    should_draw |= just_off_panel;
    should_draw &= !draw_args->is_save_popup_open;
    if (should_draw) {
        Color draw_color = ColorFromHSV(draw_args->brush_color.x, draw_args->brush_color.y, draw_args->brush_color.z);
        if (is_erase) {
            draw_color = WHITE;
        }

        Vector2 size = {
            .x = draw_args->brush_size,
            .y = draw_args->brush_size,
        };
        Vector2 pos = GetMousePosition();
        pos.x -= size.x / 2.0;
        pos.y -= size.y / 2.0;

        pos.x -= draw_panel_rec.x;
        pos.y -= draw_panel_rec.y;

        pos.y = draw_texture.texture.height - pos.y; // texture drawing flips vertically because of opengl or smthn

        BeginTextureMode(draw_texture);
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
                .y = draw_texture.texture.height - (GetMousePosition().y - (draw_args->brush_size/2) - draw_panel_rec.y)};
        

        // check if just stopped drawing, register segment
        if (draw_args->is_drawing) {
            DrawingPanelAdd(draw_args);
        }
        draw_args->is_drawing = false;     
    }

    draw_args->is_dragged = IsGestureDetected(GESTURE_DRAG) && CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);

    DrawTexture(draw_texture.texture, draw_panel_rec.x + line_thickness/2, draw_panel_rec.y + line_thickness/2, WHITE);


    // Draw what the model will see
    Rectangle model_input_rec = {
        .x = draw_panel_rec.x - 10 - (120 / draw_args->buffer_width) * draw_args->buffer_width,
        .y = draw_panel_rec.y,
        .width = (120 / draw_args->buffer_width) * draw_args->buffer_width,
        .height = (120 / draw_args->buffer_height) * draw_args->buffer_height
    };

    RenderTexture2D input_texture = draw_args->input_texture;
    BeginTextureMode(input_texture);
    {
        DrawTexturePro(draw_texture.texture, (Rectangle) {.x = 0, .y = 0, .width = draw_texture.texture.width, .height = -draw_texture.texture.height}, 
                (Rectangle) {.x = 0, .y = 0,. width = input_texture.texture.width, .height = input_texture.texture.height}, (Vector2) {0, 0}, 0, WHITE);
    }
    EndTextureMode();

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
        UnloadImage(input_image);
        
        // TODO run the model on the new input
    }

    DrawTexturePro(input_texture.texture, (Rectangle) {.x = 0, .y = 0, .width = input_texture.texture.width, .height = input_texture.texture.height}, 
            model_input_rec, (Vector2) {0, 0}, 0, WHITE);

    // TODO draw the model's output





    // Save Image Button
    if (GuiButton((Rectangle) {.x = draw_panel_rec.x + draw_panel_rec.width - 40, .y = draw_panel_undo_rec.y, .width = 40, .height = 20}, "Save")) {
        draw_args->is_save_popup_open = true;
    }
    GuiSavePopup(draw_args);
}