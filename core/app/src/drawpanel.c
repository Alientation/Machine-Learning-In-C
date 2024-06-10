#include <app/drawpanel.h>
#include <app/visutil.h>

// TODO perhaps for optimization, but not really critical, instead of storing these saved
// drawing panel images as render textures which would be put in VRAM, store as images on RAM and
// write to the drawing panel's render texture when needed (saves VRAM space and time spent sending these textures to it after each draw segment)
void DrawingPanelFreeHistory(drawing_panel_args_t *args) {
    segment_list_node_t *cur = args->segments_list_head;
    while (cur != NULL) {
        segment_list_node_t *prev = cur;
        cur = cur->next;
        UnloadRenderTexture(prev->saved_image);
        free(prev);
    }
    args->segments_list_cur = NULL;
    args->segments_list_head = NULL;
    args->segments_list_size = 0;
}

void DrawingPanelUndo(drawing_panel_args_t *args) {
    if (args->segments_list_cur == NULL || args->segments_list_cur->prev == NULL) {
        return;
    }

    args->segments_list_cur = args->segments_list_cur->prev;
    BeginTextureMode(args->draw_texture);
    {
        DrawTexture(args->segments_list_cur->saved_image.texture, 0, 0, WHITE);
    }
    EndTextureMode();

    args->updated = true;
}

void DrawingPanelRedo(drawing_panel_args_t *args) {
    if (args->segments_list_cur == NULL || args->segments_list_cur->next == NULL) {
        return;
    }

    args->segments_list_cur = args->segments_list_cur->next;
    BeginTextureMode(args->draw_texture);
    {
        DrawTexture(args->segments_list_cur->saved_image.texture, 0, 0, WHITE);
    }
    EndTextureMode();

    args->updated = true;
}

void DrawingPanelAdd(drawing_panel_args_t *args) {
    segment_list_node_t *new_node = malloc(sizeof(segment_list_node_t));
    new_node->prev = NULL;
    new_node->next = NULL;
    new_node->saved_image = LoadRenderTexture(args->draw_texture.texture.width, args->draw_texture.texture.height);
    BeginTextureMode(new_node->saved_image);
    {
        ClearBackground(WHITE);
        DrawTexture(args->draw_texture.texture, 0, 0, WHITE);
    }
    EndTextureMode();

    if (args->segments_list_cur == NULL) {
        args->segments_list_cur = new_node;
        args->segments_list_head = new_node;
        args->segments_list_size = 1;
    } else {
        segment_list_node_t *delete = args->segments_list_cur->next;
        while (delete != NULL) {
            segment_list_node_t *prev = delete;
            delete = delete->next;
            args->segments_list_size--;
            
            UnloadRenderTexture(prev->saved_image);
            free(prev);
        }

        new_node->prev = args->segments_list_cur;
        args->segments_list_cur->next = new_node;
        args->segments_list_size++;
        args->segments_list_cur = new_node;
    }

    args->updated = true;
}

void DrawingPanelClear(drawing_panel_args_t *args) {
    BeginTextureMode(args->draw_texture);
    {
        ClearBackground(WHITE);
    }
    EndTextureMode();

    DrawingPanelAdd(args);

    args->updated = true;
}

// GOALS
// Set Brush Mode
// Save image
// Load image
// Show Buffered view that will be sent to the model
void GuiDrawingPanelPopup(drawing_panel_args_t *args) {
    if (!args->is_open) {
        return;
    }
    
    // darken background
    DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, ColorAlpha(BLACK, 0.5));

    // drawing panel
    RenderTexture2D draw_texture = args->draw_texture;
    
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

    args->is_open = !GuiWindowBox(draw_window_rec, "Drawing Panel");
    
    DrawRectangleRec(draw_panel_rec, WHITE);
    DrawRectangleRoundedLines(draw_panel_rec, .02, 10, line_thickness, BLACK);


    // Brush color picker
    Rectangle color_picker_rec = {
        .x = draw_panel_rec.x + draw_panel_rec.width + 40,
        .y = draw_window_rec.y + draw_window_rec.height/2 - 120/2,
        .width = 120,
        .height = 120,
    };
    GuiColorPickerHSV(color_picker_rec, "Brush Color", &args->brush_color);

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
    GuiSliderBar(brush_size_picker_rec, TextFormat("%.2f", min_brush_size), TextFormat("%.2f", max_brush_size), &args->brush_size, min_brush_size, max_brush_size);
    DrawText("Brush Size", brush_size_picker_rec.x, brush_size_picker_rec.y - 20, 12, BLACK);

    Vector2 brush_size_display_rec = {
        .x = brush_size_picker_rec.x + brush_size_picker_rec.width + max_brush_size * 2 + 20,
        .y = brush_size_picker_rec.y + brush_size_picker_rec.height/2
    };
    DrawRectangleRoundedLines((Rectangle) {.x = brush_size_display_rec.x - max_brush_size, .y = brush_size_display_rec.y - max_brush_size, .width = max_brush_size*2, .height = max_brush_size*2}, 
            0.02, 4, 2, BLACK);
    DrawCircleV(brush_size_display_rec , args->brush_size, ColorFromHSV(args->brush_color.x, args->brush_color.y, args->brush_color.z));


    // Drawing Panel Options
    Rectangle draw_panel_clear_rec = {
        .x = draw_panel_rec.x,
        .y = draw_panel_rec.y + draw_panel_rec.height + 10,
        .width = 40,
        .height = 20,
    };
    if (GuiButton(draw_panel_clear_rec, "Clear")) {
        DrawingPanelClear(args);
    }

    Rectangle draw_panel_undo_rec = {
        .x = draw_panel_clear_rec.x + draw_panel_clear_rec.width + 10,
        .y = draw_panel_clear_rec.y,
        .width = 40,
        .height = 20,
    };
    if (GuiButton(draw_panel_undo_rec, "Undo")) {
        DrawingPanelUndo(args);
    }

    Rectangle draw_panel_redo_rec = {
        .x = draw_panel_undo_rec.x + draw_panel_undo_rec.width + 10,
        .y = draw_panel_undo_rec.y,
        .width = 40,
        .height = 20,
    };
    if (GuiButton(draw_panel_redo_rec, "Redo")) {
        DrawingPanelRedo(args);
    }


    bool is_draw = IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsGestureDetected(GESTURE_DRAG);
    bool is_erase = IsMouseButtonDown(MOUSE_BUTTON_RIGHT);
    bool just_off_panel = args->is_dragged && !CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);
    if (((is_draw || is_erase) && CheckCollisionPointRec(GetMousePosition(), draw_panel_rec)) || just_off_panel) {
        Color draw_color = ColorFromHSV(args->brush_color.x, args->brush_color.y, args->brush_color.z);
        if (is_erase) {
            draw_color = WHITE;
        }

        Vector2 size = {
            .x = args->brush_size,
            .y = args->brush_size,
        };
        Vector2 pos = GetMousePosition();
        pos.x -= size.x / 2.0;
        pos.y -= size.y / 2.0;

        pos.x -= draw_panel_rec.x;
        pos.y -= draw_panel_rec.y;

        pos.y = draw_texture.texture.height - pos.y; // texture drawing flips vertically because of opengl or smthn

        BeginTextureMode(draw_texture);
        {
            
            DrawCircleV(pos, args->brush_size, draw_color);
            
            // to connect points since the refresh rate is 60 fps which causes gaps
            if (args->prev_draw_pos.x != -1 && args->prev_draw_pos.y != -1) {
                DrawLineEx(args->prev_draw_pos, pos, args->brush_size * 2, draw_color);
                DrawCircleV(args->prev_draw_pos, args->brush_size, draw_color);
            }
            
            args->prev_draw_pos = pos;
        }
        EndTextureMode();

        args->is_drawing = true;
        args->updated = true;
    } else {
        // is not drawing
        args->prev_draw_pos = (Vector2) {.x = GetMousePosition().x - (args->brush_size/2) - draw_panel_rec.x, 
                .y = draw_texture.texture.height - (GetMousePosition().y - (args->brush_size/2) - draw_panel_rec.y)};
        

        // check if just stopped drawing, register segment
        if (args->is_drawing) {
            DrawingPanelAdd(args);
        }
        args->is_drawing = false;     
    }

    args->is_dragged = IsGestureDetected(GESTURE_DRAG) && CheckCollisionPointRec(GetMousePosition(), draw_panel_rec);

    DrawTexture(draw_texture.texture, draw_panel_rec.x + line_thickness/2, draw_panel_rec.y + line_thickness/2, WHITE);


    // Draw what the model will see
    Rectangle model_input_rec = {
        .x = draw_panel_rec.x - 10 - (120 / args->buffer_width) * args->buffer_width,
        .y = draw_panel_rec.y,
        .width = (120 / args->buffer_width) * args->buffer_width,
        .height = (120 / args->buffer_height) * args->buffer_height
    };

    RenderTexture2D input_texture = args->input_texture;

    

    BeginTextureMode(input_texture);
    {
        DrawTexturePro(draw_texture.texture, (Rectangle) {.x = 0, .y = 0, .width = draw_texture.texture.width, .height = -draw_texture.texture.height}, 
                (Rectangle) {.x = 0, .y = 0,. width = input_texture.texture.width, .height = input_texture.texture.height}, (Vector2) {0, 0}, 0, WHITE);
    }
    EndTextureMode();

    args->cur_frames++;
    if (args->updated && args->update_frames <= args->cur_frames) {
        args->cur_frames = 0;
        args->updated = false;
        Image input_image = LoadImageFromTexture(input_texture.texture);

        if (args->gray_scale) {
            ImageColorGrayscale(&input_image);
            Color *pixels = LoadImageColors(input_image);
            UpdateTexture(input_texture.texture, pixels);
            UnloadImageColors(pixels);
        }

        // read in model input into buffer
        for (int r = 0; r < input_image.height; r++) {
            for (int c = 0; c < input_image.width; c++) {
                Color color = GetImageColor(input_image, c, r);
                args->output_buffer[r * input_image.height + c] = color.r;
            }
        }
        UnloadImage(input_image);
        
        // TODO run the model on the new input
    }

    DrawTexturePro(input_texture.texture, (Rectangle) {.x = 0, .y = 0, .width = input_texture.texture.width, .height = input_texture.texture.height}, 
            model_input_rec, (Vector2) {0, 0}, 0, WHITE);

    // TODO draw the model's output
    
}