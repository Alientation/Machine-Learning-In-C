#include <app/visutil.h>

#include <math.h>

void DrawCenteredText(const char* text, int center_x, int center_y, int font_size, Color font_color) {
    int width = MeasureText(text, font_size);
    DrawText(text, center_x - width / 2, center_y - font_size/2, font_size, font_color);
}

void DrawOutlinedText(const char *text, int pos_x, int pos_y, int font_size, Color color, int outline_size, Color outline_color) {
    DrawText(text, pos_x - outline_size, pos_y - outline_size, font_size, outline_color);
    DrawText(text, pos_x + outline_size, pos_y - outline_size, font_size, outline_color);
    DrawText(text, pos_x - outline_size, pos_y + outline_size, font_size, outline_color);
    DrawText(text, pos_x + outline_size, pos_y + outline_size, font_size, outline_color);
    DrawText(text, pos_x, pos_y, font_size, color);
}

void DrawOutlinedCenteredText(const char* text, int pos_x, int pos_y, int font_size, Color color, int outline_size, Color outline_color) {
    DrawCenteredText(text, pos_x - outline_size, pos_y - outline_size, font_size, outline_color);
    DrawCenteredText(text, pos_x + outline_size, pos_y - outline_size, font_size, outline_color);
    DrawCenteredText(text, pos_x - outline_size, pos_y + outline_size, font_size, outline_color);
    DrawCenteredText(text, pos_x + outline_size, pos_y + outline_size, font_size, outline_color);
    DrawCenteredText(text, pos_x, pos_y, font_size, color);
}

void DrawOutlinedRectangle(int x, int y, int width, int height, Color color, int outline_size, Color outline_color) {
    DrawRectangle(x, y, width, height, color);
    DrawRectangleLines(x - outline_size/2.0, y - outline_size/2.0, width, height, outline_color);
}

void DrawOutlinedRectangleRec(Rectangle rec, Color color, int outline_size, Color outline_color) {
    DrawOutlinedRectangle(rec.x, rec.y, rec.width, rec.height, color, outline_size, outline_color);
}


Vector2 Vec(float x, float y) {
    return (Vector2) {.x = x, .y = y};
}

Vector2 RecPos(Rectangle rec) {
    return (Vector2) {.x = rec.x, .y = rec.y};
}

Vector2 RecDim(Rectangle rec) {
    return (Vector2) {.x = rec.width, .y = rec.height};
}

Vector2 RenderTextureDim(RenderTexture2D rtexture) {
    return (Vector2) {.x = rtexture.texture.width, .y = rtexture.texture.height};
}

Vector2 VecAddC(Vector2 vec, float c) {
    return (Vector2) {.x = vec.x + c, .y = vec.y + c};
}

Vector2 Vec2DExtend(float c) {
    return (Vector2) {.x = c, .y = c};
}


Rectangle RecShift(Rectangle rec, int shift_x, int shift_y) {
    rec.x += shift_x;
    rec.y += shift_y;
    return rec;
}

Rectangle RecShiftV(Rectangle rec, Vector2 shift) {
    return RecShift(rec, shift.x, shift.y);
}


Rectangle RecCenteredRecMargin(Rectangle window, int margin_x, int margin_y) {
    return RecOffsetV(RecCenteredMarginV(RecDim(window), (Vector2) {.x = margin_x, .y = margin_y}), RecPos(window));
}

Rectangle RecCenteredRecMarginV(Rectangle window, Vector2 margins) {
    return RecCenteredRecMargin(window, _UNPACK_VEC2(margins));
}

Rectangle RecCenteredRecDim(Rectangle window, int width, int height) {
    return RecOffsetV(RecCenteredDimV(RecDim(window), (Vector2) {.x = width, .y = height}), RecPos(window));
}

Rectangle RecCenteredRecDimV(Rectangle window, Vector2 dims) {
    return RecCenteredRecDim(window, _UNPACK_VEC2(dims));
}

Rectangle RecCenteredMargin(int window_width, int window_height, int margin_x, int margin_y) {
    return (Rectangle) {.x = margin_x, .y = margin_y, .width = window_width - 2 * margin_x, .height = window_height - 2 * margin_y};
}

Rectangle RecCenteredMarginV(Vector2 window_dim, Vector2 margins) {
    return RecCenteredMargin(_UNPACK_VEC2(window_dim), _UNPACK_VEC2(margins));
}

Rectangle RecCenteredDim(int window_width, int window_height, int width, int height) {
    return (Rectangle) {.x = window_width/2 - width/2, .y = window_height/2 - height/2, .width = width, .height = height};
}

Rectangle RecCenteredDimV(Vector2 window_dim, Vector2 dims) {
    return RecCenteredDim(_UNPACK_VEC2(window_dim), _UNPACK_VEC2(dims));
}

Rectangle RecOffset(Rectangle rec, int offsetx, int offsety) {
    return (Rectangle) {.x = rec.x + offsetx, .y = rec.y + offsety, .width = rec.width, .height = rec.height};
}

Rectangle RecOffsetV(Rectangle rec, Vector2 offsets) {
    return RecOffset(rec, _UNPACK_VEC2(offsets));
}


double sRGB_to_linear(double x) {
    if (x < 0.04045) return x/12.92;
    return pow((x + 0.055) / 1.055, 2.4);
}

double linear_to_sRGB(double y) {
    if (y <= 0.0031308) return 12.92 * y;
    return 1.055 * pow(y, 1/2.4) - 0.055;
}

Color gray_scale(int r, int g, int b) {
    double gray_linear = 0.2126 * sRGB_to_linear(r / 255.0) + 0.7152 * sRGB_to_linear(g / 255.0) + 0.0722 * sRGB_to_linear(b / 255.0);
    int gray_color = round(linear_to_sRGB(gray_linear) * 255);
    return (Color) {.a = 255, .r = gray_color, .g = gray_color, .b = gray_color};
}

const char* concat(int count, ...) {
    const char* array[count];
    va_list ptr;
    va_start(ptr, count);
    for (int i = 0; i < count; i++) {
        array[i] = va_arg(ptr, char*);
    }
    va_end(ptr);

    return TextJoin(array, count, "");
}