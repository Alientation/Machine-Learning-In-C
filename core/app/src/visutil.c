#include <app/visutil.h>

#include <math.h>

void DrawCenteredText(const char* text, int centerx, int centery, int fontsize, Color fontcolor) {
    int width = MeasureText(text, fontsize);
    DrawText(text, centerx - width / 2, centery - fontsize/2, fontsize, fontcolor);
}

void DrawOutlinedText(const char *text, int posX, int posY, int fontSize, Color color, int outlineSize, Color outlineColor) {
    DrawText(text, posX - outlineSize, posY - outlineSize, fontSize, outlineColor);
    DrawText(text, posX + outlineSize, posY - outlineSize, fontSize, outlineColor);
    DrawText(text, posX - outlineSize, posY + outlineSize, fontSize, outlineColor);
    DrawText(text, posX + outlineSize, posY + outlineSize, fontSize, outlineColor);
    DrawText(text, posX, posY, fontSize, color);
}

void DrawOutlinedCenteredText(const char* text, int posX, int posY, int fontSize, Color color, int outlineSize, Color outlineColor) {
    DrawCenteredText(text, posX - outlineSize, posY - outlineSize, fontSize, outlineColor);
    DrawCenteredText(text, posX + outlineSize, posY - outlineSize, fontSize, outlineColor);
    DrawCenteredText(text, posX - outlineSize, posY + outlineSize, fontSize, outlineColor);
    DrawCenteredText(text, posX + outlineSize, posY + outlineSize, fontSize, outlineColor);
    DrawCenteredText(text, posX, posY, fontSize, color);
}

void DrawOutlinedRectangle(int x, int y, int width, int height, Color color, int outlineSize, Color outlineColor) {
    DrawRectangle(x, y, width, height, color);
    DrawRectangleLines(x - outlineSize/2.0, y - outlineSize/2.0, width, height, outlineColor);
}

void DrawOutlinedRectangleRec(Rectangle rec, Color color, int outlineSize, Color outlineColor) {
    DrawOutlinedRectangle(rec.x, rec.y, rec.width, rec.height, color, outlineSize, outlineColor);
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


Rectangle RecCenteredRecMargin(Rectangle window, int marginx, int marginy) {
    return RecOffsetV(RecCenteredMarginV(RecDim(window), (Vector2) {.x = marginx, .y = marginy}), RecPos(window));
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

Rectangle RecCenteredMargin(int window_width, int window_height, int marginx, int marginy) {
    return (Rectangle) {.x = marginx, .y = marginy, .width = window_width - 2 * marginx, .height = window_height - 2 * marginy};
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