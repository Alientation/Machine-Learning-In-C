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