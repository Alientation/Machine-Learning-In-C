#include <app/visutil.h>

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