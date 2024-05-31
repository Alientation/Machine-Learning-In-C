#pragma once
#ifndef VISUTIL_H
#define VISUTIL_H

#include <raylib.h>

void DrawCenteredText(const char* text, int centerx, int centery, int fontsize, Color fontcolor);
void DrawOutlinedText(const char *text, int posX, int posY, int fontSize, Color color, int outlineSize, Color outlineColor);
void DrawOutlinedCenteredText(const char* text, int posX, int posY, int fontSize, Color color, int outlineSize, Color outlineColor);



#endif // VISUTIL_H