#pragma once
#ifndef VISUTIL_H
#define VISUTIL_H

#include <raylib.h>

#include <stdarg.h>

void DrawCenteredText(const char* text, int centerx, int centery, int fontsize, Color fontcolor);
void DrawOutlinedText(const char *text, int posX, int posY, int fontSize, Color color, int outlineSize, Color outlineColor);
void DrawOutlinedCenteredText(const char* text, int posX, int posY, int fontSize, Color color, int outlineSize, Color outlineColor);

double sRGB_to_linear(double x);
double linear_to_sRGB(double y);
Color gray_scale(int r, int g, int b);

const char* concat(int count, ...);

#endif // VISUTIL_H