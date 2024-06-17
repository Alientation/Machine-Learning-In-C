#pragma once
#ifndef VISUTIL_H
#define VISUTIL_H

#include <raylib.h>

#include <stdarg.h>

#define _UNPACK_VEC2(vec2d) vec2d.x, vec2d.y
#define _UNPACK_VEC3(vec3d) vec3d.x, vec3d.y, vec3d.z
#define _UNPACK_REC_POS(rec) rec.x, rec.y
#define _UNPACK_REC_DIM(rec) rec.width, rec.height
#define _UNPACK_REC(rec) rec.x, rec.y, rec.width, rec.height

#define _REC_FROM_DIM(_x,_y,_width,_height) (Rectangle) {.x = _x, .y = _y, .width = _width, .height = _height}

void DrawCenteredText(const char* text, int centerx, int centery, int fontsize, Color fontcolor);
void DrawOutlinedText(const char *text, int posX, int posY, int fontSize, Color color, int outlineSize, Color outlineColor);
void DrawOutlinedCenteredText(const char* text, int posX, int posY, int fontSize, Color color, int outlineSize, Color outlineColor);

void DrawOutlinedRectangle(int x, int y, int width, int height, Color color, int outlineSize, Color outlineColor);
void DrawOutlinedRectangleRec(Rectangle rec, Color color, int outlineSize, Color outlineColor);

Vector2 RecPos(Rectangle rec);
Vector2 RecDim(Rectangle rec);
Vector2 RenderTextureDim(RenderTexture2D rtexture);
Vector2 VecAddC(Vector2 vec, float c);
Vector2 Vec2DExtend(float c);

Rectangle RecCenteredRecMargin(Rectangle window, int marginx, int marginy);
Rectangle RecCenteredRecMarginV(Rectangle window, Vector2 margins);
Rectangle RecCenteredRecDim(Rectangle window, int width, int height);
Rectangle RecCenteredRecDimV(Rectangle window, Vector2 dims);
Rectangle RecCenteredMargin(int window_width, int window_height, int marginx, int marginy);
Rectangle RecCenteredMarginV(Vector2 window_dim, Vector2 margins);
Rectangle RecCenteredDim(int window_width, int window_height, int width, int height);
Rectangle RecCenteredDimV(Vector2 window_dim, Vector2 dims);
Rectangle RecOffset(Rectangle rec, int offsetx, int offsety);
Rectangle RecOffsetV(Rectangle rec, Vector2 offsets);

double sRGB_to_linear(double x);
double linear_to_sRGB(double y);
Color gray_scale(int r, int g, int b);

const char* concat(int count, ...);


#endif // VISUTIL_H