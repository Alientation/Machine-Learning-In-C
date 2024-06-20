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
#define _UNPACK_REC_CENTER(rec) rec.x + rec.width/2, rec.y + rec.height/2

#define _REC_FROM_DIM(_x,_y,_width,_height) (Rectangle) {.x = _x, .y = _y, .width = _width, .height = _height}

#define _TOGGLE_BOOL(bool_ptr) (*bool_ptr = !*bool_ptr)

void DrawCenteredText(const char* text, int center_x, int center_y, int font_size, Color font_color);
void DrawOutlinedText(const char *text, int pos_x, int pos_y, int font_size, Color color, int outline_size, Color outline_color);
void DrawOutlinedCenteredText(const char* text, int pos_x, int pos_y, int font_size, Color color, int outline_size, Color outline_color);
void DrawTextList(int count, int pos_x, int pos_y, int font_size, Color color, ...);

void DrawOutlinedRectangle(int x, int y, int width, int height, Color color, int outline_size, Color outline_color);
void DrawOutlinedRectangleRec(Rectangle rec, Color color, int outline_size, Color outline_color);

Vector2 Vec(float x, float y);
Vector2 RecPos(Rectangle rec);
Vector2 RecDim(Rectangle rec);
Vector2 RenderTextureDim(RenderTexture2D rtexture);
Vector2 VecAddC(Vector2 vec, float c);
Vector2 Vec2DExtend(float c);

Rectangle RecShift(Rectangle rec, int shift_x, int shift_y);
Rectangle RecShiftV(Rectangle rec, Vector2 shift);

Rectangle RecCenteredRecMargin(Rectangle window, int margin_x, int marginy);
Rectangle RecCenteredRecMarginV(Rectangle window, Vector2 margins);
Rectangle RecCenteredRecDim(Rectangle window, int width, int height);
Rectangle RecCenteredRecDimV(Rectangle window, Vector2 dims);
Rectangle RecCenteredMargin(int window_width, int window_height, int margin_x, int marginy);
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