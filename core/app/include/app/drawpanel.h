#pragma once
#ifndef DRAWPANEL_H
#define DRAWPANEL_H

#include <app/visualizer.h>

#include <raylib.h>
#include <app/raygui.h>


void DrawingPanelFreeHistory(drawing_panel_args_t *draw_args);
void DrawingPanelUndo(drawing_panel_args_t *draw_args);
void DrawingPanelRedo(drawing_panel_args_t *draw_args);
void DrawingPanelAdd(drawing_panel_args_t *draw_args);
void DrawingPanelClear(drawing_panel_args_t *draw_args);

void GuiDrawingPanelPopup(drawing_panel_args_t *draw_args);

#endif // DRAWPANEL_H