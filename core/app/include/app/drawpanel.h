#pragma once
#ifndef DRAWPANEL_H
#define DRAWPANEL_H

#include <app/visualizer.h>

#include <raylib.h>
#include <app/raygui.h>


void DrawingPanelFreeHistory(drawing_panel_args_t *args);
void DrawingPanelUndo(drawing_panel_args_t *args);
void DrawingPanelRedo(drawing_panel_args_t *args);
void DrawingPanelAdd(drawing_panel_args_t *args);
void DrawingPanelClear(drawing_panel_args_t *args);

void GuiDrawingPanelPopup(drawing_panel_args_t *args);

#endif // DRAWPANEL_H