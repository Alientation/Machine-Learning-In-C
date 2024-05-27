#include <app/visualizer.h>

#include <raylib.h>
#include <time.h>

static bool is_window_open = false;

void* window_run(void *vargp) {
    assert(!is_window_open);

    neural_network_model_t *model = (neural_network_model_t*) vargp;

    const int screenWidth = 800;
    const int screenHeight = 450;
    is_window_open = true;

    InitWindow(screenWidth, screenHeight, "model visualizer");
    SetTargetFPS(60);

    window_keep_open(model, 0);
}

void window_draw(neural_network_model_t *model) {
    BeginDrawing();
    {
        ClearBackground(RAYWHITE);
        DrawText("TODO", 190, 200, 20, LIGHTGRAY);
    }
    EndDrawing();
}

void window_close() {
    CloseWindow();
}

void window_keep_open(neural_network_model_t *model, unsigned int num_seconds) {
    if (num_seconds == 0) {
        num_seconds = ~0;
    }

    time_t now = clock();
    unsigned long long num_ms = num_seconds * 1000L;
    while (!WindowShouldClose() && clock() - now < num_ms) {
        window_draw(model);
    }
    is_window_open = false;
}

