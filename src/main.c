#include <raylib.h>

#define WIN_HEIGHT 400
#define WIN_WIDTH 600
#define TARGET_FPS 60

int main(void) {

  InitWindow(WIN_WIDTH, WIN_HEIGHT, "Ditect");
  SetTargetFPS(TARGET_FPS);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    EndDrawing();
  }
}
