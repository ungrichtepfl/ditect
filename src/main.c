#include "deepsea.h"
#include "deepsea_png.h"
#include "parser.h"
#include <assert.h>
#include <raylib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_LAYERS 3
#define NUM_INPUTS (28 * 28)
#define NUM_OUTPUTS 10
#define WIN_HEIGHT (NUM_INPUTS / 2)
#define WIN_WIDTH WIN_HEIGHT
#define TARGET_FPS 60
#define NUM_TRAINING 2

void train(const char *const data_path) {

  DS_PNG_Input *png_input = DS_PNG_input_load_grey(data_path);
  if (!png_input) {
    DS_EXIT(1);
  }
  DS_PNG_input_print(png_input);
  DS_PNG_input_free(png_input);

  size_t layer_sizes[NUM_LAYERS] = {NUM_INPUTS, 100, NUM_OUTPUTS};
  DS_FLOAT learing_rate = 0.01;
  DS_Backprop *backprop = DS_brackprop_create(layer_sizes, NUM_LAYERS, NULL);

  // DS_network_print(DS_backprop_network(backprop));

  DS_FLOAT *x = DS_randn(NUM_INPUTS);

  DS_FLOAT **ys = alloca(NUM_TRAINING * sizeof(ys[0]));
  for (size_t i = 0; i < NUM_TRAINING; ++i) {
    ys[i] = alloca(NUM_OUTPUTS * sizeof(DS_FLOAT));
    memset(ys[i], 0, NUM_OUTPUTS * sizeof(DS_FLOAT));
  }

  DS_FLOAT **xs = alloca(NUM_TRAINING * sizeof(xs[0]));
  for (size_t i = 0; i < NUM_TRAINING; ++i) {
    xs[i] = x;
  }

  DS_FLOAT cost = DS_backprop_network_cost(backprop, xs, ys, NUM_TRAINING);
  DS_PRINTF("Cost of network BEFORE learning: %.2f\n", cost);
  for (int i = 0; i < 1; ++i)
    DS_backprop_learn_once(backprop, xs, ys, NUM_TRAINING, learing_rate);
  cost = DS_backprop_network_cost(backprop, xs, ys, NUM_TRAINING);
  DS_PRINTF("cost of network AFTER learing: %.2f\n", cost);

  DS_network_print_activation_layer(DS_backprop_network(backprop));

  DS_backprop_free(backprop);
  DS_FREE(x);
}

void run_gui(void) {
  InitWindow(WIN_WIDTH, WIN_HEIGHT, "Ditect");
  SetTargetFPS(TARGET_FPS);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    EndDrawing();
  }
}

int main(int argc, char *argv[]) {
  CommandLineArgs cmd = {0};
  command_line_parse(&cmd, argc, argv);
  switch (cmd.action) {
  case CLA_GUI: {
    run_gui();

  } break;
  case CLA_TESTING: {
    // TODO:

  } break;
  case CLA_TRAINING: {
    train(cmd.data_path);
  } break;
  default:
    assert(false && "Unreachable!");
  }
}
