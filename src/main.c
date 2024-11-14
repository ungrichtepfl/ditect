#include "deepsea.h"
#include "deepsea_file.h"
#include "deepsea_png.h"
#include "parser.h"
#include <assert.h>
#include <raylib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_LAYERS 3
/* #define NUM_INPUTS (28 * 28) */
/* #define NUM_OUTPUTS 10 */
#define NUM_INPUTS 3
#define NUM_OUTPUTS 2
#define WIN_HEIGHT (NUM_INPUTS / 2)
#define WIN_WIDTH WIN_HEIGHT
#define TARGET_FPS 60
#define NUM_TRAINING 2

void train(const char *const data_path) {

  DS_FILE_FileList *data_file_paths = DS_FILE_get_files(data_path);
  DS_ASSERT(data_file_paths->count > 0, "No files found.");

  DS_FILE_file_list_print_labelled(data_file_paths, 10);
  DS_FILE_FileList *random_slice = NULL;
  bool first = true;
  while ((random_slice = DS_FILE_get_random_bucket(data_file_paths, 3)) !=
         NULL) {
    if (first) {
      DS_PNG_Input *png_input = DS_PNG_input_load_grey(random_slice->paths[0]);
      DS_ASSERT(png_input, "Could not read png.");
      DS_PNG_input_print(png_input);
      DS_PNG_input_free(png_input);
      DS_FILE_file_list_print_labelled(random_slice, 0);
      first = false;
    }
  }
  DS_FILE_file_list_free(data_file_paths);

  size_t layer_sizes[NUM_LAYERS] = {NUM_INPUTS, 4, NUM_OUTPUTS};
  DS_FLOAT learing_rate = 0.01;
  char *output_labels[NUM_OUTPUTS] = {"out1", "out2"};
  DS_Backprop *backprop =
      DS_brackprop_create(layer_sizes, NUM_LAYERS, output_labels);

  DS_FLOAT *x = DS_randn(NUM_INPUTS);

  DS_FLOAT **ys = DS_MALLOC(NUM_TRAINING * sizeof(ys[0]));
  for (size_t i = 0; i < NUM_TRAINING; ++i) {
    ys[i] = DS_MALLOC(NUM_OUTPUTS * sizeof(DS_FLOAT));
    memset(ys[i], 0, NUM_OUTPUTS * sizeof(DS_FLOAT));
  }

  DS_FLOAT **xs = DS_MALLOC(NUM_TRAINING * sizeof(xs[0]));
  for (size_t i = 0; i < NUM_TRAINING; ++i) {
    xs[i] = x;
  }
  DS_Labelled_Inputs labelled_inputs = {
      .inputs = xs, .labels = ys, .count = NUM_TRAINING};

  DS_FLOAT cost = DS_backprop_network_cost(backprop, &labelled_inputs);
  DS_PRINTF("Cost of network BEFORE learning: %.2f\n", cost);
  for (int i = 0; i < 1; ++i)
    DS_backprop_learn_once(backprop, &labelled_inputs, learing_rate);
  cost = DS_backprop_network_cost(backprop, &labelled_inputs);
  DS_PRINTF("cost of network AFTER learing: %.2f\n", cost);

  DS_network_print_activation_layer(DS_backprop_network(backprop));
  char *saved_network_file = "network.txt";
  if (!DS_network_save(DS_backprop_network(backprop), saved_network_file)) {
    DS_PRINTF("Failed to save network!\n");
  }
  DS_Network *loaded_network = DS_network_load(saved_network_file);
  if (loaded_network) {
    DS_PRINTF("-------------ORIGINAL---------------\n");
    DS_network_print(DS_backprop_network(backprop));
    DS_PRINTF("-------------SAVED---------------\n");
    DS_network_print(loaded_network);
    DS_network_free(loaded_network);
  } else
    DS_ERROR("Could not load network.");

  DS_backprop_free(backprop);
  for (size_t i = 0; i < NUM_TRAINING; ++i) {
    DS_FREE(ys[i]);
  }
  DS_FREE(ys);
  DS_FREE(x);
  DS_FREE(xs);
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
