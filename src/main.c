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
#define NUM_INPUTS (28 * 28)
#define NUM_OUTPUTS 10
#define BATCH_SIZE 20
#define WIN_HEIGHT (NUM_INPUTS / 2)
#define WIN_WIDTH WIN_HEIGHT
#define TARGET_FPS 60

#define TRAINED_NETWORK_PATH "trained_network.txt"

void train(const char *const data_path) {

  size_t layer_sizes[NUM_LAYERS] = {NUM_INPUTS, 100, NUM_OUTPUTS};
  DS_FLOAT learing_rate = 0.1;
  char *output_labels[NUM_OUTPUTS] = {"0", "1", "2", "3", "4",
                                      "5", "6", "7", "8", "9"};
  DS_Backprop *backprop =
      DS_brackprop_create(layer_sizes, NUM_LAYERS, output_labels);

  DS_FILE_FileList *data_file_paths = DS_FILE_get_files(data_path);
  DS_ASSERT(data_file_paths->count > 0, "No files found.");

  DS_FILE_FileList *random_slice = NULL;
  while ((random_slice =
              DS_FILE_get_random_bucket(data_file_paths, BATCH_SIZE)) != NULL) {
    DS_Labelled_Inputs *labelled_inputs = DS_PNG_file_list_to_labelled_inputs(
        random_slice, DS_backprop_network(backprop));
    DS_backprop_learn_once(backprop, labelled_inputs, learing_rate);
    DS_FLOAT cost = DS_backprop_network_cost(backprop, labelled_inputs);
    DS_PRINTF("cost of network AFTER learing: %.2f\n", cost);
    DS_labelled_inputs_free(labelled_inputs);
  }
  DS_FILE_file_list_free(data_file_paths);

  if (!DS_network_save(DS_backprop_network(backprop), TRAINED_NETWORK_PATH)) {
    DS_PRINTF("Failed to save network!\n");
  }

  DS_backprop_free(backprop);
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
