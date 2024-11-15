#include "deepsea.h"
#include "deepsea_file.h"
#include "deepsea_png.h"
#include "parser.h"
#include <assert.h>
#include <errno.h>
#include <raylib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_LAYERS 3
#define NUM_INPUTS (28 * 28)
#define NUM_OUTPUTS 10
#define EPOCHS 30
#define BATCH_SIZE 10
#define LEARNING_RATE 3.
#define WIN_HEIGHT (NUM_INPUTS / 2)
#define WIN_WIDTH WIN_HEIGHT
#define TARGET_FPS 60

#define TRAINED_NETWORK_PATH "trained_network.txt"

void train(const char *const data_path) {
  DS_PRINTF("Start training. May take a while.\n");

  size_t layer_sizes[NUM_LAYERS] = {NUM_INPUTS, 100, NUM_OUTPUTS};
  char *output_labels[NUM_OUTPUTS] = {"0", "1", "2", "3", "4",
                                      "5", "6", "7", "8", "9"};
  DS_Backprop *backprop =
      DS_brackprop_create(layer_sizes, NUM_LAYERS, output_labels);

  DS_FILE_FileList *data_file_paths = DS_FILE_get_files(data_path);
  DS_ASSERT(data_file_paths->count > 0, "No files found.");
  for (int i = 0; i < EPOCHS; ++i) {
    DS_FILE_FileList *random_slice = NULL;
    while ((random_slice = DS_FILE_get_random_bucket(data_file_paths,
                                                     BATCH_SIZE)) != NULL) {
      DS_Labelled_Inputs *labelled_inputs = DS_PNG_file_list_to_labelled_inputs(
          random_slice, DS_backprop_network(backprop));
      DS_ASSERT(labelled_inputs, "Could not labelled inputs.");

      DS_backprop_learn_once(backprop, labelled_inputs, LEARNING_RATE);
      DS_FLOAT cost = DS_backprop_network_cost(backprop, labelled_inputs);
      DS_PRINTF("Cost of network AFTER learing: %.2f\n", cost);
      DS_labelled_inputs_free(labelled_inputs);
    }
  }

  DS_FILE_file_list_free(data_file_paths);

  if (!DS_network_save(DS_backprop_network(backprop), TRAINED_NETWORK_PATH)) {
    DS_PRINTF("Failed to save network!\n");
  }

  DS_backprop_free(backprop);
}

void test(const char *const data_path) {
  DS_Network *network = DS_network_load(TRAINED_NETWORK_PATH);
  DS_FILE_FileList *data_file_paths = DS_FILE_get_files(data_path);
  DS_ASSERT(data_file_paths->count > 0, "No files found.");
  DS_Labelled_Inputs *labelled_inputs =
      DS_PNG_file_list_to_labelled_inputs(data_file_paths, network);
  DS_ASSERT(labelled_inputs, "Could not labelled inputs.");

  DS_FLOAT cost = DS_network_cost(network, labelled_inputs);
  DS_PRINTF("Cost of network for testing set: %.2f\n", cost);
  DS_labelled_inputs_free(labelled_inputs);
  DS_FILE_file_list_free(data_file_paths);
  DS_network_free(network);
}

void predict(const char *const data_path) {

  DS_Network *network = DS_network_load(TRAINED_NETWORK_PATH);
  DS_PNG_Input *png_input = DS_PNG_input_load_grey(data_path);
  DS_ASSERT(png_input, "Could not load png input for file \"%s\"", data_path);
  errno = 0;
  size_t label = DS_FILE_get_label_from_directory_name(data_path);
  DS_ASSERT(!(label == 0 && errno != 0), "Could not load label for file \"%s\"",
            data_path);

  DS_ASSERT(png_input->width * png_input->height ==
                DS_network_input_layer_size(network),
            "PNG data size is not compatible with network input size.");

  DS_network_print_prediction(network, png_input->data);
  DS_PRINTF("Correct label: %lu\n", label);

  DS_PNG_input_free(png_input);
  DS_network_free(network);
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
    test(cmd.data_path);

  } break;
  case CLA_TRAINING: {
    train(cmd.data_path);
  } break;

  case CLA_PREDICT: {
    predict(cmd.data_path);
  } break;
  default:
    assert(false && "Unreachable!");
  }
}
