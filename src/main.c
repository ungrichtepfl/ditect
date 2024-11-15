#include "deepsea.h"
#include "deepsea_file.h"
#include "deepsea_png.h"
#include "limits.h"
#include "parser.h"
#include <assert.h>
#include <errno.h>
#include <raylib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_LAYERS 3
#define PNG_WIDTH 28
#define NUM_INPUTS (PNG_WIDTH * PNG_WIDTH)
#define NUM_OUTPUTS 10
#define EPOCHS 30
#define BATCH_SIZE 10
#define LEARNING_RATE 3.
#define SCALING 20
#define WIN_HEIGHT (SCALING * PNG_WIDTH)
#define WIN_WIDTH WIN_HEIGHT
#define TARGET_FPS 30
#define MIN_PIXEL_AFTER_RESIZE 2

static_assert(NUM_INPUTS * SCALING * SCALING == WIN_HEIGHT * WIN_WIDTH,
              "Scaling is wrong");

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

#define MOUSE_POSITION_SIZE 4
void run_gui(void) {
  SetConfigFlags(FLAG_MSAA_4X_HINT);
  InitWindow(WIN_WIDTH, WIN_HEIGHT, "Ditect");
  SetTargetFPS(TARGET_FPS);
  float thickness = MIN_PIXEL_AFTER_RESIZE * SCALING;
  Vector2 mouse_positions[MOUSE_POSITION_SIZE] = {0};
  size_t number_of_lines = 0;
  bool predicted = false;

  while (!WindowShouldClose() && !IsKeyPressed(KEY_Q)) {

    BeginDrawing();

    if (IsKeyPressed(KEY_R)) {
      ClearBackground(BLACK);
    }

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
      if (predicted) {
        ClearBackground(BLACK);
        predicted = false;
      }
      const Vector2 current_position = GetMousePosition();
      number_of_lines++;
      if (number_of_lines > MOUSE_POSITION_SIZE) {
        if (MOUSE_POSITION_SIZE > 1)
          memmove(&mouse_positions[0], &mouse_positions[1],
                  (MOUSE_POSITION_SIZE - 1) * sizeof(current_position));
        number_of_lines = MOUSE_POSITION_SIZE;
      }
      memcpy(&mouse_positions[number_of_lines - 1], &current_position,
             sizeof(current_position));
      DrawSplineBasis(mouse_positions, number_of_lines, thickness, WHITE);
    } else {
      number_of_lines = 0;
    }
    if (IsKeyPressed(KEY_P)) {
      predicted = true;
      char *to_predict_big_file_name = "to_predict_big.png";
      char *to_predict_file_name = "to_predict.png";
      TakeScreenshot(to_predict_big_file_name);
      char cmd[256] = {0};
      snprintf(cmd, 256, "convert -resize %dx%d %s %s", PNG_WIDTH, PNG_WIDTH,
               to_predict_big_file_name, to_predict_file_name);
      DS_ASSERT(system(cmd) == 0, "Could not generate PNG.");

      DS_Network *network = DS_network_load(TRAINED_NETWORK_PATH);
      DS_PNG_Input *png_input = DS_PNG_input_load_grey(to_predict_file_name);
      DS_ASSERT(png_input, "Could not load png input for file \"%s\"",
                to_predict_file_name);

      DS_ASSERT(png_input->width * png_input->height ==
                    DS_network_input_layer_size(network),
                "PNG data size is not compatible with network input size.");

      char prediction[MAX_OUTPUT_LABEL_STRLEN + 1] = {0};

      DS_FLOAT prob = DS_network_predict(network, png_input->data, prediction);
      char out_text[MAX_OUTPUT_LABEL_STRLEN + 256] = "It's a ";

      strncat(out_text, prediction, MAX_OUTPUT_LABEL_STRLEN + 256);

      DrawText(out_text, 190, 20, 50, WHITE);

      DS_PRINTF("Predicted %s with %.1f percent.\n", prediction, prob * 100);

      DS_PNG_input_free(png_input);
      DS_network_free(network);
    }
    EndDrawing();
  }
  CloseWindow();
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
