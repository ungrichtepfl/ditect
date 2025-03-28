#include "deepsea.h"
#include "deepsea_file.h"
#include "deepsea_png.h"
#include "deepsea_raylib.h"
#include "limits.h"
#include "parser.h"
#include <assert.h>
#include <errno.h>
#include <raylib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__EMSCRIPTEN__) || defined(__wasm__) || defined(__wasm32__) ||     \
    defined(__wasm64__)
#define WASM 1
#else
#define WASM 0
#endif

#define NUM_LAYERS 3
#define PNG_WIDTH 28
#define NUM_INPUTS (PNG_WIDTH * PNG_WIDTH)
#define NUM_OUTPUTS 10
#define COST_FUNCTION DS_CROSS_ENTROPY
#define REGULARIZATION_PARAM 5.0f
#define EPOCHS 30
#define BATCH_SIZE 10
#define LEARNING_RATE 0.5f
#define TRAINED_NETWORK_PATH "trained_network.txt"

#define FONT_FILE_PATH "./Lato-Regular.ttf"
#define SCALING 20
#define WIN_HEIGHT (SCALING * PNG_WIDTH)
#define WIN_WIDTH WIN_HEIGHT
#define TARGET_FPS 60
#define DRAW_THICKNESS 35
#define MOUSE_POSITION_SIZE 4
#define CROPPING_PADDING (WIN_HEIGHT / 8)
#define BACKGROUND_COLOR BLACK

static_assert(NUM_INPUTS * SCALING * SCALING == WIN_HEIGHT * WIN_WIDTH,
              "Scaling is wrong");

#if WASM

#include <stdatomic.h>
atomic_int_fast8_t mouse_down = 0;
atomic_int_fast32_t mouse_x = 0;
atomic_int_fast32_t mouse_y = 0;

void send_mouse_button_down(int32_t x, int32_t y) {
  mouse_x = x;
  mouse_y = y;
  mouse_down = 1;
}

void send_mouse_button_released(void) { mouse_down = 0; }

atomic_int_fast8_t space_pressed = 0;
void send_space_pressed(void) { space_pressed = 1; }

atomic_int_fast8_t rkey_pressed = 0;
void send_rkey_pressed(void) { rkey_pressed = 1; }
#endif

bool is_mouse_down(void) {
#if WASM
  return mouse_down > 0;
#else
  return IsMouseButtonDown(MOUSE_BUTTON_LEFT);
#endif
}

bool is_space_pressed(void) {
  bool pressed = IsKeyPressed(KEY_SPACE);
#if WASM
  if (space_pressed > 0) {
    space_pressed = 0; // NOTE: Reset
    pressed = true;
  }
#endif
  return pressed;
}

bool is_rkey_pressed(void) {
  bool pressed = IsKeyPressed(KEY_R);
#if WASM
  if (rkey_pressed > 0) {
    rkey_pressed = 0; // NOTE: Reset
    pressed = true;
  }
#endif
  return pressed;
}

Vector2 get_mouse_position(void) {
#if WASM
  return (Vector2){.x = (float)mouse_x, .y = (float)mouse_y};
#else
  return GetMousePosition();
#endif
}

void train(const char *const data_path) {
  DS_PRINTF("Start training. May take a while.\n");

  size_t layer_sizes[NUM_LAYERS] = {NUM_INPUTS, 100, NUM_OUTPUTS};
  char *output_labels[NUM_OUTPUTS] = {"0", "1", "2", "3", "4",
                                      "5", "6", "7", "8", "9"};
  DS_Backprop *backprop =
      DS_backprop_create(layer_sizes, NUM_LAYERS, output_labels, COST_FUNCTION,
                         REGULARIZATION_PARAM);

  DS_FILE_FileList *data_file_paths = DS_FILE_get_files(data_path);
  DS_ASSERT(data_file_paths->count > 0, "No files found.");
  for (int i = 0; i < EPOCHS; ++i) {
    DS_FILE_FileList *random_slice = NULL;
    while ((random_slice = DS_FILE_get_random_bucket(data_file_paths,
                                                     BATCH_SIZE)) != NULL) {
      DS_Labelled_Inputs *labelled_inputs = DS_PNG_file_list_to_labelled_inputs(
          random_slice, DS_backprop_network(backprop));
      DS_ASSERT(labelled_inputs, "Could not labelled inputs.");

      DS_backprop_learn_once(backprop, labelled_inputs, LEARNING_RATE,
                             data_file_paths->count);
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

  DS_Backprop *backprop = DS_backprop_create_from_network(
      network, COST_FUNCTION, REGULARIZATION_PARAM);

  DS_FLOAT cost = DS_backprop_network_cost(backprop, labelled_inputs);
  DS_PRINTF("Quadratic cost of network for testing set: %.2f\n", cost);
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

void draw_text_centered_x(Font font, const char *text, const int y,
                          const int font_size, Color color) {
  const Vector2 text_len = MeasureTextEx(font, text, font_size, 0);
  DrawTextEx(font, text, (Vector2){(WIN_WIDTH - text_len.x) / 2, y}, font_size,
             0, color);
}

void run_gui(void) {
  SetConfigFlags(FLAG_MSAA_4X_HINT);
  InitWindow(WIN_WIDTH, WIN_HEIGHT, "Ditect");
  SetTargetFPS(TARGET_FPS);
  const float thickness = DRAW_THICKNESS;
  Vector2 mouse_positions[MOUSE_POSITION_SIZE] = {0};
  size_t number_of_lines = 0;
  bool predicted = false;

  RenderTexture2D number_drawing_texture = // NOTE: Needed to extract pixels
      LoadRenderTexture(WIN_WIDTH, WIN_HEIGHT);
  BeginTextureMode(number_drawing_texture);
  ClearBackground(BACKGROUND_COLOR);
  EndTextureMode();

  bool clear = false;
  char out_text[MAX_OUTPUT_LABEL_STRLEN + 256] = {0};
  const Rectangle draw_boundary = {
      .x = (float)WIN_WIDTH / 6.f,
      .y = (float)WIN_HEIGHT / 6.f,
      .width = (float)WIN_WIDTH - 2.f * draw_boundary.x,
      .height = (float)WIN_HEIGHT - 2.f * draw_boundary.y};

  Font font = LoadFontEx(FONT_FILE_PATH, 512, NULL, 0);
#if WASM
  while (!WindowShouldClose()) {
#else
  while (!WindowShouldClose() && !IsKeyPressed(KEY_Q)) {
#endif
    if (is_rkey_pressed())
      clear = true;
    else
      clear = false;

    // --- DRAW NUMBER -- //
    BeginTextureMode(number_drawing_texture);
    if (clear)
      ClearBackground(BACKGROUND_COLOR);
    else if (is_mouse_down()) {
      // Draw the number on the texture
      if (predicted) {
        ClearBackground(BACKGROUND_COLOR);
        // TODO: Clean up rendering and resetting of out text
        out_text[0] = 0;
        predicted = false;
      }
      const Vector2 current_position = get_mouse_position();
      if (CheckCollisionPointRec(current_position, draw_boundary)) {
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
      }
    } else {
      number_of_lines = 0;
    }
    EndTextureMode();
    // --- END DRAW NUMBER -- //

    BeginDrawing();
    if (clear) {
      ClearBackground(BACKGROUND_COLOR);
      out_text[0] = 0;
    }

    DrawTextureRec(number_drawing_texture.texture,
                   (Rectangle){0, 0, WIN_WIDTH, -WIN_HEIGHT}, (Vector2){0, 0},
                   WHITE);

    if (is_space_pressed()) {
      out_text[0] = 0;
      predicted = true;

      DS_Network *const network = DS_network_load(TRAINED_NETWORK_PATH);

      Image img = LoadImageFromTexture(number_drawing_texture.texture);
      const DS_PixelsBW pixels = DS_RAYLIB_load_pixels_bw_from_image(
          &img, PNG_WIDTH, PNG_WIDTH, CROPPING_PADDING, BACKGROUND_COLOR);
      UnloadImage(img);

      if (!DS_empty_pixels(&pixels)) {

        DS_ASSERT((size_t)pixels.width * (size_t)pixels.height ==
                      DS_network_input_layer_size(network),
                  "PNG data size is not compatible with network input size.");

        char prediction[MAX_OUTPUT_LABEL_STRLEN + 1] = {0};
        DS_FLOAT prob = DS_network_predict(network, pixels.data, prediction);
        strncat(out_text, "It's a ", sizeof(out_text) - 1 - strlen(out_text));
        strncat(out_text, prediction, sizeof(out_text) - 1 - strlen(out_text));

        DS_PRINTF("Predicted %s with %.1f percent.\n", prediction, prob * 100);

        DS_unload_pixels(pixels);
        DS_network_free(network);
      } else {
        strncat(out_text, "Nothing to predict!",
                sizeof(out_text) - 1 - strlen(out_text));
        DS_PRINTF("Nothing to predict\n");
      }
    }
    if (*out_text != 0)
      draw_text_centered_x(font, out_text, 20, 50, WHITE);
    DrawRectangleRoundedLines(draw_boundary, 0.025, 1, 4, DARKBLUE);

    const int info_text_size = 22;
    const int info_text_y =
        draw_boundary.y + draw_boundary.height + draw_boundary.y / 6.f;
    draw_text_centered_x(font,
                         "Draw a number in the rectangle and press SPACE.",
                         info_text_y, info_text_size, WHITE);
    draw_text_centered_x(font, "Press R to reset the drawing.",
                         info_text_y + info_text_size, info_text_size, WHITE);
#if !WASM
    draw_text_centered_x(font, "Press Q to quit.",
                         info_text_y + 2 * info_text_size, info_text_size,
                         WHITE);
#endif
    EndDrawing();
  }
  UnloadFont(font);
  UnloadRenderTexture(number_drawing_texture);
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
