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
#define COST_FUNCTION DS_CROSS_ENTROPY
#define REGULARIZATION_PARAM 5.0f
#define EPOCHS 30
#define BATCH_SIZE 10
#define LEARNING_RATE 0.5f
#define TRAINED_NETWORK_PATH "trained_network.txt"

#define SCALING 20
#define WIN_HEIGHT (SCALING * PNG_WIDTH)
#define WIN_WIDTH WIN_HEIGHT
#define TARGET_FPS 60
#define DRAW_THICKNESS 40
#define MOUSE_POSITION_SIZE 4

static_assert(NUM_INPUTS * SCALING * SCALING == WIN_HEIGHT * WIN_WIDTH,
              "Scaling is wrong");

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

/// Converts the data of the raylib image to the same form as when the training
/// data png is loaded
void convert_img_to_training_data_structure(Image *const img) {
  ImageColorGrayscale(img);
  // NOTE: The following is needed such that the data is in the same order as
  // when a PNG is loaded
  ImageFlipHorizontal(img);
  ImageRotate(img, 180);

  ImageResize(img, PNG_WIDTH,
              PNG_WIDTH); // NOTE: Resize at the end, otherhise there will be
                          // some weird issue with the image placing
}

typedef struct {
  size_t height;
  size_t width;
  DS_FLOAT *data;
} Pixels;

void print_pixels(const Pixels *const pixels) {

  DS_PRINTF("╷");
  for (size_t i = 0; i < pixels->width * 4; ++i)
    DS_PRINTF("─");
  DS_PRINTF("─╷\n");

  for (size_t j = 0; j < pixels->height; ++j) {
    DS_PRINTF("│ ");
    for (size_t i = 0; i < pixels->width; ++i) {
      DS_PRINTF("%3.d ", 0);
    }
    DS_PRINTF("│\n");

    DS_PRINTF("│ ");
    for (size_t i = 0; i < pixels->height; ++i) {
      DS_PRINTF("%3.d ", (int)(pixels->data[i + pixels->width * j] * 255.f));
    }
    DS_PRINTF("│\n");
  }

  DS_PRINTF("╵");
  for (size_t i = 0; i < pixels->width * 4; ++i)
    DS_PRINTF("─");
  DS_PRINTF("─╵\n");
}

void crop_image_to_drawing_area(Image *img) {

  Color *const colors = LoadImageColors(*img);

  Rectangle bounds = {
      .x = img->width, .y = img->height, .width = 0, .height = 0};
  // Find the smallest rectangle that contains all non-white pixels
  for (int y = 0; y < img->height; ++y) {
    for (int x = 0; x < img->width; ++x) {
      const Color color = colors[y * img->width + x];
      if (color.r > 0 || color.g > 0 ||
          color.b > 0) { // NOTE: Assuming black background
        if (x < bounds.x)
          bounds.x = x;
        if (y < bounds.y)
          bounds.y = y;
        if (x > bounds.width)
          bounds.width = x; // NOTE: At this moment width is a coordinate
        if (y > bounds.height)
          bounds.height = y; // NOTE: At this moment height is a coordinate
      }
    }
  }

  if (bounds.width > bounds.x && bounds.height > bounds.y) {
    bounds.width = bounds.width - bounds.x +
                   1; // NOTE: Convert from coordinate to actual width
    bounds.height = bounds.height - bounds.y +
                    1; // NOTE: Convert from coordinate to actual height

    const int padding = DS_MIN(img->height, img->width) / 8;
    bounds.x = (bounds.x > padding) ? (bounds.x - padding) : 0;
    bounds.y = (bounds.y > padding) ? (bounds.y - padding) : 0;
    bounds.width = ((bounds.x + bounds.width + padding) < img->width)
                       ? (bounds.width + 2 * padding)
                       : (img->width - bounds.x);
    bounds.height = ((bounds.y + bounds.height + padding) < img->height)
                        ? (bounds.height + 2 * padding)
                        : (img->height - bounds.y);

    ImageCrop(img, bounds);
  }
  UnloadImageColors(colors);
}

/// Does all the transformation such that the pixel data is ready for the neural
/// network
Pixels load_pixels(Image *const img) {
  Pixels pixels = {0};
  pixels.data = DS_MALLOC(sizeof(*pixels.data) * img->height * img->width);
  DS_ASSERT(pixels.data, "Could not load pixels. Out of memory.");

  Image converted_image = ImageCopy(*img);
  crop_image_to_drawing_area(&converted_image);
  convert_img_to_training_data_structure(&converted_image);

  pixels.width = converted_image.width;
  pixels.height = converted_image.height;
  Color *const colors = LoadImageColors(converted_image);
  for (int i = 0; i < converted_image.height * converted_image.width; ++i)
    pixels.data[i] =
        (DS_FLOAT)colors[i]
            .r // NOTE: On an grayscale image all rgb values are the same
        / 255.f;

  UnloadImageColors(colors);
  UnloadImage(converted_image);

  return pixels;
}

void unload_pixels(Pixels pixels) { DS_FREE(pixels.data); }

void draw_text_centered_x(const char *text, const int y, const int font_size,
                          Color color) {
  const int text_len = MeasureText(text, font_size);
  DrawText(text, (WIN_WIDTH - text_len) / 2, y, font_size, color);
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
  bool clear = false;
  char out_text[MAX_OUTPUT_LABEL_STRLEN + 256] = {0};
  const Rectangle draw_boundary = {
      .x = (float)WIN_WIDTH / 6.f,
      .y = (float)WIN_HEIGHT / 6.f,
      .width = (float)WIN_WIDTH - 2.f * draw_boundary.x,
      .height = (float)WIN_HEIGHT - 2.f * draw_boundary.y};

  while (!WindowShouldClose() && !IsKeyPressed(KEY_Q)) {
    if (IsKeyPressed(KEY_R))
      clear = true;
    else
      clear = false;

    // --- DRAW NUMBER -- //
    BeginTextureMode(number_drawing_texture);
    if (clear)
      ClearBackground(BLACK);
    else if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
      // Draw the number on the texture
      if (predicted) {
        ClearBackground(BLACK);
        // TODO: Clean up rendering and resetting of out text
        out_text[0] = 0;
        predicted = false;
      }
      const Vector2 current_position = GetMousePosition();
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
      ClearBackground(BLACK);
      out_text[0] = 0;
    }

    DrawTextureRec(number_drawing_texture.texture,
                   (Rectangle){0, 0, WIN_WIDTH, -WIN_HEIGHT}, (Vector2){0, 0},
                   WHITE);

    if (IsKeyPressed(KEY_P)) {
      out_text[0] = 0;
      predicted = true;

      DS_Network *const network = DS_network_load(TRAINED_NETWORK_PATH);

      Image img = LoadImageFromTexture(number_drawing_texture.texture);
      const Pixels pixels = load_pixels(&img);
      UnloadImage(img);

      DS_ASSERT((size_t)pixels.width * (size_t)pixels.height ==
                    DS_network_input_layer_size(network),
                "PNG data size is not compatible with network input size.");

      char prediction[MAX_OUTPUT_LABEL_STRLEN + 1] = {0};
      DS_FLOAT prob = DS_network_predict(network, pixels.data, prediction);
      strncat(out_text, "It's a ", sizeof(out_text) - 1 - strlen(out_text));
      strncat(out_text, prediction, sizeof(out_text) - 1 - strlen(out_text));

      DS_PRINTF("Predicted %s with %.1f percent.\n", prediction, prob * 100);

      unload_pixels(pixels);
      DS_network_free(network);
    }
    if (*out_text != 0)
      draw_text_centered_x(out_text, 20, 50, WHITE);
    DrawRectangleRoundedLines(draw_boundary, 0.025, 1, 4, DARKBLUE);

    const int info_text_size = 18;
    const int info_text_y =
        draw_boundary.y + draw_boundary.height + draw_boundary.y / 6.f;
    draw_text_centered_x(
        "Draw a number in the rectangle and press P to predict.", info_text_y,
        info_text_size, WHITE);
    draw_text_centered_x("Press R to reset drawing.",
                         info_text_y + info_text_size, info_text_size, WHITE);
    EndDrawing();
  }
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
