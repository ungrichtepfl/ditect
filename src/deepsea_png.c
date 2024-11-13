#include "deepsea_png.h"

#include <assert.h>
#include <png.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PNG_GRAY_VALUE 255.
struct DS_PNG_Input {
  DS_Input *input;
  size_t width;
  size_t height;
  DS_PNG_Type type;
};

typedef struct DS_PNG_Input DS_PNG_Input;

DS_PNG_Input *DS_PNG_input_load_grey(const char *const png_image_path) {
  png_image image;
  memset(&image, 0, sizeof(image));
  image.version = PNG_IMAGE_VERSION;

  if (!png_image_begin_read_from_file(&image, png_image_path)) {
    DS_ERROR("%s", image.message);
    return NULL;
  }

  image.format = PNG_FORMAT_GRAY;
  const size_t image_size = PNG_IMAGE_SIZE(image);
  png_bytep const buffer = DS_MALLOC(image_size);
  if (!buffer) {
    png_image_free(&image);
    DS_ASSERT(false, "Could not allocate input buffer, out of memory.");
  }
  if (!png_image_finish_read(&image, NULL, buffer, 0, NULL)) {
    DS_ERROR("%s", image.message);
    DS_FREE(buffer);
    return NULL;
  }

  DS_FLOAT *const in = DS_MALLOC(image_size * sizeof(in[0]));
  for (size_t i = 0; i < image_size; ++i)
    in[i] = (DS_FLOAT)buffer[i] / MAX_PNG_GRAY_VALUE;
  DS_FREE(buffer);

  DS_Input *const input = DS_MALLOC(sizeof(*input));
  DS_ASSERT(input, "Could not allocate input struct, out of memory.");
  input->in = in;
  input->len = image_size;

  DS_PNG_Input *png_input = DS_MALLOC(sizeof(*png_input));
  png_input->input = input;
  png_input->width = image.width;
  png_input->height = image.height;
  png_input->type = DS_PNG_Gray;

  return png_input;
}
void DS_PNG_input_print(const DS_PNG_Input *const png_input) {
  switch (png_input->type) {
  case DS_PNG_Gray: {

    DS_PRINTF("╷");
    for (size_t i = 0; i < png_input->width * 4; ++i)
      DS_PRINTF("─");
    DS_PRINTF("─╷\n");

    for (size_t j = 0; j < png_input->height; ++j) {
      DS_PRINTF("│ ");
      for (size_t i = 0; i < png_input->width; ++i) {
        DS_PRINTF("%3.d ", 0);
      }
      DS_PRINTF("│\n");

      DS_PRINTF("│ ");
      for (size_t i = 0; i < png_input->height; ++i) {
        DS_PRINTF("%3.d ",
                  (int)(png_input->input->in[i + png_input->width * j] *
                        MAX_PNG_GRAY_VALUE));
      }
      DS_PRINTF("│\n");
    }

    DS_PRINTF("╵");
    for (size_t i = 0; i < png_input->width * 4; ++i)
      DS_PRINTF("─");
    DS_PRINTF("─╵\n");
  } break;
  default:
    assert(false && "Not implemented");
  }
}

void DS_PNG_input_free(DS_PNG_Input *const png_input) {
  DS_input_free(png_input->input);
  DS_FREE(png_input);
}
