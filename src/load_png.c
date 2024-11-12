#include "load_png.h"

#include <stdlib.h>
#include <string.h>

#define MAX_PNG_GRAY_VALUE 255.

DS_Input *DS_load_input_from_grey_png(const char *const image_path) {
  png_image image;
  memset(&image, 0, sizeof(image));
  image.version = PNG_IMAGE_VERSION;

  if (!png_image_begin_read_from_file(&image, image_path)) {
    DS_ERROR("%s", image.message);
    return NULL;
  }

  image.format = PNG_FORMAT_GRAY;
  const size_t image_size = PNG_IMAGE_SIZE(image);
  png_bytep const buffer = MALLOC(image_size);
  if (!buffer) {
    png_image_free(&image);
    DS_ASSERT(false, "Could not allocate input buffer, out of memory.");
  }
  if (!png_image_finish_read(&image, NULL, buffer, 0, NULL)) {
    DS_ERROR("%s", image.message);
    free(buffer);
    return NULL;
  }

  FLOAT *const in = MALLOC(image_size * sizeof(in[0]));
  for (size_t i = 0; i < image_size; ++i)
    in[i] = (FLOAT)buffer[i] / MAX_PNG_GRAY_VALUE;
  free(buffer);

  DS_Input *const input = MALLOC(sizeof(*input));
  DS_ASSERT(input, "Could not allocate input struct, out of memory.");
  input->in = in;
  input->len = image_size;

  return input;
}
