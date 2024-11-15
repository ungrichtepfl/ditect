#include "deepsea_png.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <png.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PNG_GRAY_VALUE 255.

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

  DS_FLOAT *const data = DS_MALLOC(image_size * sizeof(data[0]));
  for (size_t i = 0; i < image_size; ++i)
    data[i] = (DS_FLOAT)buffer[i] / MAX_PNG_GRAY_VALUE;
  DS_FREE(buffer);

  DS_PNG_Input *png_input = DS_MALLOC(sizeof(*png_input));
  DS_ASSERT(png_input, "Could not create png input, out of memory.");
  png_input->data = data;
  png_input->width = image.width;
  png_input->height = image.height;
  png_input->type = DS_PNG_Gray;

  return png_input;
}

DS_Labelled_Inputs *
DS_PNG_file_list_to_labelled_inputs(const DS_FILE_FileList *const png_file_list,
                                    const DS_Network *const network) {

  const size_t input_length = DS_network_input_layer_size(network);
  const size_t output_length = DS_network_output_layer_size(network);
  const size_t max_label = (size_t)powl(2, output_length) - 1;

  DS_Labelled_Inputs *labelled_input = DS_MALLOC(sizeof(*labelled_input));
  DS_ASSERT(labelled_input, "Could not create file list. Out of memory.");

  labelled_input->inputs =
      DS_MALLOC(png_file_list->count * sizeof(labelled_input->inputs));
  DS_ASSERT(labelled_input->inputs,
            "Could not create file list. Out of memory.");
  labelled_input->labels =
      DS_MALLOC(png_file_list->count * sizeof(labelled_input->labels));
  DS_ASSERT(labelled_input->labels,
            "Could not create file list. Out of memory.");
  for (size_t i = 0; i < png_file_list->count; ++i) {
    // NOTE: Do not malloc for inputs this will be moved from PNG!

    labelled_input->labels[i] =
        DS_MALLOC(output_length * sizeof(labelled_input->labels[0]));
    DS_ASSERT(labelled_input->labels[i],
              "Could not create file list. Out of memory.");
  }
  DS_PNG_Input *png_input = NULL;
  for (size_t i = 0; i < png_file_list->count; ++i) {
    const char *const png_file_path = png_file_list->paths[i];
    errno = 0;
    size_t label = DS_FILE_get_label_from_directory_name(png_file_path);
    if (label == 0 && errno != 0) {
      DS_ERROR("Could not get output label for file \"%s\"", png_file_path);
      goto file_list_to_labelled_inputs_error;
    }
    if (label > max_label) {
      DS_ERROR("Label of PNG \"s\" is bigger than maximum label representable "
               "by the number of outputs: %lu",
               max_label);
      goto file_list_to_labelled_inputs_error;
    }

    png_input = DS_PNG_input_load_grey(png_file_path);
    if (!png_input) {
      DS_ERROR("Could not load PNG \"%s\"", png_file_path);
      goto file_list_to_labelled_inputs_error;
    }
    if (png_input->width * png_input->height != input_length) {
      DS_ERROR("PNG data length does not fit network input size, must be %lu "
               "but got %lu",
               input_length, png_input->width * png_input->height);
      goto file_list_to_labelled_inputs_error;
    }

    labelled_input->inputs[i] = png_input->data;
    png_input->data = NULL; // Move the data
    DS_FILE_file_label_to_deepsea_label(label, labelled_input->labels[i],
                                        output_length);
    DS_PNG_input_free(png_input);
  }
  labelled_input->count = png_file_list->count;
  return labelled_input;

file_list_to_labelled_inputs_error:
  DS_FREE(png_input);
  for (size_t i = 0; i < png_file_list->count; ++i) {
    // NOTE: Inputs are only set when there are no errors anymore
    DS_FREE(labelled_input->labels[i]);
  }
  DS_FREE(labelled_input->inputs);
  DS_FREE(labelled_input->labels);
  DS_FREE(labelled_input);
  return NULL;
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
        DS_PRINTF("%3.d ", (int)(png_input->data[i + png_input->width * j] *
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
  DS_FREE(png_input->data);
  DS_FREE(png_input);
}
