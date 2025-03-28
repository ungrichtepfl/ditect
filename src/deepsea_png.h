#ifndef LOAD_PNG_H
#define LOAD_PNG_H

#include "deepsea.h"
#include "deepsea_file.h"

typedef enum {
  DS_PNG_Gray,
} DS_PNG_Type;

typedef struct {
  DS_FLOAT *data;
  size_t width;
  size_t height;
  DS_PNG_Type type;
} DS_PNG_Input;

DS_PNG_Input *DS_PNG_input_load_grey(const char *const image_path);

DS_Labelled_Inputs *
DS_PNG_file_list_to_labelled_inputs(const DS_FILE_FileList *const png_file_list,
                                    const DS_Network *const network);

void DS_PNG_input_print(const DS_PNG_Input *const png_input);

void DS_PNG_input_free(DS_PNG_Input *const png_input);

DS_PixelsBW DS_PNG_load_pixels_bw(const DS_PNG_Input *const png_input);

#endif // LOAD_PNG_H
