#ifndef LOAD_PNG_H
#define LOAD_PNG_H

#include "deepsea.h"

typedef struct DS_PNG_Input DS_PNG_Input;

typedef enum {
  DS_PNG_Gray,
} DS_PNG_Type;

DS_PNG_Input *DS_PNG_input_load_grey(const char *const image_path);

void DS_PNG_input_print(const DS_PNG_Input *const png_input);

void DS_PNG_input_free(DS_PNG_Input *const png_input);

const DS_FLOAT *DS_PNG_input_get_data(const DS_PNG_Input *const png_input);

#endif // LOAD_PNG_H
