#ifndef LOAD_PNG_H
#define LOAD_PNG_H

#include "deepsea.h"
#include <png.h>
#include <stdbool.h>

DS_Input *DS_load_input_from_grey_png(const char *const image_path);

#endif // LOAD_PNG_H
