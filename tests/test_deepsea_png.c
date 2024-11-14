#include "see.h"

#define DS_MALLOC SEE_DEBUG_MALLOC
#define DS_FREE SEE_DEBUG_FREE
#define DS_CALLOC SEE_DEBUG_CALLOC
#define DS_REALLOC SEE_DEBUG_REALLOC

#include "data/4_png.h"
#include "deepsea.c"
#include "deepsea_file.c"
#include "deepsea_png.c"

#include "common.h"

void test_load_png_gray(void) {

  size_t width = 28;
  size_t height = 28;
  SEE_assert_eqlu(width * height, (size_t)PNG_4_SIZE,
                  "Wrong with or height in test definition.");

  DS_PNG_Input *png_input = DS_PNG_input_load_grey(TEST_DATA_DIR "4.png");

  SEE_assert_eqi(png_input->type, DS_PNG_Gray, "Wrong png type.");

  SEE_assert_eqlu(png_input->width, width, "Wrong width");
  SEE_assert_eqlu(png_input->height, height, "Wrong height");
  for (size_t i = 0; i < PNG_4_SIZE; ++i)
    SEE_assert_eqf(png_input->data[i], (DS_FLOAT)png_4_data[i],
                   "Wrong png data for index %lu", i);

  DS_PNG_input_free(png_input);
}

SEE_RUN_TESTS(test_load_png_gray)
