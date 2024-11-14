#include "see.h"

#define DS_MALLOC SEE_DEBUG_MALLOC
#define DS_FREE SEE_DEBUG_FREE
#define DS_CALLOC SEE_DEBUG_CALLOC

#include "deepsea.c"
#include "deepsea_file.c"


void test_get_label_from_directory_name(void) {}

SEE_RUN_TESTS(test_get_label_from_directory_name)
