#include "see.h"

#define DS_MALLOC SEE_DEBUG_MALLOC
#define DS_FREE SEE_DEBUG_FREE
#define DS_CALLOC SEE_DEBUG_CALLOC
#define DS_REALLOC SEE_DEBUG_REALLOC

#include "deepsea.c"
#include "deepsea_file.c"

#include "common.h"

void test_get_label_from_directory_name(void) {
  char *dir = "./some/dir/19/myfile";
  size_t label = DS_FILE_get_label_from_directory_name(dir);
  SEE_assert_eqlu(label, (size_t)19,
                  "Wrong label in long directory without extention.");

  dir = "17/myfile";
  label = DS_FILE_get_label_from_directory_name(dir);
  SEE_assert_eqlu(label, (size_t)17,
                  "Wrong label in short directory without extention.");

  dir = "./1/myfile";
  label = DS_FILE_get_label_from_directory_name(dir);
  SEE_assert_eqlu(label, (size_t)1,
                  "Wrong label in short relativ directory without extention.");

  dir = "/1/myfile";
  label = DS_FILE_get_label_from_directory_name(dir);
  SEE_assert_eqlu(label, (size_t)1,
                  "Wrong label in short absolute directory without extention.");

  dir = "/1/myfile.txt";
  label = DS_FILE_get_label_from_directory_name(dir);
  SEE_assert_eqlu(label, (size_t)1,
                  "Wrong label in short absolute directory with extention.");
}

void test_label_from_number_to_binary_array(void) {
  DS_FLOAT correct1[4] = {1., 1., 0., 1.};
  DS_FLOAT out1[4] = {0};
  DS_FILE_file_label_to_deepsea_label(11, out1, 4);
  for (size_t i = 0; i < 4; ++i)
    SEE_assert_eqf(out1[i], correct1[i],
                   "Wrong binary array for number 11 in index %lu.", i);

  DS_FLOAT correct2[6] = {1., 1., 0., 1., 0., 0.};
  DS_FLOAT out2[6] = {1., 1., 1., 1., 1., 1.};
  DS_FILE_file_label_to_deepsea_label(11, out2, 6);
  for (size_t i = 0; i < 6; ++i)
    SEE_assert_eqf(
        out2[i], correct2[i],
        "Wrong binary array for number 11 with excess elements in index %lu.",
        i);

  DS_FLOAT correct3[6] = {0., 1., 0., 0., 0., 0.};
  DS_FLOAT out3[6] = {1., 1., 1., 1., 1., 1.};
  DS_FILE_file_label_to_deepsea_label(2, out3, 6);
  for (size_t i = 0; i < 6; ++i)
    SEE_assert_eqf(
        out3[i], correct3[i],
        "Wrong binary array for number 2 with excess elements in index %lu.",
        i);

  DS_FLOAT correct4[2] = {0., 1.};
  DS_FLOAT out4[2] = {0};
  DS_FILE_file_label_to_deepsea_label(10, out4, 2);
  for (size_t i = 0; i < 2; ++i)
    SEE_assert_eqf(
        out4[i], correct4[i],
        "Wrong binary array for number 10 cropped output in index %lu.", i);
}

void test_file_list_creation_leak(void) {
  DS_FILE_FileList *file_list = DS_FILE_get_files(TEST_DIR);

  DS_FILE_file_list_free(file_list);
}

void test_file_list_get_random_bucket_leak(void) {
  DS_FILE_FileList *file_list = DS_FILE_get_files(TEST_DIR);
  int i = 0;
  while (DS_FILE_get_random_bucket(file_list, 2)) {
    ++i;
  }

  SEE_assert(i > 0, "It did not return any bucket");

  DS_FILE_file_list_free(file_list);
}

SEE_RUN_TESTS(test_get_label_from_directory_name,
              test_label_from_number_to_binary_array,
              test_file_list_creation_leak,
              test_file_list_get_random_bucket_leak)
