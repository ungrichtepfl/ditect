#ifndef DEEPSEE_FILE_H
#define DEEPSEE_FILE_H

#include <stddef.h>

#define DS_FILE_MAX_PATH_LENGTH 1024

typedef struct {
  char **paths;
  size_t count;
} DS_FILE_FileList;

size_t DS_FILE_get_label_from_directory_name(const char *const file_path);

DS_FILE_FileList DS_FILE_get_files(const char *const dir_path);

void DS_FILE_file_list_free(DS_FILE_FileList *const file_list);

void DS_FILE_file_list_print(const DS_FILE_FileList *const file_list);

/// If cut is equal to 0, the full list is printed
void DS_FILE_file_list_print_labelled(const DS_FILE_FileList *const file_list,
                                      const size_t cut);

DS_FILE_FileList *
DS_FILE_get_random_bucket(const DS_FILE_FileList *const file_list,
                          const size_t max_count);

#endif // DEEPSEE_FILE_H
