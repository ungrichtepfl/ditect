#ifndef DEEPSEE_FILE_H
#define DEEPSEE_FILE_H

#include <stddef.h>

#define DS_FILE_MAX_PATH_LENGTH 1024

typedef struct {
  char **paths;
  size_t count;
} DS_FILE_FileList;

size_t DS_FILE_get_label(const char *const file_path);

DS_FILE_FileList *DS_FILE_get_files(const char *const dir_path);

void DS_FILE_file_list_free(DS_FILE_FileList *const file_list);

void DS_FILE_file_list_print(const DS_FILE_FileList *const file_list);

void DS_FILE_file_list_print_labelled(const DS_FILE_FileList *const file_list);

#endif // DEEPSEE_FILE_H
