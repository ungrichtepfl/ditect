#include "deepsea_file.h"
#include "deepsea.h"
#include <dirent.h>
#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void fill_file_list(const char *const dir_path,
                           DS_FILE_FileList *const file_list) {
  DIR *dir = opendir(dir_path);
  if (!dir) {
    DS_FPRINTF(stderr, "Could not open directory \"%s\": %s\n", dir_path,
               strerror(errno));
    return;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    // Construct the full path of the entry
    const size_t path_length = strlen(dir_path) + strlen(entry->d_name) +
                               2; // Null terminating plus slash
    if (path_length > DS_FILE_MAX_PATH_LENGTH) {
      DS_ERROR("File path for file \"%s/%s\" is longer than %d. Skipping...",
               dir_path, entry->d_name, DS_FILE_MAX_PATH_LENGTH);
      continue;
    }
    char full_path[DS_FILE_MAX_PATH_LENGTH];
    sprintf(full_path, "%s/%s", dir_path, entry->d_name);

    if (entry->d_type == DT_REG) {
      file_list->paths = (char **)DS_REALLOC(
          file_list->paths, (file_list->count + 1) * sizeof(char *));
      DS_ASSERT(file_list->paths, "Could not load all files. Out of memory.");
      file_list->paths[file_list->count] = (char *)DS_MALLOC(path_length);
      DS_ASSERT(file_list->paths[file_list->count],
                "Could not load all files. Out of memory.");
      strcpy(file_list->paths[file_list->count], full_path);
      file_list->count++;
    } else if (entry->d_type == DT_LNK) {

      // Use lstat to get information about the file
      struct stat path_stat;
      if (stat(full_path, &path_stat) == -1) {
        DS_ERROR("Could not get file info: %s. Skipping...", strerror(errno));
        continue;
      }

      if (S_ISDIR(path_stat.st_mode)) {
        fill_file_list(full_path, file_list);
      } else {
        file_list->paths = (char **)DS_REALLOC(
            file_list->paths, (file_list->count + 1) * sizeof(char *));
        DS_ASSERT(file_list->paths, "Could not load all files. Out of memory.");
        file_list->paths[file_list->count] = (char *)DS_MALLOC(path_length);
        DS_ASSERT(file_list->paths[file_list->count],
                  "Could not load all files. Out of memory.");
        strcpy(file_list->paths[file_list->count], full_path);
        file_list->count++;
      }

    }

    else if (entry->d_type == DT_DIR) { // Check if the entry is a directory
      // Skip "." and ".." to avoid infinite recursion
      if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
        // Recursively get files in the subdirectory
        fill_file_list(full_path, file_list);
      }
    }
  }

  closedir(dir);
}

DS_FILE_FileList *DS_FILE_get_files(const char *const dir_path) {
  DS_FILE_FileList *const file_list = DS_CALLOC(1, sizeof(*file_list));
  DS_ASSERT(file_list, "Could not create file list, out of memory.");
  fill_file_list(dir_path, file_list);
  return file_list;
}

void DS_FILE_file_list_free(DS_FILE_FileList *const file_list) {
  for (size_t i = 0; i < file_list->count; ++i) {
    free(file_list->paths[i]);
  }
  free(file_list->paths);
  free(file_list);
}

void DS_FILE_file_list_print(const DS_FILE_FileList *const file_list) {
  const size_t cut = 10;
  const size_t stop = file_list->count > cut ? cut : file_list->count;
  for (size_t i = 0; i < stop; ++i)
    DS_PRINTF("%s\n", file_list->paths[i]);
}

void DS_FILE_file_list_print_labelled(const DS_FILE_FileList *const file_list) {
  const size_t cut = 10;
  const size_t stop = file_list->count > cut ? cut : file_list->count;
  for (size_t i = 0; i < stop; ++i)
    DS_PRINTF("%7lu: %s\n", DS_FILE_get_label(file_list->paths[i]),
              file_list->paths[i]);
}

size_t DS_FILE_get_label(const char *const file_path) {
  // Get the basename (the last part of the path)
  char *dir_path = DS_MALLOC(strlen(file_path) + 1);
  strcpy(dir_path, file_path); // Duplicate file path to safely manipulate it

  char *last_slash = strrchr(dir_path, '/');
  if (last_slash) {
    *last_slash = '\0'; // Null-terminate to get the directory part
  } else {
    DS_ERROR("Could not get label for \"%s\" wrong format. Returning 0.",
             file_path);
    DS_FREE(dir_path);
    return 0;
  }

  // Get the name of the parent directory
  char *parent_dir = strrchr(dir_path, '/');
  if (parent_dir) {
    parent_dir++; // Move past the '/' to get the actual directory name
  } else {
    parent_dir = dir_path; // If there is no '/', the directory is the root
  }

  // Convert the parent directory name to size_t
  size_t label = (size_t)strtoul(parent_dir, NULL, 10); // Convert to size_t
  DS_FREE(dir_path);
  return label;
}

void number_to_binary_array(size_t num, size_t *arr, const size_t arr_size) {
  // Start with the least significant bit and move to the most significant bit
  for (size_t i = 0; i < arr_size; i++) {
    // Set the current bit in the array (0 or 1)
    arr[arr_size - 1 - i] = (num & 1);
    // Right shift the number by 1 for the next bit
    num >>= 1;
  }
}
