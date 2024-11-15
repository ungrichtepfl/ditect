#include "deepsea_file.h"
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
        DS_ERROR("Could not get file info for \"%s\": %s. Skipping...",
                 full_path, strerror(errno));
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
  DS_FILE_FileList *file_list =
      DS_CALLOC(1, sizeof(*file_list)); // Sets file count to 0
  DS_ASSERT(file_list, "Could not create file list. Out of memory.");
  fill_file_list(dir_path, file_list);
  return file_list;
}

void DS_FILE_file_list_free(DS_FILE_FileList *const file_list) {
  for (size_t i = 0; i < file_list->count; ++i) {
    DS_FREE(file_list->paths[i]);
  }
  DS_FREE(file_list->paths);

  DS_FREE(file_list);
}

void DS_FILE_file_list_print(const DS_FILE_FileList *const file_list) {
  const size_t cut = 10;
  const size_t stop = file_list->count > cut ? cut : file_list->count;
  for (size_t i = 0; i < stop; ++i)
    DS_PRINTF("%s\n", file_list->paths[i]);
}

void DS_FILE_file_list_print_labelled(const DS_FILE_FileList *const file_list,
                                      size_t cut) {
  if (cut == 0)
    cut = file_list->count; // Take all the files

  const size_t stop = file_list->count > cut ? cut : file_list->count;
  for (size_t i = 0; i < stop; ++i)
    DS_PRINTF("%7lu: %s\n",
              DS_FILE_get_label_from_directory_name(file_list->paths[i]),
              file_list->paths[i]);
}

size_t DS_FILE_get_label_from_directory_name(const char *const file_path) {
  // Get the basename (the last part of the path)
  char *dir_path = DS_MALLOC(strlen(file_path) + 1);
  DS_ASSERT(dir_path,
            "Could not get label from directory name. Out of memory.");
  strcpy(dir_path, file_path); // Duplicate file path to safely manipulate it

  char *last_slash = strrchr(dir_path, '/');
  if (!last_slash) {
    DS_ERROR("Could not get label for \"%s\" wrong format.", file_path);
    errno = 1;
    DS_FREE(dir_path);
    return 0;
  }
  *last_slash = '\0'; // Null-terminate to get the directory part

  // Get the name of the parent directory
  char *parent_dir = strrchr(dir_path, '/');
  if (parent_dir) {
    parent_dir++; // Move past the '/' to get the actual directory name
  } else {
    parent_dir = dir_path; // If there is no '/', the directory is the root
  }

  // Convert the parent directory name to size_t
  errno = 0;
  size_t label = (size_t)strtoul(parent_dir, NULL, 10); // Convert to size_t
  if (label == 0 && errno != 0) {
    DS_ERROR("Could not parse directory \"%s\". Basename of file directory "
             "must be number but got: %s. Error: %s",
             file_path, parent_dir, strerror(errno));
    errno = 2;
    DS_FREE(dir_path);
    return 0;
  }
  DS_FREE(dir_path);
  return label;
}

void DS_FILE_file_label_to_deepsea_label(size_t file_label,
                                         DS_FLOAT *deepsea_label,
                                         const size_t num_outputs) {
  for (size_t i = 0; i < num_outputs; i++) {
    // Set the current bit in the array (0 or 1)
    deepsea_label[i] = (DS_FLOAT)(file_label & 1);
    file_label >>= 1;
  }
}

static void shuffle(size_t *shuffled, size_t length) {
  for (size_t i = 0, j = 0, k = 0, aux = 0; i < length; ++i) {
    do {
      j = rand() % length;
      k = rand() % length;
    } while (j == k);

    aux = shuffled[j];
    shuffled[j] = shuffled[k];
    shuffled[k] = aux;
  }
}

static void shuffelled_indexes(size_t *indexes, size_t length) {

  for (size_t i = 0; i < length; ++i) {
    indexes[i] = i;
  }

  shuffle(indexes, length);
}

static size_t *_random_file_array_indexes = NULL;
static size_t _current_random_bucket_count = 0;
static DS_FILE_FileList _current_random_file_list = {0};
DS_FILE_FileList *
DS_FILE_get_random_bucket(const DS_FILE_FileList *const file_list,
                          const size_t max_count) {
  if (_current_random_bucket_count ==
      0) { // This gets reset when we drained a full file list
    DS_init_rand(-1);
    _random_file_array_indexes =
        DS_MALLOC(file_list->count * sizeof(_random_file_array_indexes[0]));
    DS_ASSERT(_random_file_array_indexes,
              "Could not create random file array. Out of memory.");
    shuffelled_indexes(_random_file_array_indexes, file_list->count);
  }

  const size_t start = _current_random_bucket_count;
  size_t stop = _current_random_bucket_count + max_count;
  if (stop > file_list->count)
    stop = file_list->count;

  if (_current_random_file_list.paths) {
    DS_FREE(_current_random_file_list.paths);
    _current_random_file_list.paths = NULL;
  }

  if (stop <= start) {
    _current_random_bucket_count = 0;
    DS_FREE(_random_file_array_indexes);
    _random_file_array_indexes = NULL;
    return NULL;
  }

  const size_t count = stop - start; // we know this is at least 1

  _current_random_file_list.paths =
      DS_MALLOC(count * sizeof(_current_random_file_list.paths));
  DS_ASSERT(_current_random_file_list.paths,
            "Could not get new random bucket. Out of memory.");

  for (size_t i = start, j = 0; i < stop; ++i, ++j) {
    _current_random_file_list.paths[j] =
        file_list->paths[_random_file_array_indexes[i]];
  }
  _current_random_file_list.count = count;
  _current_random_bucket_count += count;
  return &_current_random_file_list;
}
