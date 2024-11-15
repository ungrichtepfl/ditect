#ifndef DEEPSEE_H
#define DEEPSEE_H

#include <stdbool.h>

#ifndef DS_FLOAT
#define DS_FLOAT double
#endif
#ifndef DS_MALLOC
#define DS_MALLOC malloc
#endif
#ifndef DS_CALLOC
#define DS_CALLOC calloc
#endif
#ifndef DS_REALLOC
#define DS_REALLOC realloc
#endif
#ifndef DS_FREE
#define DS_FREE free
#endif
#ifndef DS_EXIT
#define DS_EXIT(c) exit(c)
#endif
#ifndef DS_PRINTF
#include <stdio.h>
#define DS_PRINTF printf
#endif
#ifndef DS_FPRINTF
#include <stdio.h>
#define DS_FPRINTF fprintf
#endif

typedef struct DS_Network DS_Network;

typedef struct DS_Backprop DS_Backprop;

typedef struct {
  DS_FLOAT **inputs;
  DS_FLOAT **labels;
  size_t count;
} DS_Labelled_Inputs;

void DS_labelled_inputs_free(DS_Labelled_Inputs *inputs);

/// Initialize the random number generator with a seed.
/// If a negative seed is given, the current time is used as the seed.
void DS_init_rand(long seed);

DS_FLOAT *DS_randn(const size_t n);

#define DS_ERROR(...)                                                          \
  do {                                                                         \
    DS_FPRINTF(stderr,                                                         \
               "ERROR ("__FILE__                                               \
               ": %d): ",                                                      \
               __LINE__);                                                      \
    DS_FPRINTF(stderr, __VA_ARGS__);                                           \
    DS_FPRINTF(stderr, "\n");                                                  \
  } while (0)

#define DS_ASSERT(cond, ...)                                                   \
  do {                                                                         \
    if (!(cond)) {                                                             \
      DS_ERROR(__VA_ARGS__);                                                   \
      DS_EXIT(1);                                                              \
    }                                                                          \
  } while (0)

void DS_randno(DS_FLOAT *const values, const size_t n);

bool DS_network_save(const DS_Network *const network,
                     const char *const file_path);

DS_Network *DS_network_load(const char *const file_path);

DS_Network *DS_network_create_random(const size_t *const sizes,
                                     const size_t num_layers,
                                     char *const *const output_labels);

DS_Network *DS_network_create(const DS_FLOAT **const weights,
                              const DS_FLOAT **const biases,
                              const size_t *const sizes,
                              const size_t num_layers,
                              char *const *const output_labels);

DS_Network *DS_network_create_owned(DS_FLOAT **const weights,
                                    DS_FLOAT **const biases,
                                    size_t *const sizes,
                                    const size_t num_layers,
                                    char **const output_labels);

void DS_network_free(DS_Network *const network);

void DS_network_print(const DS_Network *const network);

void DS_network_feedforward(DS_Network *const network,
                            const DS_FLOAT *const input);

DS_FLOAT DS_network_cost(DS_Network *const network,
                         const DS_Labelled_Inputs *const labelled_input);

void DS_network_print_activation_layer(const DS_Network *const network);

size_t DS_network_input_layer_size(const DS_Network *const network);

size_t DS_network_output_layer_size(const DS_Network *const network);

DS_Backprop *DS_brackprop_create(const size_t *const sizes,
                                 const size_t num_layers,
                                 char *const *const output_labels);

DS_Backprop *DS_brackprop_create_from_network(DS_Network *const network);

void DS_backprop_free(DS_Backprop *const backprop);

void DS_backprop_learn_once(DS_Backprop *const backprop,
                            const DS_Labelled_Inputs *const labelled_input,
                            const DS_FLOAT learing_rate);

DS_Network const *DS_backprop_network(const DS_Backprop *const backprop);

DS_FLOAT
DS_backprop_network_cost(DS_Backprop *const backprop,
                         const DS_Labelled_Inputs *const labelled_input);

#endif // DEEPSEE_H
