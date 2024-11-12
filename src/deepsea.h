#ifndef DEEPSEE_H
#define DEEPSEE_H

#ifndef FLOAT
#define FLOAT double
#endif
#ifndef MALLOC
#define MALLOC malloc
#endif
#ifndef CALLOC
#define CALLOC calloc
#endif
#ifndef FREE
#define FREE free
#endif
#ifndef EXIT
#define EXIT(c) exit(c)
#endif
#ifndef PRINTF
#include <stdio.h>
#define PRINTF printf
#endif
#ifndef FPRINTF
#include <stdio.h>
#define FPRINTF fprintf
#endif

typedef struct DS_Network DS_Network;

typedef struct DS_Backprop DS_Backprop;

typedef struct {
  FLOAT *in;
  size_t len;
} DS_Input;

void DS_input_free(DS_Input *const input);

FLOAT *DS_randn(const size_t n);

#define DS_ERROR(...)                                                          \
  do {                                                                         \
    FPRINTF(stderr,                                                            \
            "ERROR ("__FILE__                                                  \
            ": %d): ",                                                         \
            __LINE__);                                                         \
    FPRINTF(stderr, __VA_ARGS__);                                              \
    FPRINTF(stderr, "\n");                                                     \
  } while (0)

#define DS_ASSERT(cond, ...)                                                   \
  do {                                                                         \
    if (!(cond)) {                                                             \
      DS_ERROR(__VA_ARGS__);                                                   \
      EXIT(1);                                                                 \
    }                                                                          \
  } while (0)

void DS_randno(FLOAT *const values, const size_t n);

DS_Network *DS_network_create_random(const size_t *const sizes,
                                     const size_t num_layers,
                                     char *const *const output_labels);

DS_Network *DS_network_create(const FLOAT **const weights,
                              const FLOAT **const biases,
                              const size_t *const sizes,
                              const size_t num_layers,
                              char *const *const output_labels);

DS_Network *DS_network_create_owned(FLOAT **const weights, FLOAT **const biases,
                                    size_t *const sizes,
                                    const size_t num_layers,
                                    char *const *const output_labels);

void DS_network_free(DS_Network *const network);

void DS_network_print(const DS_Network *const network);

void DS_network_feedforward(DS_Network *const network,
                            const FLOAT *const input);

FLOAT DS_network_cost(DS_Network *const network, FLOAT *const *const xs,
                      FLOAT *const *const ys, const size_t num_training);

void DS_network_print_activation_layer(const DS_Network *const network);

DS_Backprop *DS_brackprop_create(const size_t *const sizes,
                                 const size_t num_layers,
                                 char *const *const output_labels);

DS_Backprop *DS_brackprop_create_from_network(DS_Network *const network);

void DS_backprop_free(DS_Backprop *const backprop);

void DS_backprop_learn_once(DS_Backprop *const backprop, FLOAT *const *const xs,
                            FLOAT *const *const ys, const size_t num_training,
                            const FLOAT learing_rate);

DS_Network const *DS_backprop_network(const DS_Backprop *const backprop);

FLOAT DS_backprop_network_cost(DS_Backprop *const backprop,
                               FLOAT *const *const xs, FLOAT *const *const ys,
                               const size_t num_training);

#endif // DEEPSEE_H
