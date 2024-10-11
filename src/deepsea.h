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

FLOAT *DS_randn(const size_t n);

void DS_randno(FLOAT *const values, const size_t n);

DS_Network *DS_network_create_random(const size_t *const sizes,
                                     const size_t num_layers);

void DS_network_free(DS_Network *const network);

void DS_network_print(const DS_Network *const network);

void DS_network_feedforward(DS_Network *const network,
                            const FLOAT *const input);

FLOAT DS_network_cost(DS_Network *const network, FLOAT *const *const xs,
                      FLOAT *const *const ys, const size_t num_training);

void DS_network_print_activation_layer(const DS_Network *const network);

DS_Backprop *DS_brackprop_create(const size_t *const sizes,
                                 const size_t num_layers);

void DS_backprop_free(DS_Backprop *const backprop);

void DS_backprop_learn_once(DS_Backprop *const backprop, FLOAT *const *const xs,
                            FLOAT *const *const ys, const size_t num_trainig,
                            const FLOAT learing_rate);

DS_Network const *DS_backprop_network(const DS_Backprop *const backprop);

#endif // DEEPSEE_H
