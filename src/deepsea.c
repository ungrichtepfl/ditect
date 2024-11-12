#include "deepsea.h"
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static bool random_init = false;
#define INIT_RAND()                                                            \
  do {                                                                         \
    if (!random_init) {                                                        \
      srand((unsigned int)time(NULL));                                         \
      random_init = true;                                                      \
    }                                                                          \
  } while (0)

#define IDX(i, j, m) ((i) * (m) + (j))

#define MAX_OUTPUT_LABEL_STRLEN 0xFF

typedef struct {
  DS_FLOAT **activations;
  DS_FLOAT **inputs;
} DS_NetworkResult;

struct DS_Network {
  size_t num_layers;
  size_t *layer_sizes;
  DS_FLOAT **biases;
  DS_FLOAT **weights;
  DS_NetworkResult *result;
  char **output_labels;
};
typedef struct DS_Network DS_Network;

/// Normal random numbers generator - Marsaglia algorithm.
DS_FLOAT *DS_randn(const size_t n) {
  INIT_RAND();
  size_t m = n + n % 2;
  DS_FLOAT *values = (DS_FLOAT *)DS_CALLOC(m, sizeof(values[0]));
  DS_ASSERT(values, "Could not create random array, out of memory.");
  for (size_t i = 0; i < m; i += 2) {
    DS_FLOAT x, y, rsq, f;
    do {
      x = 2.0 * rand() / (DS_FLOAT)RAND_MAX - 1.0;
      y = 2.0 * rand() / (DS_FLOAT)RAND_MAX - 1.0;
      rsq = x * x + y * y;
    } while (rsq >= 1. || rsq == 0.);
    f = sqrt(-2.0 * log(rsq) / rsq);
    values[i] = x * f;
    values[i + 1] = y * f;
  }
  return values;
}

void DS_randno(DS_FLOAT *const values, const size_t n) {
  DS_FLOAT *r = DS_randn(n);
  DS_ASSERT(memcpy(values, r, n * sizeof(values[0])),
            "Could copy random values");
  DS_FREE(r);
}

static DS_NetworkResult *create_empty_result(const size_t num_layers,
                                             const size_t *const layer_sizes) {
  DS_NetworkResult *result = DS_MALLOC(sizeof(*result));
  DS_ASSERT(result, "Could not create result out of memory.");
  result->inputs = DS_MALLOC(num_layers * sizeof(result->inputs[0]));
  DS_ASSERT(result->inputs, "Could not create result out of memory.");
  result->activations = DS_MALLOC(num_layers * sizeof(result->activations[0]));
  DS_ASSERT(result->activations, "Could not create result out of memory.");
  for (size_t l = 0; l < num_layers; ++l) {
    result->inputs[l] = DS_CALLOC(layer_sizes[l], sizeof(result->inputs[l][0]));
    DS_ASSERT(result->inputs[l], "Could not create result out of memory.");
    result->activations[l] =
        DS_CALLOC(layer_sizes[l], sizeof(result->activations[l][0]));
    DS_ASSERT(result->activations[l], "Could not create result out of memory.");
  }
  return result;
}

DS_Network *DS_network_create_owned(DS_FLOAT **const weights,
                                    DS_FLOAT **const biases,
                                    size_t *const sizes,
                                    const size_t num_layers,
                                    char *const *const output_labels) {
  DS_ASSERT(num_layers > 1,
            "Cannot create network. At least 2 layers are needed.");
  DS_Network *network = DS_MALLOC(sizeof(*network));
  DS_ASSERT(network, "Could not create network, out of memory.");

  network->layer_sizes = sizes;
  network->num_layers = num_layers;
  network->weights = weights;
  network->biases = biases;
  network->result = create_empty_result(num_layers, sizes);
  network->output_labels = NULL;

  if (output_labels) {
    const size_t L = sizes[num_layers - 1];
    network->output_labels = DS_MALLOC(L * sizeof(network->output_labels[0]));
    for (size_t i = 0; i < L; ++i) {
      const size_t label_len = strlen(output_labels[i]);
      DS_ASSERT(label_len <= MAX_OUTPUT_LABEL_STRLEN,
                "Output label at index %lu is longer than %d characters.", i,
                MAX_OUTPUT_LABEL_STRLEN);
      network->output_labels[i] =
          DS_MALLOC((label_len + 1) * sizeof(output_labels[i][0]));
      strcpy(network->output_labels[i], output_labels[i]);
    }
  }

  return network;
}

DS_Network *DS_network_create_random(const size_t *const sizes,
                                     const size_t num_layers,
                                     char *const *const output_labels) {
  DS_ASSERT(num_layers > 1,
            "Cannot create network. At least 2 layers are needed.");
  size_t *layer_sizes = DS_MALLOC(num_layers * sizeof(layer_sizes[0]));
  DS_FLOAT **biases = DS_MALLOC((num_layers - 1) * sizeof(biases[0]));
  DS_FLOAT **weights = DS_MALLOC((num_layers - 1) * sizeof(weights[0]));
  DS_ASSERT(layer_sizes && biases && weights,
            "Could not create network, out of memory.");
  DS_ASSERT(memcpy(layer_sizes, sizes, num_layers * sizeof(sizes[0])),
            "Could copy layer sizes.");

  for (size_t l = 0; l < num_layers - 1; ++l) {
    biases[l] = DS_randn(layer_sizes[l + 1]);
    weights[l] = DS_randn(layer_sizes[l] * layer_sizes[l + 1]);
  }
  return DS_network_create_owned(weights, biases, layer_sizes, num_layers,
                                 output_labels);
}

DS_Network *DS_network_create(const DS_FLOAT **const weights,
                              const DS_FLOAT **const biases,
                              const size_t *const sizes,
                              const size_t num_layers,
                              char *const *const output_labels) {
  DS_ASSERT(num_layers > 1,
            "Cannot create network. At least 2 layers are needed.");
  size_t *layer_sizes = DS_MALLOC(num_layers * sizeof(layer_sizes[0]));
  DS_FLOAT **network_biases =
      DS_MALLOC((num_layers - 1) * sizeof(network_biases[0]));
  DS_FLOAT **network_weights =
      DS_MALLOC((num_layers - 1) * sizeof(network_weights[0]));
  DS_ASSERT(layer_sizes && network_biases && network_weights,
            "Could not create network, out of memory.");
  DS_ASSERT(memcpy(layer_sizes, sizes, num_layers * sizeof(sizes[0])),
            "Could copy layer sizes.");

  for (size_t l = 0; l < num_layers - 1; ++l) {
    network_biases[l] =
        DS_MALLOC(layer_sizes[l + 1] * sizeof(network_biases[l][0]));
    DS_ASSERT(network_biases[l], "Could not create network, out of memory.");
    DS_ASSERT(memcpy(network_biases[l], biases[l],
                     layer_sizes[l + 1] * sizeof(network_biases[l][0])),
              "Could not copy biases.");
    network_weights[l] = DS_MALLOC(layer_sizes[l] * layer_sizes[l + 1] *
                                   sizeof(network_weights[l][0]));
    DS_ASSERT(network_weights[l], "Could not create network, out of memory.");
    DS_ASSERT(memcpy(network_weights[l], weights[l],
                     layer_sizes[l] * layer_sizes[l + 1] *
                         sizeof(network_weights[l][0])),
              "Could not copy weights.");
  }
  return DS_network_create_owned(network_weights, network_biases, layer_sizes,
                                 num_layers, output_labels);
}
static void network_result_free(DS_NetworkResult *result,
                                const size_t num_layers) {
  for (size_t l = 0; l < num_layers; ++l) {
    DS_FREE(result->inputs[l]);
  }
  for (size_t l = 0; l < num_layers; ++l) {
    DS_FREE(result->activations[l]);
  }
  DS_FREE(result->inputs);
  DS_FREE(result->activations);
  DS_FREE(result);
}

void DS_network_free(DS_Network *const network) {
  network_result_free(network->result, network->num_layers);

  if (network->output_labels) {
    const size_t L = network->layer_sizes[network->num_layers - 1];
    for (size_t i = 0; i < L; ++i)
      DS_FREE(network->output_labels[i]);
    DS_FREE(network->output_labels);
  }

  for (size_t i = 0; i < network->num_layers - 1; ++i) {
    DS_FREE(network->biases[i]);
    DS_FREE(network->weights[i]);
  }
  DS_FREE(network->biases);
  DS_FREE(network->weights);
  DS_FREE(network->layer_sizes);

  DS_FREE(network);
}

void DS_network_print(const DS_Network *const network) {

  DS_PRINTF("Network: \n");
  DS_PRINTF("Number of layers: %zu\n", network->num_layers);
  DS_PRINTF("Layer sizes: [ ");
  for (size_t l = 0; l < network->num_layers; ++l) {
    DS_PRINTF("%zu ", network->layer_sizes[l]);
  }
  DS_PRINTF("]\n");

  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    DS_PRINTF("Biases %zu: [ ", l);
    for (size_t i = 0; i < network->layer_sizes[l + 1]; ++i) {
      DS_PRINTF("%f ", network->biases[l][i]);
    }
    DS_PRINTF("]\n");
  }

  // Print the matrix weights
  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    DS_PRINTF("Weights %zu:\n", l);
    size_t n = network->layer_sizes[l + 1];
    size_t m = network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        DS_PRINTF("%f ", network->weights[l][IDX(i, j, m)]);
      }
      DS_PRINTF("\n");
    }
    DS_PRINTF("\n");
  }
}

static inline DS_FLOAT sigmoid_s(const DS_FLOAT z) { return 1 / (1 + exp(-z)); }

static inline void sigmoid(const DS_FLOAT *const z, DS_FLOAT *const out,
                           const size_t len) {
  for (size_t i = 0; i < len; ++i) {
    out[i] = sigmoid_s(z[i]);
  }
}

static inline DS_FLOAT sigmoid_prime_s(const DS_FLOAT z) {
  DS_FLOAT e = exp(-z);
  return e / ((1 + e) * (1 + e));
}

static inline DS_FLOAT distance_squared(const DS_FLOAT *const x,
                                        const DS_FLOAT *const y,
                                        const size_t n) {
  DS_FLOAT out = 0;
  for (size_t i = 0; i < n; ++i) {
    DS_FLOAT diff = (x[i] - y[i]);
    out += diff * diff;
  }
  return out;
}

static inline void dot_add(const DS_FLOAT *const W, const DS_FLOAT *const x,
                           const DS_FLOAT *const b, DS_FLOAT *const out,
                           const size_t n, const size_t m) {
  for (size_t i = 0; i < n; ++i) {
    DS_FLOAT tmp = 0;
    for (size_t j = 0; j < m; ++j) {
      tmp += W[IDX(i, j, m)] * x[j];
    }
    out[i] = tmp + b[i];
  }
}

void DS_network_feedforward(DS_Network *const network,
                            const DS_FLOAT *const input) {
  DS_ASSERT(memcpy(network->result->inputs[0], input,
                   network->layer_sizes[0] * sizeof(input[0])),
            "Could not copy inputs.");

  DS_ASSERT(memcpy(network->result->activations[0], input,
                   network->layer_sizes[0] * sizeof(input[0])),
            "Could not copy activations.");
  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    const size_t n = network->layer_sizes[l + 1];
    const size_t m = network->layer_sizes[l];
    const DS_FLOAT *const W = network->weights[l];
    const DS_FLOAT *const b = network->biases[l];
    dot_add(W, network->result->activations[l], b,
            network->result->activations[l + 1], n, m);
    DS_ASSERT(memcpy(network->result->inputs[l + 1],
                     network->result->activations[l + 1],
                     network->layer_sizes[l + 1] *
                         sizeof(network->result->inputs[l + 1][0])),
              "Could not copy inputs.");
    sigmoid(network->result->activations[l + 1],
            network->result->activations[l + 1], n); // Inplace
  }
}

void DS_input_free(DS_Input *const input) {
  DS_FREE(input->in);
  DS_FREE(input);
}

DS_FLOAT DS_network_cost(DS_Network *const network, DS_FLOAT *const *const xs,
                         DS_FLOAT *const *const ys, const size_t num_training) {
  DS_FLOAT cost = 0;
  size_t len_output = network->layer_sizes[network->num_layers - 1];
  for (size_t p = 0; p < num_training; ++p) {
    DS_network_feedforward(network, xs[p]);
    cost +=
        distance_squared(network->result->activations[network->num_layers - 1],
                         ys[p], len_output);
  }
  return 1. / (2. * (DS_FLOAT)num_training) * cost;
}

void DS_network_print_activation_layer(const DS_Network *const network) {
  DS_PRINTF("---------------- OUTPUT PER NEURON ----------------\n");
  for (size_t i = 0; i < network->layer_sizes[network->num_layers - 1]; ++i) {
    DS_PRINTF("%lu => %f \n", i,
              network->result->activations[network->num_layers - 1][i]);
  }
  DS_PRINTF("---------------------------------------------------\n");
}

struct DS_Backprop {
  DS_FLOAT **errors;
  DS_FLOAT **weight_error_sums;
  DS_FLOAT **bias_error_sums;
  DS_Network *network;
};

typedef struct DS_Backprop DS_Backprop;

DS_Backprop *DS_brackprop_create_from_network(DS_Network *const network) {

  DS_Backprop *backprop = DS_MALLOC(sizeof(*backprop));
  DS_ASSERT(backprop, "Could not create backprop. Out of memory.");
  backprop->errors =
      DS_MALLOC(network->num_layers * sizeof(backprop->errors[0]));
  DS_ASSERT(backprop->errors, "Could not create backprop. Out of memory.");
  backprop->weight_error_sums = DS_MALLOC(
      (network->num_layers - 1) * sizeof(backprop->weight_error_sums[0]));
  DS_ASSERT(backprop->weight_error_sums,
            "Could not create backprop. Out of memory.");
  backprop->bias_error_sums = DS_MALLOC((network->num_layers - 1) *
                                        sizeof(backprop->bias_error_sums[0]));
  DS_ASSERT(backprop->bias_error_sums,
            "Could not create backprop. Out of memory.");

  for (size_t l = 0; l < network->num_layers; ++l) {
    backprop->errors[l] =
        DS_MALLOC(network->layer_sizes[l] * sizeof(backprop->errors[l][0]));
    DS_ASSERT(backprop->errors[l], "Could not create backprop. Out of memory.");
  }
  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    backprop->bias_error_sums[l] = DS_MALLOC(
        network->layer_sizes[l + 1] * sizeof(backprop->bias_error_sums[l][0]));
    DS_ASSERT(backprop->bias_error_sums[l],
              "Could not create backprop. Out of memory.");
    backprop->weight_error_sums[l] =
        DS_MALLOC(network->layer_sizes[l] * network->layer_sizes[l + 1] *
                  sizeof(backprop->weight_error_sums[l][0]));
    DS_ASSERT(backprop->weight_error_sums[l],
              "Could not create backprop. Out of memory.");
  }
  backprop->network = network;
  return backprop;
}

DS_Backprop *DS_brackprop_create(const size_t *const sizes,
                                 const size_t num_layers,
                                 char *const *const output_labels) {
  DS_Network *network =
      DS_network_create_random(sizes, num_layers, output_labels);
  return DS_brackprop_create_from_network(network);
}

void DS_backprop_free(DS_Backprop *const backprop) {
  for (size_t l = 0; l < backprop->network->num_layers; ++l) {
    DS_FREE(backprop->errors[l]);
  }
  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    DS_FREE(backprop->bias_error_sums[l]);
    DS_FREE(backprop->weight_error_sums[l]);
  }
  DS_FREE(backprop->errors);
  DS_FREE(backprop->bias_error_sums);
  DS_FREE(backprop->weight_error_sums);
  DS_network_free(backprop->network);
  DS_FREE(backprop);
}

static inline DS_FLOAT last_output_error_s(const DS_FLOAT a, const DS_FLOAT z,
                                           const DS_FLOAT y) {
  return (a - y) * sigmoid_prime_s(z);
}

static void calculate_output_error(DS_Backprop *const backprop,
                                   const DS_FLOAT *const y) {
  const size_t L = backprop->network->num_layers - 1;
  size_t n = backprop->network->layer_sizes[L];
  for (size_t i = 0; i < n; ++i) {
    backprop->errors[L][i] =
        last_output_error_s(backprop->network->result->activations[L][i],
                            backprop->network->result->inputs[L][i], y[i]);
  }

  for (long long l = L - 1; l >= 0; --l) {
    n = backprop->network->layer_sizes[l + 1];
    const size_t m = backprop->network->layer_sizes[l];
    const DS_FLOAT *const W = backprop->network->weights[l];
    const DS_FLOAT *const previous_error = backprop->errors[l + 1];
    const DS_FLOAT *const z = backprop->network->result->inputs[l];

    for (size_t j = 0; j < m; ++j) {
      backprop->errors[l][j] = 0;
      for (size_t i = 0; i < n; ++i) {
        backprop->errors[l][j] += W[IDX(i, j, m)] * previous_error[i];
      }
      backprop->errors[l][j] *= sigmoid_prime_s(z[j]);
    }
  }
}

static void calculate_error_sums(DS_Backprop *const backprop,
                                 DS_FLOAT *const *const xs,
                                 DS_FLOAT *const *const ys,
                                 size_t num_training) {

  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    const size_t n = backprop->network->layer_sizes[l + 1];
    const size_t m = backprop->network->layer_sizes[l];
    memset(backprop->bias_error_sums[l], 0,
           n * sizeof(backprop->bias_error_sums[l][0]));
    memset(backprop->weight_error_sums[l], 0,
           m * n * sizeof(backprop->weight_error_sums[l][0]));
  }

  for (size_t d = 0; d < num_training; ++d) {
    const DS_FLOAT *const x = xs[d];
    const DS_FLOAT *const y = ys[d];
    DS_network_feedforward(backprop->network, x);
    calculate_output_error(backprop, y);
    for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
      const size_t n = backprop->network->layer_sizes[l + 1];
      const size_t m = backprop->network->layer_sizes[l];
      const DS_FLOAT *const a = backprop->network->result->activations[l];
      for (size_t i = 0; i < n; ++i) {
        backprop->bias_error_sums[l][i] += backprop->errors[l + 1][i];
      }
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
          backprop->weight_error_sums[l][IDX(i, j, m)] +=
              a[j] * backprop->errors[l + 1][i];
        }
      }
    }
  }
}

static void update_weights_and_biases(DS_Backprop *const backprop,
                                      const DS_FLOAT rate) {
  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    const size_t n = backprop->network->layer_sizes[l + 1];
    const size_t m = backprop->network->layer_sizes[l];
    DS_FLOAT *const W = backprop->network->weights[l];
    DS_FLOAT *const b = backprop->network->biases[l];
    DS_FLOAT *const weight_update = backprop->weight_error_sums[l];
    DS_FLOAT *const bias_update = backprop->bias_error_sums[l];

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        W[IDX(i, j, m)] -= rate * weight_update[IDX(i, j, m)];
      }
      b[i] -= rate * bias_update[i];
    }
  }
}

void DS_backprop_learn_once(DS_Backprop *const backprop,
                            DS_FLOAT *const *const xs,
                            DS_FLOAT *const *const ys,
                            const size_t num_training,
                            const DS_FLOAT learing_rate) {
  calculate_error_sums(backprop, xs, ys, num_training);

  const DS_FLOAT rate = learing_rate / (DS_FLOAT)num_training;
  update_weights_and_biases(backprop, rate);
}

DS_FLOAT DS_backprop_network_cost(DS_Backprop *const backprop,
                                  DS_FLOAT *const *const xs,
                                  DS_FLOAT *const *const ys,
                                  const size_t num_training) {
  return DS_network_cost(backprop->network, xs, ys, num_training);
}

DS_Network const *DS_backprop_network(const DS_Backprop *const backprop) {
  return backprop->network;
}
