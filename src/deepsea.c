#include "deepsea.h"
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static bool random_init = false;
#define RAND rand
#define INIT_RAND()                                                            \
  do {                                                                         \
    if (!random_init) {                                                        \
      srand((unsigned int)time(NULL));                                         \
      random_init = true;                                                      \
    }                                                                          \
  } while (0)

#define IDX(i, j, m) ((i) * (m) + (j))

#define ASSERT(cond, ...)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      FPRINTF(stderr,                                                          \
              "ERROR ("__FILE__                                                \
              ": %d): ",                                                       \
              __LINE__);                                                       \
      FPRINTF(stderr, __VA_ARGS__);                                            \
      FPRINTF(stderr, "\n");                                                   \
      EXIT(1);                                                                 \
    }                                                                          \
  } while (0)

typedef struct {
  FLOAT **activations;
  FLOAT **inputs;
} DS_NetworkResult;

struct DS_Network {
  size_t num_layers;
  size_t *layer_sizes;
  FLOAT **biases;
  FLOAT **weights;
  DS_NetworkResult *result;
};
typedef struct DS_Network DS_Network;

/// Normal random numbers generator - Marsaglia algorithm.
FLOAT *DS_randn(const size_t n) {
  INIT_RAND();
  size_t m = n + n % 2;
  FLOAT *values = (FLOAT *)CALLOC(m, sizeof(values[0]));
  ASSERT(values, "Could not create random array, out of memory.");
  for (size_t i = 0; i < m; i += 2) {
    FLOAT x, y, rsq, f;
    do {
      x = 2.0 * rand() / (FLOAT)RAND_MAX - 1.0;
      y = 2.0 * rand() / (FLOAT)RAND_MAX - 1.0;
      rsq = x * x + y * y;
    } while (rsq >= 1. || rsq == 0.);
    f = sqrt(-2.0 * log(rsq) / rsq);
    values[i] = x * f;
    values[i + 1] = y * f;
  }
  return values;
}

void DS_randno(FLOAT *const values, const size_t n) {
  FLOAT *r = DS_randn(n);
  ASSERT(memcpy(values, r, n * sizeof(values[0])), "Could copy random values");
  free(r);
}

static DS_NetworkResult *create_empty_result(const size_t num_layers,
                                             const size_t *const layer_sizes) {
  DS_NetworkResult *result = MALLOC(sizeof(*result));
  ASSERT(result, "Could not create result out of memory.");
  result->inputs = MALLOC(num_layers * sizeof(result->inputs[0]));
  ASSERT(result->inputs, "Could not create result out of memory.");
  result->activations = MALLOC(num_layers * sizeof(result->activations[0]));
  ASSERT(result->activations, "Could not create result out of memory.");
  for (size_t l = 0; l < num_layers; ++l) {
    result->inputs[l] = CALLOC(layer_sizes[l], sizeof(result->inputs[l][0]));
    ASSERT(result->inputs[l], "Could not create result out of memory.");
    result->activations[l] =
        CALLOC(layer_sizes[l], sizeof(result->activations[l][0]));
    ASSERT(result->activations[l], "Could not create result out of memory.");
  }
  return result;
}

DS_Network *DS_network_create_owned(FLOAT **const weights, FLOAT **const biases,
                                    size_t *const sizes,
                                    const size_t num_layers) {
  ASSERT(num_layers > 1,
         "Cannot create network. At least 2 layers are needed.");
  DS_Network *network = MALLOC(sizeof(*network));
  ASSERT(network, "Could not create network, out of memory.");

  network->layer_sizes = sizes;
  network->num_layers = num_layers;
  network->weights = weights;
  network->biases = biases;
  network->result = create_empty_result(num_layers, sizes);

  return network;
}

DS_Network *DS_network_create_random(const size_t *const sizes,
                                     const size_t num_layers) {
  ASSERT(num_layers > 1,
         "Cannot create network. At least 2 layers are needed.");
  size_t *layer_sizes = MALLOC(num_layers * sizeof(layer_sizes[0]));
  FLOAT **biases = MALLOC((num_layers - 1) * sizeof(biases[0]));
  FLOAT **weights = MALLOC((num_layers - 1) * sizeof(weights[0]));
  ASSERT(layer_sizes && biases && weights,
         "Could not create network, out of memory.");
  ASSERT(memcpy(layer_sizes, sizes, num_layers * sizeof(sizes[0])),
         "Could copy layer sizes.");

  for (size_t l = 0; l < num_layers - 1; ++l) {
    biases[l] = DS_randn(layer_sizes[l + 1]);
    weights[l] = DS_randn(layer_sizes[l] * layer_sizes[l + 1]);
  }
  return DS_network_create_owned(weights, biases, layer_sizes, num_layers);
}

DS_Network *DS_network_create(const FLOAT **const weights,
                              const FLOAT **const biases,
                              const size_t *const sizes,
                              const size_t num_layers) {
  ASSERT(num_layers > 1,
         "Cannot create network. At least 2 layers are needed.");
  size_t *layer_sizes = MALLOC(num_layers * sizeof(layer_sizes[0]));
  FLOAT **network_biases = MALLOC((num_layers - 1) * sizeof(network_biases[0]));
  FLOAT **network_weights =
      MALLOC((num_layers - 1) * sizeof(network_weights[0]));
  ASSERT(layer_sizes && network_biases && network_weights,
         "Could not create network, out of memory.");
  ASSERT(memcpy(layer_sizes, sizes, num_layers * sizeof(sizes[0])),
         "Could copy layer sizes.");

  for (size_t l = 0; l < num_layers - 1; ++l) {
    network_biases[l] =
        MALLOC(layer_sizes[l + 1] * sizeof(network_biases[l][0]));
    ASSERT(network_biases[l], "Could not create network, out of memory.");
    ASSERT(memcpy(network_biases[l], biases[l],
                  layer_sizes[l + 1] * sizeof(network_biases[l][0])),
           "Could not copy biases.");
    network_weights[l] = MALLOC(layer_sizes[l] * layer_sizes[l + 1] *
                                sizeof(network_weights[l][0]));
    ASSERT(network_weights[l], "Could not create network, out of memory.");
    ASSERT(memcpy(network_weights[l], weights[l],
                  layer_sizes[l] * layer_sizes[l + 1] *
                      sizeof(network_weights[l][0])),
           "Could not copy weights.");
  }
  return DS_network_create_owned(network_weights, network_biases, layer_sizes,
                                 num_layers);
}
static void network_result_free(DS_NetworkResult *result,
                                const size_t num_layers) {
  for (size_t l = 0; l < num_layers; ++l) {
    free(result->inputs[l]);
  }
  for (size_t l = 0; l < num_layers; ++l) {
    free(result->activations[l]);
  }
  free(result->inputs);
  free(result->activations);
  free(result);
}

void DS_network_free(DS_Network *const network) {
  network_result_free(network->result, network->num_layers);
  for (size_t i = 0; i < network->num_layers - 1; ++i) {
    free(network->biases[i]);
    free(network->weights[i]);
  }
  free(network->biases);
  free(network->weights);
  free(network->layer_sizes);

  free(network);
}

void DS_network_print(const DS_Network *const network) {

  PRINTF("Network: \n");
  PRINTF("Number of layers: %zu\n", network->num_layers);
  PRINTF("Layer sizes: [ ");
  for (size_t l = 0; l < network->num_layers; ++l) {
    PRINTF("%zu ", network->layer_sizes[l]);
  }
  PRINTF("]\n");

  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    PRINTF("Biases %zu: [ ", l);
    for (size_t i = 0; i < network->layer_sizes[l + 1]; ++i) {
      PRINTF("%f ", network->biases[l][i]);
    }
    PRINTF("]\n");
  }

  // Print the matrix weights
  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    PRINTF("Weights %zu:\n", l);
    size_t n = network->layer_sizes[l + 1];
    size_t m = network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        PRINTF("%f ", network->weights[l][IDX(i, j, m)]);
      }
      PRINTF("\n");
    }
    PRINTF("\n");
  }
}

static inline FLOAT sigmoid_s(const FLOAT z) { return 1 / (1 + exp(-z)); }

static inline void sigmoid(const FLOAT *const z, FLOAT *const out,
                           const size_t len) {
  for (size_t i = 0; i < len; ++i) {
    out[i] = sigmoid_s(z[i]);
  }
}

static inline FLOAT sigmoid_prime_s(const FLOAT z) {
  FLOAT e = exp(-z);
  return e / ((1 + e) * (1 + e));
}

static inline FLOAT distance_squared(const FLOAT *const x, const FLOAT *const y,
                                     const size_t n) {
  FLOAT out = 0;
  for (size_t i = 0; i < n; ++i) {
    FLOAT diff = (x[i] - y[i]);
    out += diff * diff;
  }
  return out;
}

static inline void dot_add(const FLOAT *const W, const FLOAT *const x,
                           const FLOAT *const b, FLOAT *const out,
                           const size_t n, const size_t m) {
  for (size_t i = 0; i < n; ++i) {
    FLOAT tmp = 0;
    for (size_t j = 0; j < m; ++j) {
      tmp += W[IDX(i, j, m)] * x[j];
    }
    out[i] = tmp + b[i];
  }
}

void DS_network_feedforward(DS_Network *const network,
                            const FLOAT *const input) {
  ASSERT(memcpy(network->result->inputs[0], input,
                network->layer_sizes[0] * sizeof(input[0])),
         "Could not copy inputs.");

  ASSERT(memcpy(network->result->activations[0], input,
                network->layer_sizes[0] * sizeof(input[0])),
         "Could not copy activations.");
  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    const size_t n = network->layer_sizes[l + 1];
    const size_t m = network->layer_sizes[l];
    const FLOAT *const W = network->weights[l];
    const FLOAT *const b = network->biases[l];
    dot_add(W, network->result->activations[l], b,
            network->result->activations[l + 1], n, m);
    ASSERT(memcpy(network->result->inputs[l + 1],
                  network->result->activations[l + 1],
                  network->layer_sizes[l + 1] *
                      sizeof(network->result->inputs[l + 1][0])),
           "Could not copy inputs.");
    sigmoid(network->result->activations[l + 1],
            network->result->activations[l + 1], n); // Inplace
  }
}

FLOAT DS_network_cost(DS_Network *const network, FLOAT *const *const xs,
                      FLOAT *const *const ys, const size_t num_training) {
  FLOAT cost = 0;
  size_t len_output = network->layer_sizes[network->num_layers - 1];
  for (size_t p = 0; p < num_training; ++p) {
    DS_network_feedforward(network, xs[p]);
    cost +=
        distance_squared(network->result->activations[network->num_layers - 1],
                         ys[p], len_output);
  }
  return 1. / (2. * (FLOAT)num_training) * cost;
}

void DS_network_print_activation_layer(const DS_Network *const network) {
  PRINTF("---------------- OUTPUT PER NEURON ----------------\n");
  for (size_t i = 0; i < network->layer_sizes[network->num_layers - 1]; ++i) {
    PRINTF("%lu => %f \n", i,
           network->result->activations[network->num_layers - 1][i]);
  }
  PRINTF("---------------------------------------------------\n");
}

struct DS_Backprop {
  FLOAT **errors;
  FLOAT **weight_error_sums;
  FLOAT **bias_error_sums;
  DS_Network *network;
};

typedef struct DS_Backprop DS_Backprop;

DS_Backprop *DS_brackprop_create_from_network(DS_Network *const network) {

  DS_Backprop *backprop = MALLOC(sizeof(*backprop));
  ASSERT(backprop, "Could not create backprop. Out of memory.");
  backprop->errors = MALLOC(network->num_layers * sizeof(backprop->errors[0]));
  ASSERT(backprop->errors, "Could not create backprop. Out of memory.");
  backprop->weight_error_sums = MALLOC((network->num_layers - 1) *
                                       sizeof(backprop->weight_error_sums[0]));
  ASSERT(backprop->weight_error_sums,
         "Could not create backprop. Out of memory.");
  backprop->bias_error_sums =
      MALLOC((network->num_layers - 1) * sizeof(backprop->bias_error_sums[0]));
  ASSERT(backprop->bias_error_sums,
         "Could not create backprop. Out of memory.");

  for (size_t l = 0; l < network->num_layers; ++l) {
    backprop->errors[l] =
        MALLOC(network->layer_sizes[l] * sizeof(backprop->errors[l][0]));
    ASSERT(backprop->errors[l], "Could not create backprop. Out of memory.");
  }
  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    backprop->bias_error_sums[l] = MALLOC(
        network->layer_sizes[l + 1] * sizeof(backprop->bias_error_sums[l][0]));
    ASSERT(backprop->bias_error_sums[l],
           "Could not create backprop. Out of memory.");
    backprop->weight_error_sums[l] =
        MALLOC(network->layer_sizes[l] * network->layer_sizes[l + 1] *
               sizeof(backprop->weight_error_sums[l][0]));
    ASSERT(backprop->weight_error_sums[l],
           "Could not create backprop. Out of memory.");
  }
  backprop->network = network;
  return backprop;
}

DS_Backprop *DS_brackprop_create(const size_t *const sizes,
                                 const size_t num_layers) {
  DS_Network *network = DS_network_create_random(sizes, num_layers);
  return DS_brackprop_create_from_network(network);
}

void DS_backprop_free(DS_Backprop *const backprop) {
  for (size_t l = 0; l < backprop->network->num_layers; ++l) {
    free(backprop->errors[l]);
  }
  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    free(backprop->bias_error_sums[l]);
    free(backprop->weight_error_sums[l]);
  }
  free(backprop->errors);
  free(backprop->bias_error_sums);
  free(backprop->weight_error_sums);
  DS_network_free(backprop->network);
  free(backprop);
}

static inline FLOAT last_output_error_s(const FLOAT a, const FLOAT z,
                                        const FLOAT y) {
  return (a - y) * sigmoid_prime_s(z);
}

static void calculate_output_error(DS_Backprop *const backprop,
                                   const FLOAT *const y) {
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
    const FLOAT *const W = backprop->network->weights[l];
    const FLOAT *const previous_error = backprop->errors[l + 1];
    const FLOAT *const z = backprop->network->result->inputs[l];

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
                                 FLOAT *const *const xs, FLOAT *const *const ys,
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
    const FLOAT *const x = xs[d];
    const FLOAT *const y = ys[d];
    DS_network_feedforward(backprop->network, x);
    calculate_output_error(backprop, y);
    for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
      const size_t n = backprop->network->layer_sizes[l + 1];
      const size_t m = backprop->network->layer_sizes[l];
      const FLOAT *const a = backprop->network->result->activations[l];
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
                                      const FLOAT rate) {
  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    const size_t n = backprop->network->layer_sizes[l + 1];
    const size_t m = backprop->network->layer_sizes[l];
    FLOAT *const W = backprop->network->weights[l];
    FLOAT *const b = backprop->network->biases[l];
    FLOAT *const weight_update = backprop->weight_error_sums[l];
    FLOAT *const bias_update = backprop->bias_error_sums[l];

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        W[IDX(i, j, m)] -= rate * weight_update[IDX(i, j, m)];
      }
      b[i] -= rate * bias_update[i];
    }
  }
}

void DS_backprop_learn_once(DS_Backprop *const backprop, FLOAT *const *const xs,
                            FLOAT *const *const ys, const size_t num_training,
                            const FLOAT learing_rate) {
  calculate_error_sums(backprop, xs, ys, num_training);

  const FLOAT rate = learing_rate / (FLOAT)num_training;
  update_weights_and_biases(backprop, rate);
}

FLOAT DS_backprop_network_cost(DS_Backprop *const backprop,
                               FLOAT *const *const xs, FLOAT *const *const ys,
                               const size_t num_training) {
  return DS_network_cost(backprop->network, xs, ys, num_training);
}

DS_Network const *DS_backprop_network(const DS_Backprop *const backprop) {
  return backprop->network;
}
