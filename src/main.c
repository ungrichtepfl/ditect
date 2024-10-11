#include <math.h>
#include <raylib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PRINTF printf
#define NUM_LAYERS 3
#define NUM_INPUTS (28 * 28)
#define NUM_OUTPUTS 10
#define WIN_HEIGHT (NUM_INPUTS / 2)
#define WIN_WIDTH WIN_HEIGHT
#define TARGET_FPS 60
#define FLOAT double
#define MALLOC malloc
#define CALLOC calloc
#define FREE free
static bool random_init = false;
#define RAND rand
#define INIT_RAND()                                                            \
  do {                                                                         \
    if (!random_init) {                                                        \
      srand((unsigned int)time(NULL));                                         \
      random_init = true;                                                      \
    }                                                                          \
  } while (0)
#define EXIT(c) exit(c)

#define IDX(i, j, m) ((i) * (m) + (j))

typedef struct {
  FLOAT **activations;
  FLOAT **inputs;
} NetworkResult;

typedef struct {
  size_t num_layers;
  size_t *layer_sizes;
  FLOAT **biases;
  FLOAT **weights;
  NetworkResult *result;
} Network_t;

typedef Network_t *Network;

/// Normal random numbers generator - Marsaglia algorithm.
FLOAT *randn(const size_t n) {
  INIT_RAND();
  size_t m = n + n % 2;
  FLOAT *values = (FLOAT *)CALLOC(m, sizeof(values[0]));
  if (!values) {
    TraceLog(LOG_FATAL, "Could not create random array, out of memory");
    EXIT(1);
  }
  for (size_t i = 0; i < m; i += 2) {
    FLOAT x, y, rsq, f;
    do {
      x = 2.0 * RAND() / (FLOAT)RAND_MAX - 1.0;
      y = 2.0 * RAND() / (FLOAT)RAND_MAX - 1.0;
      rsq = x * x + y * y;
    } while (rsq >= 1. || rsq == 0.);
    f = sqrt(-2.0 * log(rsq) / rsq);
    values[i] = x * f;
    values[i + 1] = y * f;
  }
  return values;
}

void randno(FLOAT *const values, const size_t n) {
  FLOAT *r = randn(n);
  if (!memcpy(values, r, n * sizeof(values[0]))) {
    TraceLog(LOG_FATAL, "Could copy random values");
    EXIT(1);
  }
  free(r);
}

static NetworkResult *create_empty_result(const size_t num_layers,
                                          const size_t *const layer_sizes) {
  NetworkResult *result = MALLOC(sizeof(*result));
  if (!result) {
    TraceLog(LOG_FATAL, "Could not create result out of memory");
    EXIT(1);
  }
  result->inputs = MALLOC(num_layers * sizeof(result->inputs[0]));
  if (!result->inputs) {
    TraceLog(LOG_FATAL, "Could not create result out of memory");
    EXIT(1);
  }
  result->activations = MALLOC(num_layers * sizeof(result->activations[0]));
  if (!result->activations) {
    TraceLog(LOG_FATAL, "Could not create result out of memory");
    EXIT(1);
  }
  for (size_t l = 0; l < num_layers; ++l) {
    result->inputs[l] = CALLOC(layer_sizes[l], sizeof(result->inputs[l][0]));
    if (!result->inputs[l]) {
      TraceLog(LOG_FATAL, "Could not create result out of memory");
      EXIT(1);
    }
    result->activations[l] =
        CALLOC(layer_sizes[l], sizeof(result->activations[l][0]));
    if (!result->activations[l]) {
      TraceLog(LOG_FATAL, "Could not create result out of memory");
      EXIT(1);
    }
  }
  return result;
}

Network network_create_random(const size_t *const sizes,
                              const size_t num_layers) {
  if (num_layers < 2) {
    TraceLog(LOG_FATAL, "Cannot create network. At least 2 layers are needed.");
    EXIT(1);
  }
  Network network = MALLOC(sizeof(*network));
  size_t *layer_sizes = MALLOC(num_layers * sizeof(layer_sizes[0]));
  FLOAT **biases = MALLOC((num_layers - 1) * sizeof(biases[0]));
  FLOAT **weights = MALLOC((num_layers - 1) * sizeof(weights[0]));
  if (!network || !layer_sizes || !biases || !weights) {
    TraceLog(LOG_FATAL, "Could not create network, out of memory");
    EXIT(1);
  }
  if (!memcpy(layer_sizes, sizes, num_layers * sizeof(sizes[0]))) {
    TraceLog(LOG_FATAL, "Could copy layer sizes");
    EXIT(1);
  }

  for (size_t i = 0; i < num_layers - 1; ++i) {
    biases[i] = randn(layer_sizes[i + 1]);
    weights[i] = randn(layer_sizes[i] * layer_sizes[i + 1]);
  }

  network->layer_sizes = layer_sizes;
  network->num_layers = num_layers;
  network->weights = weights;
  network->biases = biases;
  network->result = create_empty_result(num_layers, sizes);

  return network;
}

void network_result_free(NetworkResult *result, const size_t num_layers) {
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

void network_free(Network network) {
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

void network_print(const Network network) {

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

static inline void sigmoid_prime(const FLOAT *z, FLOAT *const out,
                                 const size_t len) {
  for (size_t i = 0; i < len; ++i) {
    out[i] = sigmoid_prime_s(z[i]);
  }
}

static inline FLOAT distance(const FLOAT *const x, const FLOAT *const y,
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

void network_feedforward(Network network, const FLOAT *const input) {
  if (!memcpy(network->result->inputs[0], input,
              network->layer_sizes[0] * sizeof(input[0]))) {
    TraceLog(LOG_FATAL, "Could not copy inputs");
    EXIT(1);
  }

  if (!memcpy(network->result->activations[0], input,
              network->layer_sizes[0] * sizeof(input[0]))) {
    TraceLog(LOG_FATAL, "Could not copy activations");
    EXIT(1);
  }
  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    const size_t n = network->layer_sizes[l + 1];
    const size_t m = network->layer_sizes[l];
    const FLOAT *const W = network->weights[l];
    const FLOAT *const b = network->biases[l];
    dot_add(W, network->result->activations[l], b,
            network->result->activations[l + 1], n, m);
    if (!memcpy(network->result->inputs[l + 1],
                network->result->activations[l + 1],
                network->layer_sizes[l + 1] *
                    sizeof(network->result->inputs[l + 1][0]))) {
      TraceLog(LOG_FATAL, "Could not copy inputs");
      EXIT(1);
    }
    sigmoid(network->result->activations[l + 1],
            network->result->activations[l + 1], n); // Inplace
  }
}

FLOAT network_cost(const Network network, FLOAT *const *const xs,
                   FLOAT *const *const ys, const size_t num_training) {
  FLOAT cost = 0;
  size_t len_output = network->layer_sizes[network->num_layers - 1];
  for (size_t p = 0; p < num_training; ++p) {
    network_feedforward(network, xs[p]);
    cost += distance(network->result->activations[network->num_layers - 1],
                     ys[p], len_output);
  }
  return 1. / (2. * (FLOAT)num_training) * cost;
}

void network_print_activation_layer(Network network) {
  PRINTF("---------------- OUTPUT PER NEURON ----------------\n");
  for (size_t i = 0; i < network->layer_sizes[network->num_layers - 1]; ++i) {
    PRINTF("%lu => %f \n", i,
           network->result->activations[network->num_layers - 1][i]);
  }
  PRINTF("---------------------------------------------------\n");
}

typedef struct {
  FLOAT **errors;
  FLOAT **weight_error_sums;
  FLOAT **bias_error_sums;
  Network network;
} Backprop_t;

typedef Backprop_t *Backprop;

Backprop brackprop_create(const size_t *const sizes, const size_t num_layers) {
  Network network = network_create_random(sizes, num_layers);

  Backprop backprop = MALLOC(sizeof(*backprop));
  if (!backprop) {
    TraceLog(LOG_FATAL, "Could not create backprop. Out of memory.");
    EXIT(1);
  }
  backprop->errors = MALLOC(num_layers * sizeof(backprop->errors[0]));
  if (!backprop->errors) {
    TraceLog(LOG_FATAL, "Could not create backprop. Out of memory.");
    EXIT(1);
  }
  backprop->weight_error_sums =
      MALLOC((num_layers - 1) * sizeof(backprop->weight_error_sums[0]));
  if (!backprop->weight_error_sums) {
    TraceLog(LOG_FATAL, "Could not create backprop. Out of memory.");
    EXIT(1);
  }
  backprop->bias_error_sums =
      MALLOC((num_layers - 1) * sizeof(backprop->bias_error_sums[0]));
  if (!backprop->bias_error_sums) {
    TraceLog(LOG_FATAL, "Could not create backprop. Out of memory.");
    EXIT(1);
  }

  for (size_t l = 0; l < num_layers; ++l) {
    backprop->errors[l] =
        MALLOC(network->layer_sizes[l] * sizeof(backprop->errors[l][0]));
    if (!backprop->errors[l]) {
      TraceLog(LOG_FATAL, "Could not create backprop. Out of memory.");
      EXIT(1);
    }
  }
  for (size_t l = 0; l < num_layers - 1; ++l) {
    backprop->bias_error_sums[l] = MALLOC(
        network->layer_sizes[l + 1] * sizeof(backprop->bias_error_sums[l][0]));
    if (!backprop->bias_error_sums[l]) {
      TraceLog(LOG_FATAL, "Could not create backprop. Out of memory.");
      EXIT(1);
    }
    backprop->weight_error_sums[l] =
        MALLOC(network->layer_sizes[l] * network->layer_sizes[l + 1] *
               sizeof(backprop->weight_error_sums[l][0]));
    if (!backprop->weight_error_sums[l]) {
      TraceLog(LOG_FATAL, "Could not create backprop. Out of memory.");
      EXIT(1);
    }
  }
  backprop->network = network;
  return backprop;
}

void backprop_free(Backprop backprop) {
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
  network_free(backprop->network);
  free(backprop);
}

static inline FLOAT last_output_error_s(const FLOAT a, const FLOAT z,
                                        const FLOAT y) {
  return (a - y) * sigmoid_s(z);
}

static void calculate_output_error(Backprop backprop, const FLOAT *const y) {
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

static void calculate_error_sums(Backprop backprop, FLOAT *const *const xs,
                                 FLOAT *const *const ys, size_t num_trainig) {

  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    const size_t n = backprop->network->layer_sizes[l + 1];
    const size_t m = backprop->network->layer_sizes[l];
    memset(backprop->bias_error_sums[l], 0,
           n * sizeof(backprop->bias_error_sums[l][0]));
    memset(backprop->weight_error_sums[l], 0,
           m * n * sizeof(backprop->weight_error_sums[l][0]));
  }

  for (size_t d = 0; d < num_trainig; ++d) {
    const FLOAT *const x = xs[d];
    const FLOAT *const y = ys[d];
    network_feedforward(backprop->network, x);
    calculate_output_error(backprop, y);
    for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
      const size_t n = backprop->network->layer_sizes[l + 1];
      const size_t m = backprop->network->layer_sizes[l];
      const FLOAT *const a = backprop->network->result->activations[l];
      for (size_t i = 0; i < n; ++i) {
        backprop->bias_error_sums[l][i] += backprop->errors[l][i];
      }
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
          backprop->weight_error_sums[l][IDX(i, j, m)] +=
              a[j] * backprop->errors[l][i];
        }
      }
    }
  }
}

void update_weights_and_biases(Backprop backprop, const FLOAT rate) {
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

void backprop_learn(Backprop backprop, FLOAT *const *const xs,
                    FLOAT *const *const ys, const size_t num_trainig,
                    const FLOAT learing_rate) {
  FLOAT cost = network_cost(backprop->network, xs, ys, num_trainig);
  TraceLog(LOG_INFO, "Cost of network BEFORE learning: %.2f", cost);

  calculate_error_sums(backprop, xs, ys, num_trainig);

  const FLOAT rate = learing_rate / (FLOAT)num_trainig;
  update_weights_and_biases(backprop, rate);

  cost = network_cost(backprop->network, xs, ys, num_trainig);
  TraceLog(LOG_INFO, "cost of network AFTER learing: %.2f", cost);
}

#define NUM_TRAINING 2

int main(void) {
  size_t layer_sizes[NUM_LAYERS] = {NUM_INPUTS, 100, NUM_OUTPUTS};
  FLOAT learing_rate = 1.;
  Backprop backprop = brackprop_create(layer_sizes, NUM_LAYERS);

  /* network_print(backprop->network); */

  FLOAT *x = randn(NUM_INPUTS);

  FLOAT **ys = alloca(NUM_TRAINING * sizeof(ys[0]));
  for (size_t i = 0; i < NUM_TRAINING; ++i) {
    ys[i] = alloca(NUM_OUTPUTS * sizeof(FLOAT));
    memset(ys[i], 0, NUM_OUTPUTS * sizeof(FLOAT));
  }

  FLOAT **xs = alloca(NUM_TRAINING * sizeof(xs[0]));
  for (size_t i = 0; i < NUM_TRAINING; ++i) {
    xs[i] = x;
  }

  backprop_learn(backprop, xs, ys, NUM_TRAINING, learing_rate);

  network_print_activation_layer(backprop->network);

  backprop_free(backprop);
  free(x);
}

/**/
/* int main(void) { */
/**/
/*   InitWindow(WIN_WIDTH, WIN_HEIGHT, "Ditect"); */
/*   SetTargetFPS(TARGET_FPS); */
/**/
/*   while (!WindowShouldClose()) { */
/*     BeginDrawing(); */
/*     ClearBackground(BLACK); */
/**/
/*     EndDrawing(); */
/*   } */
/* } */
