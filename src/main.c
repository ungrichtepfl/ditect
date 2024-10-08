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
  size_t num_layers;
  size_t *layer_sizes;
  FLOAT **biases;
  FLOAT **weights;
} Network_t;

typedef Network_t *Network;

/// Normal random numbers generator - Marsaglia algorithm.
static FLOAT *randn(size_t n) {
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

Network network_create(size_t *sizes, size_t num_layers) {
  Network network = MALLOC(sizeof(*network));
  size_t *layer_sizes = MALLOC(num_layers * sizeof(layer_sizes[0]));
  FLOAT **biases = MALLOC((num_layers - 1) * sizeof(FLOAT));
  FLOAT **weights = MALLOC((num_layers - 1) * sizeof(FLOAT));
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

  return network;
}

void network_free(Network network) {
  for (size_t i = 0; i < network->num_layers - 1; ++i) {
    free(network->biases[i]);
    free(network->weights[i]);
  }
  free(network->biases);
  free(network->weights);
  free(network->layer_sizes);
  free(network);
}

void network_print(Network network) {

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

static inline void sigmoid(FLOAT *z, FLOAT *out, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    out[i] = 1 / (1 + exp(-z[i]));
  }
}

static inline void dot_add(FLOAT *W, FLOAT *x, FLOAT *b, FLOAT *out, size_t n,
                           size_t m) {
  for (size_t i = 0; i < n; ++i) {
    FLOAT tmp = 0;
    for (size_t j = 0; j < m; ++j) {
      tmp += W[IDX(i, j, m)] * x[j];
    }
    out[i] = tmp + b[i];
  }
}

void network_feedforward(Network network, FLOAT *input, FLOAT *output) {
  FLOAT *x = input;
  FLOAT *a = NULL;
  size_t n = 0;
  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    n = network->layer_sizes[l + 1];
    size_t m = network->layer_sizes[l];
    a = alloca(network->layer_sizes[l + 1] * sizeof(a[0]));
    FLOAT *W = network->weights[l];
    FLOAT *b = network->biases[l];
    dot_add(W, x, b, a, n, m);
    sigmoid(a, a, n); // Inplace
    x = a;
  }
  memcpy(output, a, n * sizeof(output[0]));
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

void print_output_layer(FLOAT *out, size_t n) {
  PRINTF("---------------- OUTPUT PER NEURON ----------------\n");
  for (size_t i = 0; i < n; ++i) {
    PRINTF("%lu => %f \n", i, out[i]);
  }
  PRINTF("---------------------------------------------------\n");
}

int main(void) {
  size_t layer_sizes[NUM_LAYERS] = {NUM_INPUTS, 100, NUM_OUTPUTS};
  Network network = network_create(layer_sizes, NUM_LAYERS);

  /* network_print(network); */

  FLOAT out[NUM_OUTPUTS] = {0};
  FLOAT *in = randn(NUM_INPUTS);

  network_feedforward(network, in, &out[0]);

  print_output_layer(out, NUM_OUTPUTS);

  network_free(network);
}
