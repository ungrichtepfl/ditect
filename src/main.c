#include <math.h>
#include <raylib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PRINTF printf
#define WIN_HEIGHT 28
#define WIN_WIDTH 28
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
  FLOAT *values = (FLOAT *)CALLOC(m, sizeof(FLOAT));
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
  size_t *layer_sizes = MALLOC(num_layers * sizeof(size_t));
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

static inline size_t weight_index(size_t i, size_t j, size_t m) {
  return i * m + j;
}

void network_print(Network network) {

  PRINTF("Network: \n");
  PRINTF("Number of layers: %zu\n", network->num_layers);
  PRINTF("Layer sizes: [ ");
  for (size_t i = 0; i < network->num_layers; ++i) {
    PRINTF("%zu ", network->layer_sizes[i]);
  }
  PRINTF("]\n");

  for (size_t i = 0; i < network->num_layers - 1; ++i) {
    PRINTF("Biases %zu: [ ", i);
    for (size_t j = 0; j < network->layer_sizes[i + 1]; ++j) {
      PRINTF("%f ", network->biases[i][j]);
    }
    PRINTF("]\n");
  }

  // Print the matrix weights
  for (size_t i = 0; i < network->num_layers - 1; ++i) {
    PRINTF("Weights %zu:\n", i);
    size_t n = network->layer_sizes[i + 1];
    size_t m = network->layer_sizes[i];
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < m; ++k) {
        PRINTF("%f ", network->weights[i][weight_index(j, k, m)]);
      }
      PRINTF("\n");
    }
    PRINTF("\n");
  }
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

#define NUM_LAYERS 3

int main(void) {
  size_t layer_sizes[NUM_LAYERS] = {2, 3, 1};
  Network network = network_create(layer_sizes, NUM_LAYERS);

  network_print(network);

  network_free(network);
}
