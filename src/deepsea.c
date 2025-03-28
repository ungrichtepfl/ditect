#include "deepsea.h"
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

void DS_init_rand(long seed) {
  static bool random_init = false;

  if (!random_init) {
    const unsigned int s = (unsigned int)(seed >= 0 ? seed : time(NULL));
    srand(s);
    random_init = true;
  }
}

#define IDX(i, j, m) ((i) * (m) + (j))

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

#define SERIAL_SEP ";"

static char *read_line(FILE *file) {
  int bufferSize = 128; // Initial buffer size
  int length = 0;       // Current length of the string
  char *buffer = DS_MALLOC(bufferSize);

  if (!buffer) {
    DS_ERROR("Could not read line. Out of memory.");
    return NULL;
  }

  int character;
  while ((character = fgetc(file)) != EOF && character != '\n') {
    buffer[length++] = (char)character;

    // If we exceed the buffer, reallocate more memory
    if (length >= bufferSize) {
      bufferSize *= 2; // Double the buffer size
      char *newBuffer = DS_REALLOC(buffer, bufferSize);
      if (!newBuffer) {
        DS_ERROR("Could not read line. Out of memory.");
        DS_FREE(buffer);
        return NULL;
      }
      buffer = newBuffer;
    }
  }

  if (length == 0 && character == EOF) {
    DS_FREE(buffer);
    return NULL; // End of file and no characters read
  }

  buffer[length] = '\0'; // Null-terminate the string
  return buffer;
}

bool DS_network_save(const DS_Network *const network,
                     const char *const file_path) {
  FILE *f = NULL;
  if ((f = fopen(file_path, "w")) == NULL) {
    DS_ERROR("Could not open file \"%s\": %s", file_path, strerror(errno));
    return false;
  }

  fprintf(f, "%lu" SERIAL_SEP "\n", network->num_layers);
  for (size_t l = 0; l < network->num_layers; ++l)
    fprintf(f, "%lu" SERIAL_SEP, network->layer_sizes[l]);
  fprintf(f, "\n");

  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    const size_t n = network->layer_sizes[l + 1];
    for (size_t i = 0; i < n; ++i)
      fprintf(f, "%f" SERIAL_SEP, (double)network->biases[l][i]);
    fprintf(f, "\n");
  }

  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    const size_t n = network->layer_sizes[l + 1];
    const size_t m = network->layer_sizes[l];
    for (size_t i = 0; i < n * m; ++i)
      fprintf(f, "%f" SERIAL_SEP, (double)network->weights[l][i]);
    fprintf(f, "\n");
  }
  if (network->output_labels) {
    const size_t L = DS_network_output_layer_size(network);
    for (size_t i = 0; i < L; ++i)
      fprintf(f, "%s" SERIAL_SEP, network->output_labels[i]);
    fprintf(f, "\n");
  }

  if (fclose(f) != 0) {
    DS_ERROR("Could not close file \"%s\": %s", file_path, strerror(errno));
    return false;
  }

  return true;
}

typedef enum {
  PS_NUM_LAYERS,
  PS_LAYER_SIZES,
  PS_BIASES,
  PS_WEIGHTS,
  PS_OUTPUT_LABELS,
  PS_PARSING_ERROR,
} NetworkParsingState;

DS_Network *DS_network_load(const char *const file_path) {
  FILE *f = NULL;
  if ((f = fopen(file_path, "r")) == NULL) {
    DS_ERROR("Could not open file \"%s\": %s", file_path, strerror(errno));
    return NULL;
  }

  DS_FLOAT **weights = NULL;
  DS_FLOAT **biases = NULL;
  size_t *sizes = NULL;
  size_t num_layers = 0;
  char **output_labels = NULL;

  NetworkParsingState parsing_state = PS_NUM_LAYERS;
  size_t current_line = 1;
  size_t relative_line_index = 0;
  char *line = NULL;
  for (line = read_line(f); line != NULL;
       ++current_line, DS_FREE(line), line = read_line(f)) {
    switch (parsing_state) {
    case PS_NUM_LAYERS: {
      errno = 0;
      num_layers = strtoul(line, NULL, 10);
      if (num_layers == 0 && errno != 0) {
        DS_ERROR("Could not parse line %lu: %s", current_line, strerror(errno));
        goto load_error;
      }
      parsing_state = PS_LAYER_SIZES;
      relative_line_index = 0;
      continue;
    } break;
    case PS_LAYER_SIZES: {
      sizes = DS_MALLOC(num_layers * sizeof(sizes));
      DS_ASSERT(sizes, "Could not parse network. Out of memeory.");
      size_t i = 0;
      for (char *size_s = strtok(line, SERIAL_SEP); size_s != NULL;
           size_s = strtok(NULL, SERIAL_SEP), ++i) {
        if (i < num_layers) {
          errno = 0;
          sizes[i] = strtoul(size_s, NULL, 10);
          if (sizes[i] == 0 && errno != 0) {
            DS_ERROR("Could not parse line %lu. Wrong layer size format: %s",
                     current_line, strerror(errno));
            goto load_error;
          }
        }
      }
      if (i != num_layers) {
        DS_ERROR("Could not parse line %lu. Wrong number of sizes, expected "
                 "%lu got %lu",
                 current_line, num_layers, i);
        goto load_error;
      }
      parsing_state = PS_BIASES;
      relative_line_index = 0;
      continue;
    } break;
    case PS_BIASES: {
      if (relative_line_index == 0) {
        biases =
            DS_CALLOC(num_layers - 1,
                      sizeof(biases)); // Such that go to works (check if NULL)
        DS_ASSERT(biases, "Could not load network. Out of memory.");
      }
      const size_t n = sizes[relative_line_index + 1];
      biases[relative_line_index] = DS_MALLOC(n * sizeof(biases[0]));
      DS_ASSERT(biases[relative_line_index],
                "Could not load network. Out of memory.");

      size_t i = 0;
      for (char *bias_s = strtok(line, SERIAL_SEP); bias_s != NULL;
           bias_s = strtok(NULL, SERIAL_SEP), ++i) {
        if (i < n) {
          errno = 0;
          biases[relative_line_index][i] = strtof(bias_s, NULL);
          if (biases[relative_line_index][i] == 0 && errno != 0) {
            DS_ERROR("Could not parse line %lu. Wrong layer size format: %s",
                     current_line, strerror(errno));
            goto load_error;
          }
        }
      }
      if (i != n) {
        DS_ERROR("Could not parse line %lu. Wrong number of biases, expected "
                 "%lu got %lu",
                 current_line, n, i);
        goto load_error;
      }
      if (relative_line_index + 1 == num_layers - 1) {
        parsing_state = PS_WEIGHTS;
        relative_line_index = 0;
        continue;
      }
    } break;
    case PS_WEIGHTS: {
      if (relative_line_index == 0) {
        weights =
            DS_CALLOC(num_layers - 1,
                      sizeof(weights)); // Such that go to works (check if NULL)
        DS_ASSERT(weights, "Could not load network. Out of memory.");
      }
      const size_t n = sizes[relative_line_index + 1];
      const size_t m = sizes[relative_line_index];
      const size_t len = m * n;

      weights[relative_line_index] = DS_MALLOC(len * sizeof(weights[0]));
      DS_ASSERT(weights[relative_line_index],
                "Could not load network. Out of memory.");

      size_t i = 0;
      for (char *weight_s = strtok(line, SERIAL_SEP); weight_s != NULL;
           weight_s = strtok(NULL, SERIAL_SEP), ++i) {
        if (i < len) {
          errno = 0;
          weights[relative_line_index][i] = strtof(weight_s, NULL);
          if (weights[relative_line_index][i] == 0 && errno != 0) {
            DS_ERROR("Could not parse line %lu. Wrong layer size format: %s",
                     current_line, strerror(errno));
            goto load_error;
          }
        }
      }
      if (i != len) {
        DS_ERROR("Could not parse line %lu. Wrong number of weights, expected "
                 "%lu got %lu",
                 current_line, len, i);
        goto load_error;
      }
      if (relative_line_index + 1 == num_layers - 1) {
        parsing_state = PS_OUTPUT_LABELS;
        relative_line_index = 0;
        continue;
      }
    } break;
    case PS_OUTPUT_LABELS: {
      const size_t L = sizes[num_layers - 1];
      size_t i = 0;
      output_labels = DS_CALLOC(L, sizeof(output_labels));
      DS_ASSERT(output_labels, "Could not load network. Out of memory.");
      for (char *label = strtok(line, SERIAL_SEP); label != NULL;
           label = strtok(NULL, SERIAL_SEP), ++i) {
        if (i < L) {
          output_labels[i] = DS_MALLOC(strlen(label) + 1);
          DS_ASSERT(output_labels[i], "Could not load network. Out of memory.");
          strcpy(output_labels[i], label);
        }
      }

      if (i != L) {
        DS_ERROR(
            "Could not parse line %lu. Wrong number of output labels, expected "
            "%lu got %lu",
            current_line, L, i);
        goto load_error;
      }
      parsing_state = PS_PARSING_ERROR;
      relative_line_index = 0;
      continue;

    } break;
    case PS_PARSING_ERROR: {
      DS_ERROR("Too many lines in file. Ignoring line %lu...", current_line);
    } break;
    default:
      DS_ASSERT(false, "Unknown parsing state %d!", parsing_state);
    }
    ++relative_line_index;
  }

  DS_Network *network = DS_network_create_owned(weights, biases, sizes,
                                                num_layers, output_labels);

  if (fclose(f) != 0) {
    DS_ERROR("Could not close file \"%s\": %s", file_path, strerror(errno));
    DS_network_free(network);
    return NULL;
  }

  return network;

load_error:
  if (line)
    DS_FREE(line);
  if (sizes)
    DS_FREE(sizes);
  if (biases) {
    for (size_t l = 0; l < num_layers - 1; ++l) {
      if (biases[l])
        DS_FREE(biases[l]);
    }
  }
  if (weights) {
    for (size_t l = 0; l < num_layers - 1; ++l) {
      if (weights[l])
        DS_FREE(weights[l]);
    }
  }
  if (output_labels) {
    const size_t L =
        sizes[num_layers -
              1]; // sizes and num_layers must exist if output_labels exist
    for (size_t l = 0; l < L; ++l) {
      if (output_labels[l])
        DS_FREE(output_labels[l]);
    }
  }
  return NULL;
}

/// Normal random numbers generator - Marsaglia algorithm.
DS_FLOAT *DS_randn(const size_t n) {
  DS_init_rand(-1);
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

static char **create_owned_output_labels(char *const *const output_labels,
                                         const size_t *const sizes,
                                         const size_t num_layers) {
  char **owned_output_labels = NULL;
  if (output_labels) {
    const size_t L = sizes[num_layers - 1];
    owned_output_labels = DS_MALLOC(L * sizeof(owned_output_labels[0]));
    for (size_t i = 0; i < L; ++i) {
      const size_t label_len = strlen(output_labels[i]);
      DS_ASSERT(label_len <= MAX_OUTPUT_LABEL_STRLEN,
                "Output label at index %lu is longer than %d characters.", i,
                MAX_OUTPUT_LABEL_STRLEN);
      owned_output_labels[i] =
          DS_MALLOC((label_len + 1) * sizeof(output_labels[i][0]));
      strcpy(owned_output_labels[i], output_labels[i]);
    }
  }
  return owned_output_labels;
}

DS_Network *DS_network_create_owned(DS_FLOAT **const weights,
                                    DS_FLOAT **const biases,
                                    size_t *const sizes,
                                    const size_t num_layers,
                                    char **const output_labels) {
  DS_ASSERT(num_layers > 1,
            "Cannot create network. At least 2 layers are needed, got %lu.",
            num_layers);
  DS_Network *network = DS_MALLOC(sizeof(*network));
  DS_ASSERT(network, "Could not create network, out of memory.");

  network->layer_sizes = sizes;
  network->num_layers = num_layers;
  network->weights = weights;
  network->biases = biases;
  network->result = create_empty_result(num_layers, sizes);
  network->output_labels = output_labels;

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
    const size_t n = sizes[l];
    const size_t m = sizes[l + 1];
    const DS_FLOAT normalizer =
        1.f / sqrtf(n); // NOTE: Normalize weights by the number of other
                        // weights connected to the same neuron
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        weights[l][IDX(i, j, m)] = weights[l][IDX(i, j, m)] * normalizer;
      }
    }
  }
  return DS_network_create_owned(
      weights, biases, layer_sizes, num_layers,
      create_owned_output_labels(output_labels, sizes, num_layers));
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
  return DS_network_create_owned(
      network_weights, network_biases, layer_sizes, num_layers,
      create_owned_output_labels(output_labels, sizes, num_layers));
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
    const size_t L = DS_network_output_layer_size(network);
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

  DS_PRINTF("Network:\n");
  if (network->output_labels) {
    DS_PRINTF("Output labels: [ ");
    const size_t L = DS_network_output_layer_size(network);
    for (size_t i = 0; i < L; ++i)
      DS_PRINTF("%s ", network->output_labels[i]);
    DS_PRINTF("]\n");
  } else {
    DS_PRINTF("No output layers.");
  }
  DS_PRINTF("Number of layers: %zu\n", network->num_layers);
  DS_PRINTF("Layer sizes: [ ");
  for (size_t l = 0; l < network->num_layers; ++l)
    DS_PRINTF("%zu ", network->layer_sizes[l]);

  DS_PRINTF("]\n");

  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    DS_PRINTF("Biases %zu: [ ", l);
    for (size_t i = 0; i < network->layer_sizes[l + 1]; ++i)
      DS_PRINTF("%f ", network->biases[l][i]);
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

static DS_FLOAT quadratic_cost(const DS_FLOAT *const a, const DS_FLOAT *const y,
                               const size_t n) {
  DS_FLOAT cost = 0;
  for (size_t i = 0; i < n; ++i) {
    DS_FLOAT diff = (a[i] - y[i]);
    cost += diff * diff;
  }
  return 0.5f * cost;
}

static DS_FLOAT cross_entropy_cost(const DS_FLOAT *const a,
                                   const DS_FLOAT *const y, const size_t n) {
  DS_FLOAT out = 0;
  for (size_t i = 0; i < n; ++i) {
    DS_FLOAT tmp = y[i] * logf(a[i]) + (1 - y[i]) * logf(1 - a[i]);
    if (!isnanf(tmp))
      out += tmp;
  }
  return -out;
}

static DS_FLOAT l2_regularization_cost(const DS_FLOAT *const W, const size_t n,
                                       const size_t m) {
  DS_FLOAT cost = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      const DS_FLOAT tmp = W[IDX(i, j, m)];
      cost += tmp * tmp;
    }
  }
  return 0.5f * cost;
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

static const DS_FLOAT *
network_get_output_activations(const DS_Network *const network) {

  return network->result->activations[network->num_layers - 1];
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

DS_FLOAT DS_network_predict(DS_Network *const network,
                            const DS_FLOAT *const input,
                            char prediction[MAX_OUTPUT_LABEL_STRLEN + 1]) {

  DS_network_feedforward(network, input);
  size_t prediction_index = 0;
  DS_FLOAT max_activation = 0.;
  DS_FLOAT sum_activation = 0.;

  const size_t L = DS_network_output_layer_size(network);
  const DS_FLOAT *output_activations = network_get_output_activations(network);

  for (size_t i = 0; i < L; ++i) {
    sum_activation += output_activations[i];
    if (output_activations[i] > max_activation) {
      max_activation = output_activations[i];
      prediction_index = i;
    }
  }

  if (network->output_labels)
    strcpy(prediction, network->output_labels[prediction_index]);
  else
    snprintf(prediction, MAX_OUTPUT_LABEL_STRLEN, "%lu", prediction_index);

  return max_activation / sum_activation;
}

void DS_network_print_prediction(DS_Network *const network,
                                 const DS_FLOAT *const input) {

  char prediction[MAX_OUTPUT_LABEL_STRLEN + 1] = {0};
  DS_FLOAT probability = DS_network_predict(network, input, prediction);
  DS_PRINTF("Prediction is %s with probability of %.1f%%\n", prediction,
            probability);
}

void DS_network_print_activation_layer(const DS_Network *const network) {
  DS_PRINTF("---------------- OUTPUT PER NEURON ----------------\n");
  for (size_t i = 0; i < network->layer_sizes[network->num_layers - 1]; ++i) {
    DS_PRINTF("%lu => %f \n", i,
              network->result->activations[network->num_layers - 1][i]);
  }
  DS_PRINTF("---------------------------------------------------\n");
}

size_t DS_network_input_layer_size(const DS_Network *const network) {
  return network->layer_sizes[0];
}

size_t DS_network_output_layer_size(const DS_Network *const network) {
  return network->layer_sizes[network->num_layers - 1];
}

struct DS_Backprop {
  DS_FLOAT **errors;
  DS_FLOAT **weight_error_sums;
  DS_FLOAT **bias_error_sums;
  // NOTE: This functions only compues the cost for a single input (not
  // normalized by the number of inputs yet)
  DS_FLOAT (*cost_function)(const DS_FLOAT *const a, const DS_FLOAT *const y,
                            const size_t n);
  DS_FLOAT (*last_output_error)(const DS_FLOAT a, const DS_FLOAT z,
                                const DS_FLOAT y);
  DS_FLOAT regularization_param;
  DS_Network *network;
};

typedef struct DS_Backprop DS_Backprop;

void DS_labelled_inputs_free(DS_Labelled_Inputs *inputs) {
  for (size_t i = 0; i < inputs->count; ++i) {
    DS_FREE(inputs->inputs[i]);
    DS_FREE(inputs->labels[i]);
  }

  DS_FREE(inputs->inputs);
  DS_FREE(inputs->labels);

  DS_FREE(inputs);
}

static DS_FLOAT last_output_error_quadratic(const DS_FLOAT a, const DS_FLOAT z,
                                            const DS_FLOAT y) {
  return (a - y) * sigmoid_prime_s(z);
}

static DS_FLOAT last_output_error_cross_entropy(const DS_FLOAT a,
                                                const DS_FLOAT z,
                                                const DS_FLOAT y) {
  (void)z; // NOTE: Unused but needed for the interface
  return (a - y);
}

DS_Backprop *
DS_backprop_create_from_network(DS_Network *const network,
                                const DS_CostFunctionType cost_function_type,
                                const DS_FLOAT regularization_param) {

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
  switch (cost_function_type) {
  case DS_QUADRATIC: {
    backprop->cost_function = &quadratic_cost;
    backprop->last_output_error = &last_output_error_quadratic;
  } break;

  case DS_CROSS_ENTROPY: {
    backprop->cost_function = &cross_entropy_cost;
    backprop->last_output_error = &last_output_error_cross_entropy;
  } break;
  default: {
    DS_ASSERT(false, "Unreachable");
  } break;
  }
  backprop->regularization_param = regularization_param;
  backprop->network = network;
  return backprop;
}

DS_Backprop *DS_backprop_create(const size_t *const sizes,
                                const size_t num_layers,
                                char *const *const output_labels,
                                const DS_CostFunctionType cost_function_type,
                                const DS_FLOAT regularization_param) {
  DS_Network *network =
      DS_network_create_random(sizes, num_layers, output_labels);
  return DS_backprop_create_from_network(network, cost_function_type,
                                         regularization_param);
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

DS_FLOAT
DS_backprop_network_cost(DS_Backprop *const backprop,
                         const DS_Labelled_Inputs *const labelled_input) {
  DS_FLOAT cost = 0;
  size_t len_output =
      backprop->network->layer_sizes[backprop->network->num_layers - 1];
  for (size_t p = 0; p < labelled_input->count; ++p) {
    const DS_FLOAT *x = labelled_input->inputs[p];
    const DS_FLOAT *y = labelled_input->labels[p];
    DS_network_feedforward(backprop->network, x);
    cost += backprop->cost_function(
        backprop->network->result
            ->activations[backprop->network->num_layers - 1],
        y, len_output);
  }
  DS_FLOAT regularization_cost = 0;
  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    const size_t n = backprop->network->layer_sizes[l + 1];
    const size_t m = backprop->network->layer_sizes[l];
    const DS_FLOAT *const W = backprop->network->weights[l];
    regularization_cost +=
        l2_regularization_cost(W, n, m); // TODO: Let user choose type
  }
  return 1.f / (DS_FLOAT)labelled_input->count *
         (cost + backprop->regularization_param * regularization_cost);
}

static void calculate_output_error(DS_Backprop *const backprop,
                                   const DS_FLOAT *const y) {
  const size_t L = backprop->network->num_layers - 1;
  size_t n = backprop->network->layer_sizes[L];
  for (size_t i = 0; i < n; ++i) {
    backprop->errors[L][i] = backprop->last_output_error(
        backprop->network->result->activations[L][i],
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

static void
calculate_error_sums(DS_Backprop *const backprop,
                     const DS_Labelled_Inputs *const labelled_input) {

  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    const size_t n = backprop->network->layer_sizes[l + 1];
    const size_t m = backprop->network->layer_sizes[l];
    memset(backprop->bias_error_sums[l], 0,
           n * sizeof(backprop->bias_error_sums[l][0]));
    memset(backprop->weight_error_sums[l], 0,
           m * n * sizeof(backprop->weight_error_sums[l][0]));
  }

  for (size_t d = 0; d < labelled_input->count; ++d) {
    const DS_FLOAT *const x = labelled_input->inputs[d];
    const DS_FLOAT *const y = labelled_input->labels[d];
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
                                      const DS_FLOAT learning_rate,
                                      const size_t batch_size,
                                      const size_t total_training_set_size) {
  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    const size_t n = backprop->network->layer_sizes[l + 1];
    const size_t m = backprop->network->layer_sizes[l];
    DS_FLOAT *const W = backprop->network->weights[l];
    DS_FLOAT *const b = backprop->network->biases[l];
    DS_FLOAT *const weight_update = backprop->weight_error_sums[l];
    DS_FLOAT *const bias_update = backprop->bias_error_sums[l];

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        W[IDX(i, j, m)] =
            (1.f - learning_rate * backprop->regularization_param /
                       (DS_FLOAT)total_training_set_size) *
                W[IDX(i, j, m)] -
            learning_rate / (DS_FLOAT)batch_size * weight_update[IDX(i, j, m)];
      }
      b[i] -= learning_rate / (DS_FLOAT)batch_size * bias_update[i];
    }
  }
}

void DS_backprop_learn_once(DS_Backprop *const backprop,
                            const DS_Labelled_Inputs *const labelled_input,
                            const DS_FLOAT learing_rate,
                            const size_t total_training_set_size) {

  calculate_error_sums(backprop, labelled_input);

  update_weights_and_biases(backprop, learing_rate, labelled_input->count,
                            total_training_set_size);
}

DS_Network const *DS_backprop_network(const DS_Backprop *const backprop) {
  return backprop->network;
}

void DS_print_pixels_bw(const DS_PixelsBW *const pixels) {

  DS_PRINTF("╷");
  for (size_t i = 0; i < pixels->width * 4; ++i)
    DS_PRINTF("─");
  DS_PRINTF("─╷\n");

  for (size_t j = 0; j < pixels->height; ++j) {
    DS_PRINTF("│ ");
    for (size_t i = 0; i < pixels->width; ++i) {
      DS_PRINTF("%3.d ", 0);
    }
    DS_PRINTF("│\n");

    DS_PRINTF("│ ");
    for (size_t i = 0; i < pixels->width; ++i) {
      DS_PRINTF("%3.d ", (int)(pixels->data[i + pixels->width * j] * 255.f));
    }
    DS_PRINTF("│\n");
  }

  DS_PRINTF("╵");
  for (size_t i = 0; i < pixels->width * 4; ++i)
    DS_PRINTF("─");
  DS_PRINTF("─╵\n");
}

void DS_unload_pixels(DS_PixelsBW pixels) { DS_FREE(pixels.data); }
