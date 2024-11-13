#include "see.h"

#define DS_MALLOC SEE_DEBUG_MALLOC
#define DS_FREE SEE_DEBUG_FREE
#define DS_CALLOC SEE_DEBUG_CALLOC

#include "deepsea.c"
#include <stdio.h>

void test_idx(void) {

  DS_FLOAT W[6] = {1., 2., 3., 4., 5., 6.};
  SEE_assert_eqf(W[IDX(0, 1, 3)], 2., "idx 3 columns first row");
  SEE_assert_eqf(W[IDX(1, 2, 3)], 6., "idx 3 columns second row");
  SEE_assert_eqf(W[IDX(0, 1, 2)], 2., "idx 2 columns first row");
  SEE_assert_eqf(W[IDX(1, 0, 2)], 3., "idx 2 columns second row");
  SEE_assert_eqf(W[IDX(2, 0, 2)], 5., "idx 2 columns second row");
}

void test_dot_add_identity(void) {

  DS_FLOAT W[4] = {1., 0., 0., 1.};
  DS_FLOAT x[2] = {2., 3.};
  DS_FLOAT b[2] = {3., 7.};
  DS_FLOAT out[2] = {0};
  DS_FLOAT res[2] = {5., 10.};
  dot_add(&W[0], &x[0], &b[0], &out[0], 2, 2);
  for (int i = 0; i < 2; ++i)
    SEE_assert_eqf(out[i], res[i], "Dot add identity index %d", i);
}

void test_dot_add_sum(void) {

  DS_FLOAT W[4] = {1., 1., 1., 1.};
  DS_FLOAT x[2] = {2., 3.};
  DS_FLOAT b[2] = {3., 7.};
  DS_FLOAT out[2] = {0};
  DS_FLOAT res[2] = {8., 12.};
  dot_add(&W[0], &x[0], &b[0], &out[0], 2, 2);
  for (int i = 0; i < 2; ++i)
    SEE_assert_eqf(out[i], res[i], "Dot add sum index %d", i);
}

void test_dot_add_permutation(void) {

  DS_FLOAT W[4] = {0., 1., 1., 0.};
  DS_FLOAT x[2] = {2., 3.};
  DS_FLOAT b[2] = {3., 7.};
  DS_FLOAT out[2] = {0};
  DS_FLOAT res[2] = {6., 9.};
  dot_add(&W[0], &x[0], &b[0], &out[0], 2, 2);
  for (int i = 0; i < 2; ++i)
    SEE_assert_eqf(out[i], res[i], "Dot add permutation index %d", i);
}

void test_dot_add_non_symmetric(void) {

  DS_FLOAT W[6] = {1., 2., 3., 4., 5., 6.};
  DS_FLOAT x[3] = {7., 8., 9.};
  DS_FLOAT b[2] = {1., 2.};
  DS_FLOAT out[2] = {0};
  DS_FLOAT res[2] = {51., 124.};
  dot_add(&W[0], &x[0], &b[0], &out[0], 2, 3);
  for (int i = 0; i < 2; ++i)
    SEE_assert_eqf(out[i], res[i], "Dot add non-symmetric index %d", i);
}

void test_sigmoid_single(void) {
  DS_FLOAT z = 0;
  SEE_assert_eqf(sigmoid_s(z), 0.5, "Single sigmoid zero");
  z = 0.3;
  SEE_assert_eqf(sigmoid_s(z), 0.5744425168116848, "Single sigmoid positive");
  z = -0.4;
  SEE_assert_eqf(sigmoid_s(z), 0.40131233988751425, "Single sigmoid negative");
}

void test_sigmoid_multi(void) {
  DS_FLOAT z[3] = {0., 0.3, -0.4};
  DS_FLOAT res[3] = {0.5, 0.5744425168116848, 0.40131233988751425};

  DS_FLOAT out[3] = {0};
  sigmoid(&z[0], &out[0], 3);
  for (int i = 0; i < 3; ++i)
    SEE_assert_eqf(out[i], res[i], "Multi sigmoid value %d", i);
}

void test_sigmoid_prime_single(void) {
  DS_FLOAT z = 0;
  SEE_assert_eqf(sigmoid_prime_s(z), 0.25, "Single sigmoid_prime zero");
  z = 0.3;
  SEE_assert_eqf(sigmoid_prime_s(z), 0.24445831169074203,
                 "Single sigmoid_prime positive");
  z = -0.4;
  SEE_assert_eqf(sigmoid_prime_s(z), 0.24026074574152248,
                 "Single sigmoid_prime negative");
}

void test_distance_squared_zero(void) {
  DS_FLOAT x[3] = {1., 2., 3.};
  DS_FLOAT y[3] = {1., 2., 3.};
  SEE_assert_eqf(distance_squared(&x[0], &y[0], 3), 0., "Zero distance");
}

void test_distance_squared(void) {
  DS_FLOAT x[3] = {1., 2., -3.};
  DS_FLOAT y[3] = {-3., 1., 2.};
  SEE_assert_eqf(distance_squared(&x[0], &y[0], 3), 42., "Zero distance");
}

void check_random_network(const DS_Network *const network,
                          const size_t num_layers, const size_t *const sizes,
                          char *const *const labels) {
  SEE_assert_eqlu(network->num_layers, num_layers, "Network num layers");
  for (size_t l = 0; l < num_layers; ++l)
    SEE_assert_eqlu(network->layer_sizes[l], sizes[l], "Layer sizes index %lu",
                    l);
  if (labels) {
    SEE_assert_neqp(network->output_labels, NULL, "Output labels are NULL.");
    const size_t L = sizes[num_layers - 1];
    for (size_t i = 0; i < L; ++i)
      SEE_assert_eqstr(network->output_labels[i], labels[i],
                       "Output label index %lu", i);
  } else {
    SEE_assert_eqp(network->output_labels, NULL, "Output labels are not NULL.");
  }

  // Check for memory segfaults
  for (size_t l = 0; l < num_layers - 1; ++l) {
    size_t m = network->layer_sizes[l];
    size_t n = network->layer_sizes[l + 1];
    for (size_t i = 0; i < n; ++i) {
      volatile DS_FLOAT b = network->biases[l][i];
      (void)b;
      for (size_t j = 0; j < m; ++j) {
        volatile DS_FLOAT w = network->weights[l][IDX(i, j, m)];
        (void)w;
      }
    }
  }
  for (size_t l = 0; l < num_layers; ++l) {
    size_t n = network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      volatile DS_FLOAT a = network->result->activations[l][i];
      (void)a;
      volatile DS_FLOAT in = network->result->inputs[l][i];
      (void)in;
    }
  }
}

void test_network_creation_random(void) {
  size_t num_layers = 3;
  size_t sizes[3] = {2, 5, 7};
  char *labels[7] = {"1", "2", "3", "4", "5", "6", "7"};
  DS_Network *network = DS_network_create_random(&sizes[0], num_layers, labels);
  check_random_network(network, num_layers, sizes, labels);
  DS_network_free(network);
}

/* -------- START DEFINITION OF TEST NETWORK -------- */
#define NUM_LAYERS 3
#define LAYER_1 2
#define LAYER_2 3
#define LAYER_3 2
const DS_FLOAT WEIGHT_1[LAYER_1 * LAYER_2] = {.1, .2, .3, .4, .5, .6};
const DS_FLOAT WEIGHT_2[LAYER_2 * LAYER_3] = {.6, .5, .4, .3, .2, .1};
const DS_FLOAT BIAS_1[LAYER_2] = {.1, .2, .3};
const DS_FLOAT BIAS_2[LAYER_3] = {.4, .5};
const size_t LAYER_SIZES[NUM_LAYERS] = {LAYER_1, LAYER_2, LAYER_3};
const DS_FLOAT *WEIGHTS[NUM_LAYERS - 1] = {&WEIGHT_1[0], &WEIGHT_2[0]};
const DS_FLOAT *BIASES[NUM_LAYERS - 1] = {&BIAS_1[0], &BIAS_2[0]};
char *OUTPUT_LABELS[LAYER_3] = {"First Label", "Second Label"};

DS_Network *create_test_network(void) {
  return DS_network_create(&WEIGHTS[0], &BIASES[0], LAYER_SIZES, NUM_LAYERS,
                           OUTPUT_LABELS);
}

void check_network_empty_label_correctness(const DS_Network *const network) {
  SEE_assert_eqp(network->output_labels, NULL, "Output labels are not NULL.");
}

void check_network_label_correctness(const DS_Network *const network) {
  SEE_assert_neqp(network->output_labels, NULL, "Output labels are NULL.");
  const size_t L = LAYER_SIZES[NUM_LAYERS - 1];
  for (size_t i = 0; i < L; ++i)
    SEE_assert_eqstr(network->output_labels[i], OUTPUT_LABELS[i],
                     "Output label index %lu", i);
}

void check_network_correctness(const DS_Network *const network) {
  SEE_assert_eqlu((size_t)NUM_LAYERS, network->num_layers,
                  "Wrong number of layers.");
  for (size_t l = 0; l < network->num_layers; ++l)
    SEE_assert_eqlu(network->layer_sizes[l], LAYER_SIZES[l],
                    "Not the same layer sizes for layer %lu.", l);

  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    size_t n = network->layer_sizes[l + 1];
    size_t m = network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j)
        SEE_assert_eqf(network->weights[l][IDX(i, j, m)],
                       WEIGHTS[l][IDX(i, j, m)],
                       "Weight for layer %lu, index i=%lu, j=%lu", l + 1, i, j);
      SEE_assert_eqf(network->biases[l][i], BIASES[l][i],
                     "Bias for layer %lu, index i=%lu", l + 1, i);
    }
  }
}
/* -------- END DEFINITION OF TEST NETWORK -------- */

void test_create_test_network(void) {
  DS_Network *network = create_test_network();
  check_network_correctness(network);
  check_network_label_correctness(network);
  DS_network_free(network);
}

void test_create_test_network_owned(void) {

  size_t *sizes = DS_MALLOC(NUM_LAYERS * sizeof(sizes[0]));
  memcpy(sizes, LAYER_SIZES, NUM_LAYERS * sizeof(sizes[0]));
  DS_FLOAT **biases = DS_MALLOC((NUM_LAYERS - 1) * sizeof(biases[0]));
  DS_FLOAT **weights = DS_MALLOC((NUM_LAYERS - 1) * sizeof(weights[0]));
  for (size_t l = 0; l < NUM_LAYERS - 1; ++l) {
    biases[l] = DS_MALLOC(LAYER_SIZES[l + 1] * sizeof(biases[l][0]));
    memcpy(biases[l], BIASES[l], LAYER_SIZES[l + 1] * sizeof(biases[l][0]));
    weights[l] =
        DS_MALLOC(LAYER_SIZES[l] * LAYER_SIZES[l + 1] * sizeof(weights[l][0]));
    memcpy(weights[l], WEIGHTS[l],
           LAYER_SIZES[l] * LAYER_SIZES[l + 1] * sizeof(weights[l][0]));
  }
  DS_Network *network =
      DS_network_create_owned(weights, biases, sizes, NUM_LAYERS, NULL);
  check_network_correctness(network);
  check_network_empty_label_correctness(network);
  DS_network_free(network);
}

void test_network_feedforward(void) {
  DS_Network *network = create_test_network();
  const DS_FLOAT input[LAYER_1] = {.1, .2};
  const DS_FLOAT res_input_1[LAYER_1] = {.1, .2};
  const DS_FLOAT res_activation_1[LAYER_1] = {.1, .2};
  const DS_FLOAT res_input_2[LAYER_2] = {0.15, 0.31, 0.47};
  const DS_FLOAT res_activation_2[LAYER_2] = {0.53742985, 0.57688526,
                                              0.61538376};
  const DS_FLOAT res_input_3[LAYER_3] = {1.25705404, 0.83814438};
  const DS_FLOAT res_activation_3[LAYER_3] = {0.77851856, 0.69807426};

  const DS_FLOAT *res_inputs[NUM_LAYERS] = {&res_input_1[0], &res_input_2[0],
                                            &res_input_3[0]};
  const DS_FLOAT *res_activations[NUM_LAYERS] = {
      &res_activation_1[0], &res_activation_2[0], &res_activation_3[0]};

  DS_network_feedforward(network, &input[0]);

  for (size_t l = 0; l < NUM_LAYERS; ++l) {
    size_t n = network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      SEE_assert_eqf(network->result->inputs[l][i], res_inputs[l][i],
                     "Input for layer %lu for index %lu", l, i);
      SEE_assert_eqf(network->result->activations[l][i], res_activations[l][i],
                     "Activation for layer %lu for index %lu", l, i);
    }
  }
  DS_network_free(network);
}

void test_network_backprop_last_error(void) {
  DS_FLOAT a = 0.3;
  DS_FLOAT z = 0.5;
  DS_FLOAT y = 0.3;
  SEE_assert_eqf(last_output_error_s(a, z, y), 0.,
                 "Last output error: No error.");

  a = 0.8;
  z = 0.3;
  y = 0.1;
  SEE_assert_eqf(last_output_error_s(a, z, y), 0.17112081818352212,
                 "Last output error: Positive error.");
  a = 0.7;
  z = 0.8;
  y = 0.9;
  SEE_assert_eqf(last_output_error_s(a, z, y), -0.042781939304058894,
                 "Last output error: Negative error.");
}

void check_backprop_segfauls(const DS_Backprop *const backprop) {

  // Check for memory segfaults
  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    size_t m = backprop->network->layer_sizes[l];
    size_t n = backprop->network->layer_sizes[l + 1];
    for (size_t i = 0; i < n; ++i) {
      volatile DS_FLOAT b = backprop->bias_error_sums[l][i];
      (void)b;
      for (size_t j = 0; j < m; ++j) {
        volatile DS_FLOAT w = backprop->weight_error_sums[l][IDX(i, j, m)];
        (void)w;
      }
    }
  }
  for (size_t l = 0; l < backprop->network->num_layers; ++l) {
    size_t n = backprop->network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      volatile DS_FLOAT e = backprop->errors[l][i];
      (void)e;
    }
  }
}

void test_backprop_create(void) {
  size_t num_layers = 4;
  size_t sizes[4] = {2, 5, 3};
  char **labels = NULL;
  DS_Backprop *backprop = DS_brackprop_create(sizes, num_layers, labels);
  check_backprop_segfauls(backprop);
  check_random_network(DS_backprop_network(backprop), num_layers, sizes,
                       labels);
  DS_backprop_free(backprop);
}

void test_backprop_create_from_network(void) {
  DS_Backprop *backprop =
      DS_brackprop_create_from_network(create_test_network());
  check_backprop_segfauls(backprop);
  check_network_correctness(DS_backprop_network(backprop));
  DS_backprop_free(backprop);
}

void test_network_backprop_error_sums_single_input(void) {
  DS_Backprop *backprop =
      DS_brackprop_create_from_network(create_test_network());
  DS_FLOAT learning_rate = 1;
  size_t num_training = 1;
  DS_FLOAT x[LAYER_1] = {0.2, 0.1};
  DS_FLOAT y[LAYER_1] = {0.5, -0.3};
  DS_FLOAT *xs[1] = {&x[0]};
  DS_FLOAT *ys[1] = {&y[0]};
  DS_Labelled_Inputs labelled_inputs = {
      .inputs = xs, .labels = ys, .count = num_training};

  DS_FLOAT error_bias_1[LAYER_2] = {0.02287103, 0.01615626, 0.0095477};
  DS_FLOAT error_bias_2[LAYER_3] = {0.04801298, 0.21041785};
  DS_FLOAT error_weight_1[LAYER_1 * LAYER_2] = {
      0.00457421, 0.0022871, 0.00323125, 0.00161563, 0.00190954, 0.00095477};
  DS_FLOAT error_weight_2[LAYER_2 * LAYER_3] = {
      0.0256842, 0.0275807, 0.02943264, 0.11256154, 0.12087296, 0.12898913};

  DS_FLOAT *error_biases[NUM_LAYERS - 1] = {&error_bias_1[0], &error_bias_2[0]};
  DS_FLOAT *error_weights[NUM_LAYERS - 1] = {&error_weight_1[0],
                                             &error_weight_2[0]};

  DS_backprop_learn_once(backprop, &labelled_inputs, learning_rate);

  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    size_t n = backprop->network->layer_sizes[l + 1];
    size_t m = backprop->network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j)
        SEE_assert_eqf(backprop->weight_error_sums[l][IDX(i, j, m)],
                       error_weights[l][IDX(i, j, m)],
                       "Weight errors l=%lu, i=%lu, j=%lu", l, i, j);

      SEE_assert_eqf(backprop->bias_error_sums[l][i], error_biases[l][i],
                     "Bias error l=%lu, i=%lu", l, i);
    }
  }

  DS_backprop_free(backprop);
}

void test_network_backprop_double_input(void) {
  DS_Backprop *backprop =
      DS_brackprop_create_from_network(create_test_network());
  DS_FLOAT learning_rate = 0.8;
  size_t num_training = 2;

  DS_FLOAT x1[LAYER_1] = {0.3, 0.2};
  DS_FLOAT x2[LAYER_1] = {0.4, 0.5};
  DS_FLOAT y1[LAYER_1] = {0.1, -0.2};
  DS_FLOAT y2[LAYER_1] = {-0.3, 0.7};
  DS_FLOAT *xs[2] = {&x1[0], &x2[0]};
  DS_FLOAT *ys[2] = {&y1[0], &y2[0]};
  DS_FLOAT bias_1[LAYER_2] = {0.07672889, 0.1822576, 0.28776812};
  DS_FLOAT bias_2[LAYER_3] = {0.28116617, 0.42410679};
  DS_FLOAT weight_1[LAYER_1 * LAYER_2] = {0.09194743, 0.19213205, 0.29383053,
                                          0.39391128, 0.49571109, 0.5956956};
  DS_FLOAT weight_2[LAYER_2 * LAYER_3] = {0.53429254, 0.42713372, 0.32038983,
                                          0.25883112, 0.15510289, 0.05150874};

  DS_FLOAT *biases[NUM_LAYERS - 1] = {&bias_1[0], &bias_2[0]};
  DS_FLOAT *weights[NUM_LAYERS - 1] = {&weight_1[0], &weight_2[0]};

  DS_Labelled_Inputs labelled_inputs = {
      .inputs = xs, .labels = ys, .count = num_training};

  DS_backprop_learn_once(backprop, &labelled_inputs, learning_rate);

  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    size_t n = backprop->network->layer_sizes[l + 1];
    size_t m = backprop->network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j)
        SEE_assert_eqf(backprop->network->weights[l][IDX(i, j, m)],
                       weights[l][IDX(i, j, m)],
                       "Weight l=%lu, index i=%lu, j=%lu", l, i, j);

      SEE_assert_eqf(backprop->network->biases[l][i], biases[l][i],
                     "Bias l=%lu, i=%lu", l, i);
    }
  }

  DS_backprop_free(backprop);
}

void test_network_backprop_double_input_twice(void) {
  DS_Backprop *backprop =
      DS_brackprop_create_from_network(create_test_network());
  DS_FLOAT learning_rate = 0.8;
  size_t num_training = 2;

  DS_FLOAT x1[LAYER_1] = {0.3, 0.2};
  DS_FLOAT x2[LAYER_1] = {0.4, 0.5};
  DS_FLOAT y1[LAYER_1] = {0.1, -0.2};
  DS_FLOAT y2[LAYER_1] = {-0.3, 0.7};
  DS_FLOAT *xs[2] = {&x1[0], &x2[0]};
  DS_FLOAT *ys[2] = {&y1[0], &y2[0]};
  DS_Labelled_Inputs labelled_inputs = {
      .inputs = xs, .labels = ys, .count = num_training};

  DS_FLOAT bias_1[LAYER_2] = {0.05485079, 0.16633528, 0.27771601};
  DS_FLOAT bias_2[LAYER_3] = {0.15204091, 0.34987652};
  DS_FLOAT weight_1[LAYER_1 * LAYER_2] = {0.08434973, 0.18465363, 0.28826394,
                                          0.38835713, 0.49214933, 0.59204678};
  DS_FLOAT weight_2[LAYER_2 * LAYER_3] = {0.46380924, 0.34860201, 0.23426936,
                                          0.21911876, 0.11167309, 0.00448532};

  DS_FLOAT *biases[NUM_LAYERS - 1] = {&bias_1[0], &bias_2[0]};
  DS_FLOAT *weights[NUM_LAYERS - 1] = {&weight_1[0], &weight_2[0]};

  DS_backprop_learn_once(backprop, &labelled_inputs, learning_rate);
  DS_backprop_learn_once(backprop, &labelled_inputs, learning_rate);

  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    size_t n = backprop->network->layer_sizes[l + 1];
    size_t m = backprop->network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j)
        SEE_assert_eqf(backprop->network->weights[l][IDX(i, j, m)],
                       weights[l][IDX(i, j, m)],
                       "Weight l=%lu, index i=%lu, j=%lu", l, i, j);

      SEE_assert_eqf(backprop->network->biases[l][i], biases[l][i],
                     "Bias l=%lu, i=%lu", l, i);
    }
  }

  DS_backprop_free(backprop);
}

SEE_RUN_TESTS(test_sigmoid_single, test_sigmoid_multi,
              test_sigmoid_prime_single, test_dot_add_identity,
              test_dot_add_sum, test_dot_add_permutation, test_idx,
              test_distance_squared_zero, test_distance_squared,
              test_dot_add_non_symmetric, test_network_creation_random,
              test_create_test_network, test_create_test_network_owned,
              test_network_feedforward, test_backprop_create,
              test_backprop_create_from_network,
              test_network_backprop_last_error,
              test_network_backprop_error_sums_single_input,
              test_network_backprop_double_input,
              test_network_backprop_double_input_twice)
