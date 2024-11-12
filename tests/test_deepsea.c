#include "deepsea.c"
#include "see.h"
#include <stdio.h>

void test_idx(void) {

  FLOAT W[6] = {1., 2., 3., 4., 5., 6.};
  assert_eqf(W[IDX(0, 1, 3)], 2., "idx 3 columns first row");
  assert_eqf(W[IDX(1, 2, 3)], 6., "idx 3 columns second row");
  assert_eqf(W[IDX(0, 1, 2)], 2., "idx 2 columns first row");
  assert_eqf(W[IDX(1, 0, 2)], 3., "idx 2 columns second row");
  assert_eqf(W[IDX(2, 0, 2)], 5., "idx 2 columns second row");
}

void test_dot_add_identity(void) {

  FLOAT W[4] = {1., 0., 0., 1.};
  FLOAT x[2] = {2., 3.};
  FLOAT b[2] = {3., 7.};
  FLOAT out[2] = {0};
  FLOAT res[2] = {5., 10.};
  dot_add(&W[0], &x[0], &b[0], &out[0], 2, 2);
  for (int i = 0; i < 2; ++i)
    assert_eqf(out[i], res[i], "Dot add identity index %d", i);
}

void test_dot_add_sum(void) {

  FLOAT W[4] = {1., 1., 1., 1.};
  FLOAT x[2] = {2., 3.};
  FLOAT b[2] = {3., 7.};
  FLOAT out[2] = {0};
  FLOAT res[2] = {8., 12.};
  dot_add(&W[0], &x[0], &b[0], &out[0], 2, 2);
  for (int i = 0; i < 2; ++i)
    assert_eqf(out[i], res[i], "Dot add sum index %d", i);
}

void test_dot_add_permutation(void) {

  FLOAT W[4] = {0., 1., 1., 0.};
  FLOAT x[2] = {2., 3.};
  FLOAT b[2] = {3., 7.};
  FLOAT out[2] = {0};
  FLOAT res[2] = {6., 9.};
  dot_add(&W[0], &x[0], &b[0], &out[0], 2, 2);
  for (int i = 0; i < 2; ++i)
    assert_eqf(out[i], res[i], "Dot add permutation index %d", i);
}

void test_dot_add_non_symmetric(void) {

  FLOAT W[6] = {1., 2., 3., 4., 5., 6.};
  FLOAT x[3] = {7., 8., 9.};
  FLOAT b[2] = {1., 2.};
  FLOAT out[2] = {0};
  FLOAT res[2] = {51., 124.};
  dot_add(&W[0], &x[0], &b[0], &out[0], 2, 3);
  for (int i = 0; i < 2; ++i)
    assert_eqf(out[i], res[i], "Dot add non-symmetric index %d", i);
}

void test_sigmoid_single(void) {
  FLOAT z = 0;
  assert_eqf(sigmoid_s(z), 0.5, "Single sigmoid zero");
  z = 0.3;
  assert_eqf(sigmoid_s(z), 0.5744425168116848, "Single sigmoid positive");
  z = -0.4;
  assert_eqf(sigmoid_s(z), 0.40131233988751425, "Single sigmoid negative");
}

void test_sigmoid_multi(void) {
  FLOAT z[3] = {0., 0.3, -0.4};
  FLOAT res[3] = {0.5, 0.5744425168116848, 0.40131233988751425};

  FLOAT out[3] = {0};
  sigmoid(&z[0], &out[0], 3);
  for (int i = 0; i < 3; ++i)
    assert_eqf(out[i], res[i], "Multi sigmoid value %d", i);
}

void test_sigmoid_prime_single(void) {
  FLOAT z = 0;
  assert_eqf(sigmoid_prime_s(z), 0.25, "Single sigmoid_prime zero");
  z = 0.3;
  assert_eqf(sigmoid_prime_s(z), 0.24445831169074203,
             "Single sigmoid_prime positive");
  z = -0.4;
  assert_eqf(sigmoid_prime_s(z), 0.24026074574152248,
             "Single sigmoid_prime negative");
}

void test_distance_squared_zero(void) {
  FLOAT x[3] = {1., 2., 3.};
  FLOAT y[3] = {1., 2., 3.};
  assert_eqf(distance_squared(&x[0], &y[0], 3), 0., "Zero distance");
}

void test_distance_squared(void) {
  FLOAT x[3] = {1., 2., -3.};
  FLOAT y[3] = {-3., 1., 2.};
  assert_eqf(distance_squared(&x[0], &y[0], 3), 42., "Zero distance");
}

void check_random_network(const DS_Network *const network,
                          const size_t num_layers, const size_t *const sizes,
                          char *const *const labels) {
  assert_eqlu(network->num_layers, num_layers, "Network num layers");
  for (size_t l = 0; l < num_layers; ++l)
    assert_eqlu(network->layer_sizes[l], sizes[l], "Layer sizes index %lu", l);
  if (labels) {
    assert_neqp(network->output_labels, NULL, "Output labels are NULL.");
    const size_t L = sizes[num_layers - 1];
    for (size_t i = 0; i < L; ++i)
      assert_eqstr(network->output_labels[i], labels[i],
                   "Output label index %lu", i);
  } else {
    assert_eqp(network->output_labels, NULL, "Output labels are not NULL.");
  }

  // Check for memory segfaults
  for (size_t l = 0; l < num_layers - 1; ++l) {
    size_t m = network->layer_sizes[l];
    size_t n = network->layer_sizes[l + 1];
    for (size_t i = 0; i < n; ++i) {
      volatile FLOAT b = network->biases[l][i];
      (void)b;
      for (size_t j = 0; j < m; ++j) {
        volatile FLOAT w = network->weights[l][IDX(i, j, m)];
        (void)w;
      }
    }
  }
  for (size_t l = 0; l < num_layers; ++l) {
    size_t n = network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      volatile FLOAT a = network->result->activations[l][i];
      (void)a;
      volatile FLOAT in = network->result->inputs[l][i];
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
const FLOAT WEIGHT_1[LAYER_1 * LAYER_2] = {.1, .2, .3, .4, .5, .6};
const FLOAT WEIGHT_2[LAYER_2 * LAYER_3] = {.6, .5, .4, .3, .2, .1};
const FLOAT BIAS_1[LAYER_2] = {.1, .2, .3};
const FLOAT BIAS_2[LAYER_3] = {.4, .5};
const size_t LAYER_SIZES[NUM_LAYERS] = {LAYER_1, LAYER_2, LAYER_3};
const FLOAT *WEIGHTS[NUM_LAYERS - 1] = {&WEIGHT_1[0], &WEIGHT_2[0]};
const FLOAT *BIASES[NUM_LAYERS - 1] = {&BIAS_1[0], &BIAS_2[0]};
char *OUTPUT_LABELS[LAYER_3] = {"First Label", "Second Label"};

DS_Network *create_test_network(void) {
  return DS_network_create(&WEIGHTS[0], &BIASES[0], LAYER_SIZES, NUM_LAYERS,
                           OUTPUT_LABELS);
}

void check_network_empty_label_correctness(const DS_Network *const network) {
  assert_eqp(network->output_labels, NULL, "Output labels are not NULL.");
}

void check_network_label_correctness(const DS_Network *const network) {
  assert_neqp(network->output_labels, NULL, "Output labels are NULL.");
  const size_t L = LAYER_SIZES[NUM_LAYERS - 1];
  for (size_t i = 0; i < L; ++i)
    assert_eqstr(network->output_labels[i], OUTPUT_LABELS[i],
                 "Output label index %lu", i);
}

void check_network_correctness(const DS_Network *const network) {
  assert_eqlu((size_t)NUM_LAYERS, network->num_layers,
              "Wrong number of layers.");
  for (size_t l = 0; l < network->num_layers; ++l)
    assert_eqlu(network->layer_sizes[l], LAYER_SIZES[l],
                "Not the same layer sizes for layer %lu.", l);

  for (size_t l = 0; l < network->num_layers - 1; ++l) {
    size_t n = network->layer_sizes[l + 1];
    size_t m = network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j)
        assert_eqf(network->weights[l][IDX(i, j, m)], WEIGHTS[l][IDX(i, j, m)],
                   "Weight for layer %lu, index i=%lu, j=%lu", l + 1, i, j);
      assert_eqf(network->biases[l][i], BIASES[l][i],
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

  size_t *sizes = MALLOC(NUM_LAYERS * sizeof(sizes[0]));
  memcpy(sizes, LAYER_SIZES, NUM_LAYERS * sizeof(sizes[0]));
  FLOAT **biases = MALLOC((NUM_LAYERS - 1) * sizeof(biases[0]));
  FLOAT **weights = MALLOC((NUM_LAYERS - 1) * sizeof(weights[0]));
  for (size_t l = 0; l < NUM_LAYERS - 1; ++l) {
    biases[l] = MALLOC(LAYER_SIZES[l + 1] * sizeof(biases[l][0]));
    memcpy(biases[l], BIASES[l], LAYER_SIZES[l + 1] * sizeof(biases[l][0]));
    weights[l] =
        MALLOC(LAYER_SIZES[l] * LAYER_SIZES[l + 1] * sizeof(weights[l][0]));
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
  const FLOAT input[LAYER_1] = {.1, .2};
  const FLOAT res_input_1[LAYER_1] = {.1, .2};
  const FLOAT res_activation_1[LAYER_1] = {.1, .2};
  const FLOAT res_input_2[LAYER_2] = {0.15, 0.31, 0.47};
  const FLOAT res_activation_2[LAYER_2] = {0.53742985, 0.57688526, 0.61538376};
  const FLOAT res_input_3[LAYER_3] = {1.25705404, 0.83814438};
  const FLOAT res_activation_3[LAYER_3] = {0.77851856, 0.69807426};

  const FLOAT *res_inputs[NUM_LAYERS] = {&res_input_1[0], &res_input_2[0],
                                         &res_input_3[0]};
  const FLOAT *res_activations[NUM_LAYERS] = {
      &res_activation_1[0], &res_activation_2[0], &res_activation_3[0]};

  DS_network_feedforward(network, &input[0]);

  for (size_t l = 0; l < NUM_LAYERS; ++l) {
    size_t n = network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      assert_eqf(network->result->inputs[l][i], res_inputs[l][i],
                 "Input for layer %lu for index %lu", l, i);
      assert_eqf(network->result->activations[l][i], res_activations[l][i],
                 "Activation for layer %lu for index %lu", l, i);
    }
  }
  DS_network_free(network);
}

void test_network_backprop_last_error(void) {
  FLOAT a = 0.3;
  FLOAT z = 0.5;
  FLOAT y = 0.3;
  assert_eqf(last_output_error_s(a, z, y), 0., "Last output error: No error.");

  a = 0.8;
  z = 0.3;
  y = 0.1;
  assert_eqf(last_output_error_s(a, z, y), 0.17112081818352212,
             "Last output error: Positive error.");
  a = 0.7;
  z = 0.8;
  y = 0.9;
  assert_eqf(last_output_error_s(a, z, y), -0.042781939304058894,
             "Last output error: Negative error.");
}

void check_backprop_segfauls(const DS_Backprop *const backprop) {

  // Check for memory segfaults
  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    size_t m = backprop->network->layer_sizes[l];
    size_t n = backprop->network->layer_sizes[l + 1];
    for (size_t i = 0; i < n; ++i) {
      volatile FLOAT b = backprop->bias_error_sums[l][i];
      (void)b;
      for (size_t j = 0; j < m; ++j) {
        volatile FLOAT w = backprop->weight_error_sums[l][IDX(i, j, m)];
        (void)w;
      }
    }
  }
  for (size_t l = 0; l < backprop->network->num_layers; ++l) {
    size_t n = backprop->network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      volatile FLOAT e = backprop->errors[l][i];
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
  FLOAT learning_rate = 1;
  size_t num_training = 1;
  FLOAT x[LAYER_1] = {0.2, 0.1};
  FLOAT y[LAYER_1] = {0.5, -0.3};
  FLOAT *xs[1] = {&x[0]};
  FLOAT *ys[1] = {&y[0]};

  FLOAT error_bias_1[LAYER_2] = {0.02287103, 0.01615626, 0.0095477};
  FLOAT error_bias_2[LAYER_3] = {0.04801298, 0.21041785};
  FLOAT error_weight_1[LAYER_1 * LAYER_2] = {
      0.00457421, 0.0022871, 0.00323125, 0.00161563, 0.00190954, 0.00095477};
  FLOAT error_weight_2[LAYER_2 * LAYER_3] = {
      0.0256842, 0.0275807, 0.02943264, 0.11256154, 0.12087296, 0.12898913};

  FLOAT *error_biases[NUM_LAYERS - 1] = {&error_bias_1[0], &error_bias_2[0]};
  FLOAT *error_weights[NUM_LAYERS - 1] = {&error_weight_1[0],
                                          &error_weight_2[0]};

  DS_backprop_learn_once(backprop, xs, ys, num_training, learning_rate);

  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    size_t n = backprop->network->layer_sizes[l + 1];
    size_t m = backprop->network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j)
        assert_eqf(backprop->weight_error_sums[l][IDX(i, j, m)],
                   error_weights[l][IDX(i, j, m)],
                   "Weight errors l=%lu, i=%lu, j=%lu", l, i, j);

      assert_eqf(backprop->bias_error_sums[l][i], error_biases[l][i],
                 "Bias error l=%lu, i=%lu", l, i);
    }
  }

  DS_backprop_free(backprop);
}

void test_network_backprop_double_input(void) {
  DS_Backprop *backprop =
      DS_brackprop_create_from_network(create_test_network());
  FLOAT learning_rate = 0.8;
  size_t num_training = 2;

  FLOAT x1[LAYER_1] = {0.3, 0.2};
  FLOAT x2[LAYER_1] = {0.4, 0.5};
  FLOAT y1[LAYER_1] = {0.1, -0.2};
  FLOAT y2[LAYER_1] = {-0.3, 0.7};
  FLOAT *xs[2] = {&x1[0], &x2[0]};
  FLOAT *ys[2] = {&y1[0], &y2[0]};
  FLOAT bias_1[LAYER_2] = {0.07672889, 0.1822576, 0.28776812};
  FLOAT bias_2[LAYER_3] = {0.28116617, 0.42410679};
  FLOAT weight_1[LAYER_1 * LAYER_2] = {0.09194743, 0.19213205, 0.29383053,
                                       0.39391128, 0.49571109, 0.5956956};
  FLOAT weight_2[LAYER_2 * LAYER_3] = {0.53429254, 0.42713372, 0.32038983,
                                       0.25883112, 0.15510289, 0.05150874};

  FLOAT *biases[NUM_LAYERS - 1] = {&bias_1[0], &bias_2[0]};
  FLOAT *weights[NUM_LAYERS - 1] = {&weight_1[0], &weight_2[0]};

  DS_backprop_learn_once(backprop, xs, ys, num_training, learning_rate);

  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    size_t n = backprop->network->layer_sizes[l + 1];
    size_t m = backprop->network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j)
        assert_eqf(backprop->network->weights[l][IDX(i, j, m)],
                   weights[l][IDX(i, j, m)], "Weight l=%lu, index i=%lu, j=%lu",
                   l, i, j);

      assert_eqf(backprop->network->biases[l][i], biases[l][i],
                 "Bias l=%lu, i=%lu", l, i);
    }
  }

  DS_backprop_free(backprop);
}

void test_network_backprop_double_input_twice(void) {
  DS_Backprop *backprop =
      DS_brackprop_create_from_network(create_test_network());
  FLOAT learning_rate = 0.8;
  size_t num_training = 2;

  FLOAT x1[LAYER_1] = {0.3, 0.2};
  FLOAT x2[LAYER_1] = {0.4, 0.5};
  FLOAT y1[LAYER_1] = {0.1, -0.2};
  FLOAT y2[LAYER_1] = {-0.3, 0.7};
  FLOAT *xs[2] = {&x1[0], &x2[0]};
  FLOAT *ys[2] = {&y1[0], &y2[0]};

  FLOAT bias_1[LAYER_2] = {0.05485079, 0.16633528, 0.27771601};
  FLOAT bias_2[LAYER_3] = {0.15204091, 0.34987652};
  FLOAT weight_1[LAYER_1 * LAYER_2] = {0.08434973, 0.18465363, 0.28826394,
                                       0.38835713, 0.49214933, 0.59204678};
  FLOAT weight_2[LAYER_2 * LAYER_3] = {0.46380924, 0.34860201, 0.23426936,
                                       0.21911876, 0.11167309, 0.00448532};

  FLOAT *biases[NUM_LAYERS - 1] = {&bias_1[0], &bias_2[0]};
  FLOAT *weights[NUM_LAYERS - 1] = {&weight_1[0], &weight_2[0]};

  DS_backprop_learn_once(backprop, xs, ys, num_training, learning_rate);
  DS_backprop_learn_once(backprop, xs, ys, num_training, learning_rate);

  for (size_t l = 0; l < backprop->network->num_layers - 1; ++l) {
    size_t n = backprop->network->layer_sizes[l + 1];
    size_t m = backprop->network->layer_sizes[l];
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j)
        assert_eqf(backprop->network->weights[l][IDX(i, j, m)],
                   weights[l][IDX(i, j, m)], "Weight l=%lu, index i=%lu, j=%lu",
                   l, i, j);

      assert_eqf(backprop->network->biases[l][i], biases[l][i],
                 "Bias l=%lu, i=%lu", l, i);
    }
  }

  DS_backprop_free(backprop);
}

RUN_TESTS(test_sigmoid_single, test_sigmoid_multi, test_sigmoid_prime_single,
          test_dot_add_identity, test_dot_add_sum, test_dot_add_permutation,
          test_idx, test_distance_squared_zero, test_distance_squared,
          test_dot_add_non_symmetric, test_network_creation_random,
          test_create_test_network, test_create_test_network_owned,
          test_network_feedforward, test_backprop_create,
          test_backprop_create_from_network, test_network_backprop_last_error,
          test_network_backprop_error_sums_single_input,
          test_network_backprop_double_input,
          test_network_backprop_double_input_twice)
