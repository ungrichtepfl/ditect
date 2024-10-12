#include "deepsea.c"
#include "see.h"

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
  assert_eqf(sigmoid_s(z), 0.5744425168116848, "Single sigmoid positiv");
  z = -0.4;
  assert_eqf(sigmoid_s(z), 0.40131233988751425, "Single sigmoid negativ");
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
             "Single sigmoid_prime positiv");
  z = -0.4;
  assert_eqf(sigmoid_prime_s(z), 0.24026074574152248,
             "Single sigmoid_prime negativ");
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

void test_network_creation_random(void) {
  size_t num_layers = 3;
  size_t sizes[3] = {2, 5, 7};
  DS_Network *network = DS_network_create_random(&sizes[0], num_layers);
  assert_eqlu(network->num_layers, num_layers, "Network num layers");
  for (size_t i = 0; i < num_layers; ++i)
    assert_eqlu(network->layer_sizes[i], sizes[i], "Layer sizes index %lu", i);

  // Check for memeory segfaults
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
  DS_network_free(network);
}

RUN_TESTS(test_sigmoid_single, test_sigmoid_multi, test_sigmoid_prime_single,
          test_dot_add_identity, test_dot_add_sum, test_dot_add_permutation,
          test_idx, test_distance_squared_zero, test_distance_squared,
          test_dot_add_non_symmetric, test_network_creation_random)
