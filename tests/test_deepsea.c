#include "deepsea.c"
#include "see.h"

void sigmoid_single(void) {
  FLOAT z = 0;
  assert_eqf(sigmoid_s(z), 0.5, "Single sigmoid zero");
  z = 0.3;
  assert_eqf(sigmoid_s(z), 0.5744425168116848, "Single sigmoid positiv");
  z = -0.4;
  assert_eqf(sigmoid_s(z), 0.40131233988751425, "Single sigmoid negativ");
}
void sigmoid_multi(void) {
  FLOAT z[3] = {0., 0.3, -0.4};
  FLOAT res[3] = {0.5, 0.5744425168116848, 0.40131233988751425};

  FLOAT out[3] = {0};
  sigmoid(&z[0], &out[0], 3);
  for (int i = 0; i < 3; ++i)
    assert_eqf(out[i], res[i], "Multi sigmoid value %d", i);
}

RUN_TESTS(sigmoid_single, sigmoid_multi)
