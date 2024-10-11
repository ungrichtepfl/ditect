#include "deepsea.c"
#include "see.h"

void sigmoid_test(void) {
  int x = 1;
  int y = 2;
  assert_eqi(x, y, "Assertion failed");
}
void sigmoid_test2(void) {
  float x = 1.1;
  float y = 2.;
  assert_eqf(x + x, y, "Assertion failed");
}

RUN_TESTS(sigmoid_test, sigmoid_test2)
