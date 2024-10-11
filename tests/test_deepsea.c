#include "deepsea.c"

int sigmoid_test(void) { return 0; }

int main(void) {
  int (*tests[])(void) = {sigmoid_test};

  unsigned long failed_tests = 0;
  for (unsigned long i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    if (tests[i]()) {
      printf("Test %lul failed.\n", i);
      ++failed_tests;
    }
  }

  if (failed_tests) {
    printf("FAILURE: %lul tests failed\n", failed_tests);
  } else {
    printf("SUCCESS: All tests passed!\n");
  }

  return failed_tests;
}
