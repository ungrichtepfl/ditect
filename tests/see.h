#ifndef SEE_H
#define SEE_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static bool _test_failed = false;

#define _RED() printf("\033[0;31m")
#define _GREEN() printf("\033[0;32m")
#define _YELLOW() printf("\033[0;33m")
#define _RESET() printf("\033[0m")

#define assert(cond, ...)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      _test_failed = true;                                                     \
      _RED();                                                                  \
      printf("TEST FAILED ("__FILE__                                           \
             ": %d): ",                                                        \
             __LINE__);                                                        \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      printf("   \"%s\" is False!\n", #cond);                                  \
      _RESET();                                                                \
    }                                                                          \
  } while (0)

#define assert_eq(exp1, exp2, cond, type, ...)                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      _test_failed = true;                                                     \
      _RED();                                                                  \
      printf("TEST FAILED ("__FILE__                                           \
             ": %d): ",                                                        \
             __LINE__);                                                        \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      printf("   \"%s == %s\" not equal (%" type " != %" type ")!\n", #exp1,   \
             #exp2, (exp1), (exp2));                                           \
      _RESET();                                                                \
    }                                                                          \
  } while (0)

#define assert_eqi(exp1, exp2, ...)                                            \
  assert_eq(exp1, exp1, (exp1) == (exp1), "d", __VA_ARGS__)

#define assert_eqlu(exp1, exp2, ...)                                           \
  assert_eq(exp1, exp1, (exp1) == (exp1), "lu", __VA_ARGS__)

#define SEE_EPSILON 1.e-5
#define assert_eqf(exp1, exp2, ...)                                            \
  assert_eq(exp1, exp1, fabs((exp1) - (exp2)) < SEE_EPSILON, "f", __VA_ARGS__)

static inline int _run_tests(void (*tests[])(void), const unsigned long n) {
  if (!n) {
    _YELLOW();
    printf("NO TESTS TO EXECUTE!\n");
    _RESET();
    return 0;
  }

  unsigned long failed_tests = 0;
  printf("---------------------------------------------------------------------"
         "--\n");
  for (unsigned long i = 0; i < n; ++i) {
    _test_failed = false;
    tests[i]();
    if (_test_failed) {
      _RED();
      printf("Test %lu FAILED\n", i + 1);
      _RESET();
      ++failed_tests;
    } else {
      _GREEN();
      printf("Test %lu PASSED\n", i + 1);
      _RESET();
    }
  }
  printf("---------------------------------------------------------------------"
         "--\n");
  if (failed_tests) {
    _RED();
    if (failed_tests == 1)
      printf("FAILURE: 1 test failed!\n");
    else
      printf("FAILURE: %lu tests failed!\n", failed_tests);
    _RESET();
  } else {
    _GREEN();
    printf("SUCCESS: All tests passed!\n");
    _RESET();
  }
  return failed_tests;
}

#define RUN_TESTS(...)                                                         \
  int main(void) {                                                             \
    void (*tests[])(void) = {__VA_ARGS__};                                     \
    const unsigned long n = sizeof(tests) / sizeof(tests[0]);                  \
    return _run_tests(tests, n);                                               \
  }

#endif // SEE_H
