#ifndef SEE_H
#define SEE_H

#include <alloca.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool _test_failed = false;
static char *_current_test = NULL;

#define _RED() printf("\033[0;31m")
#define _GREEN() printf("\033[0;32m")
#define _YELLOW() printf("\033[0;33m")
#define _RESET() printf("\033[0m")

#define _STR_ARGS(...) #__VA_ARGS__

#define assert(cond, ...)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      _test_failed = true;                                                     \
      _RED();                                                                  \
      printf("ASSERTION IN TEST \"%s\" FAILED ("__FILE__                       \
             ": %d): ",                                                        \
             _current_test, __LINE__);                                         \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      printf("     \"%s\" is False!\n", #cond);                                \
      _RESET();                                                                \
    }                                                                          \
  } while (0)

#define assert_eq(exp1, exp2, cond, type, ...)                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      _test_failed = true;                                                     \
      _RED();                                                                  \
      printf("  ASSERTION IN TEST \"%s\" FAILED ("__FILE__                     \
             ": %d): ",                                                        \
             _current_test, __LINE__);                                         \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      printf("     \"%s == %s\" not equal (%" type " != %" type ")!\n", #exp1, \
             #exp2, (exp1), (exp2));                                           \
      _RESET();                                                                \
    }                                                                          \
  } while (0)

#define assert_eq_fun(exp1, exp2, fun, type, ...)                              \
  assert_eq(exp1, exp2, fun(), type, __VA_ARGS__)

#define assert_eqi(exp1, exp2, ...)                                            \
  assert_eq(exp1, exp2, (exp1) == (exp2), "d", __VA_ARGS__)

#define assert_eqlu(exp1, exp2, ...)                                           \
  assert_eq(exp1, exp2, (exp1) == (exp2), "lu", __VA_ARGS__)

#define SEE_EPSILON 1.e-8
#define assert_eqf(exp1, exp2, ...)                                            \
  assert_eq(exp1, exp2, fabs((exp1) - (exp2)) < SEE_EPSILON, "f", __VA_ARGS__)

#define assert_eqf_eps(exp1, exp2, eps, ...)                                   \
  assert_eq(exp1, exp2, fabs((exp1) - (exp2)) < (eps), "f", __VA_ARGS__)

static inline int _run_tests(void (*tests[])(void), const unsigned long n,
                             const char *test_names_str) {
  if (!n) {
    _YELLOW();
    printf("NO TESTS TO EXECUTE!\n");
    _RESET();
    return 0;
  }

  // Extract names
  unsigned long test_names_str_len = strlen(test_names_str);
  char *test_names_str_buffer = alloca(test_names_str_len * sizeof(char));
  strcpy(test_names_str_buffer, test_names_str);
  char *test_names[n];
  unsigned long i = 0;
  for (char *name = strtok(test_names_str_buffer, ", "); name != NULL && i < n;
       name = strtok(NULL, ", "), ++i) {
    test_names[i] = name;
  }

  unsigned long failed_tests = 0;
  printf("---------------------------------------------------------------------"
         "--\n");
  for (unsigned long i = 0; i < n; ++i) {
    _test_failed = false;
    _current_test = test_names[i];
    tests[i]();
    if (_test_failed) {
      _RED();
      printf("Test \"%s\" FAILED\n", _current_test);
      _RESET();
      ++failed_tests;
    } else {
      _GREEN();
      printf("Test \"%s\" PASSED\n", _current_test);
      _RESET();
    }
  }
  printf("---------------------------------------------------------------------"
         "--\n");
  if (failed_tests) {
    _RED();
    printf("FAILURE: %lu of %lu tests failed!\n", failed_tests, n);
    _RESET();
  } else {
    _GREEN();
    if (n == 1)
      printf("SUCCESS: Tests passed!\n");
    else
      printf("SUCCESS: All %lu tests passed!\n", n);
    _RESET();
  }
  return failed_tests;
}

#define RUN_TESTS(...)                                                         \
  int main(void) {                                                             \
    void (*tests[])(void) = {__VA_ARGS__};                                     \
    const unsigned long n = sizeof(tests) / sizeof(tests[0]);                  \
    return _run_tests(tests, n, _STR_ARGS(__VA_ARGS__));                       \
  }

#endif // SEE_H
