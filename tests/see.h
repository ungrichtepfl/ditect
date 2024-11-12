#ifndef SEE_H
#define SEE_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SEE_MALLOC
#define SEE_MALLOC malloc
#endif
#ifndef SEE_FREE
#define SEE_FREE free
#endif

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
      printf("     \"%s == %s\" but not equal (%" type " != %" type ")!\n",    \
             #exp1, #exp2, (exp1), (exp2));                                    \
      _RESET();                                                                \
    }                                                                          \
  } while (0)

#define assert_neq(exp1, exp2, cond, type, ...)                                \
  do {                                                                         \
    if ((cond)) {                                                              \
      _test_failed = true;                                                     \
      _RED();                                                                  \
      printf("  ASSERTION IN TEST \"%s\" FAILED ("__FILE__                     \
             ": %d): ",                                                        \
             _current_test, __LINE__);                                         \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      printf("     \"%s != %s\" but equal (%" type " == %" type ")!\n", #exp1, \
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

#define assert_eqp(exp1, exp2, ...)                                            \
  assert_eq((void *)(exp1), (void *)(exp2), (exp1) == (exp2), "p", __VA_ARGS__)

#define assert_neqp(exp1, exp2, ...)                                           \
  assert_neq((void *)(exp1), (void *)(exp2), (exp1) == (exp2), "p", __VA_ARGS__)

#define assert_eqstr(str1, str2, ...)                                          \
  assert_eq(str1, str2, strcmp((str1), (str2)) == 0, "s", __VA_ARGS__)

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
  unsigned long test_names_str_len = strlen(test_names_str) + 1;
  char *test_names_str_buffer = SEE_MALLOC(test_names_str_len * sizeof(char));
  if (!test_names_str_buffer) {
    fprintf(stderr, "Could not run tests. Out of memory");
    return 1;
  }
  strcpy(test_names_str_buffer, test_names_str);
  char *test_names[n];
  unsigned long i = 0;
  for (char *name = strtok(test_names_str_buffer, ", "); name != NULL && i < n;
       name = strtok(NULL, ", "), ++i) {
    test_names[i] = name;
  }

  unsigned long num_failed_tests = 0;
  char *failed_tests[n];

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
      failed_tests[num_failed_tests++] = _current_test;
    } else {
      _GREEN();
      printf("Test \"%s\" PASSED\n", _current_test);
      _RESET();
    }
  }
  printf("---------------------------------------------------------------------"
         "--\n");
  if (num_failed_tests) {
    _RED();
    printf("FAILURE: %lu of %lu tests failed!\n", num_failed_tests, n);
    for (unsigned int i = 0; i < num_failed_tests; ++i)
      printf("  TEST \"%s\" FAILED\n", failed_tests[i]);
    _RESET();
  } else {
    _GREEN();
    if (n == 1)
      printf("SUCCESS: Tests passed!\n");
    else
      printf("SUCCESS: All %lu tests passed!\n", n);
    _RESET();
  }

  SEE_FREE(test_names_str_buffer);
  return num_failed_tests;
}

#define RUN_TESTS(...)                                                         \
  int main(void) {                                                             \
    void (*tests[])(void) = {__VA_ARGS__};                                     \
    const unsigned long n = sizeof(tests) / sizeof(tests[0]);                  \
    return _run_tests(tests, n, _STR_ARGS(__VA_ARGS__));                       \
  }

#endif // SEE_H
