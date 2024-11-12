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
#ifndef SEE_CALLOC
#define SEE_CALLOC calloc
#endif
#ifndef SEE_FREE
#define SEE_FREE free
#endif

typedef struct _SEE_Allocation {
  void *ptr;
  size_t size;
  struct _SEE_Allocation *next;
} _SEE_Allocation;

static _SEE_Allocation *_SEE_allocations = NULL;
static int _SEE_free_untracked = 0;

#define _SEE_RED() printf("\033[0;31m")
#define _SEE_GREEN() printf("\033[0;32m")
#define _SEE_YELLOW() printf("\033[0;33m")
#define _SEE_BLUE() printf("\033[0;34m")
#define _SEE_RESET() printf("\033[0m")

static inline void _SEE_add_allocation(void *ptr, size_t size) {
  _SEE_Allocation *new_alloc = SEE_MALLOC(sizeof(*new_alloc));
  if (!new_alloc) {
    _SEE_YELLOW();
    printf("WARNING: Cannot track allocations anymore in debug malloc and "
           "calloc. Out of memory.\n");
    _SEE_RESET();
    return;
  }
  new_alloc->ptr = ptr;
  new_alloc->size = size;
  new_alloc->next = _SEE_allocations;
  _SEE_allocations = new_alloc;
}

// Helper function to remove an allocation record
static inline void _SEE_remove_allocation(void *ptr) {
  _SEE_Allocation **current = &_SEE_allocations;
  while (*current) {
    if ((*current)->ptr == ptr) {
      _SEE_Allocation *to_free = *current;
      *current = (*current)->next;
      SEE_FREE(to_free);
      return;
    }
    current = &(*current)->next;
  }
  // If the pointer is not found, it means an attempt to free untracked memory
  ++_SEE_free_untracked;
  _SEE_RED();
  printf("ERROR: Attempted to free untracked memory: %p\n", ptr);
  _SEE_RESET();
}

static inline void *_SEE_debug_malloc(size_t size) {
  void *ptr = SEE_MALLOC(size);
  if (ptr) {
    _SEE_add_allocation(ptr, size);
  }
  return ptr;
}

static inline void *_SEE_debug_calloc(size_t num, size_t size) {
  void *ptr = SEE_CALLOC(num, size);
  if (ptr) {
    _SEE_add_allocation(ptr, num * size);
  }
  return ptr;
}

static inline void _SEE_debug_free(void *ptr) {
  if (ptr) {
    _SEE_remove_allocation(ptr);
    SEE_FREE(ptr);
  }
}

static inline int _SEE_check_memory(void) {
  _SEE_Allocation *current = _SEE_allocations;
  int leaks = 0;
  size_t total_bytes = 0;
  while (current) {
    _SEE_RED();
    printf("ERROR: Memory leak detected: %p, size: %zu bytes\n", current->ptr,
           current->size);
    _SEE_RESET();
    ++leaks;
    total_bytes += current->size;
    current = current->next;
  }
  if (leaks == 0) {
    _SEE_GREEN();
    printf("SUCCESS: No memory leaks detected!\n");
    _SEE_RESET();
  } else if (leaks == 1) {
    _SEE_RED();
    printf("ERROR: 1 memory leak detected (%lu bytes)!\n", total_bytes);
    _SEE_RESET();
  } else {
    _SEE_RED();
    printf("ERROR: %d memory leaks detected (%lu bytes in total)!\n", leaks,
           total_bytes);
    _SEE_RESET();
  }
  if (_SEE_free_untracked == 0) {
    _SEE_GREEN();
    printf("SUCCESS: No untracked frees detected!\n");
    _SEE_RESET();
  } else if (_SEE_free_untracked == 1) {
    _SEE_RED();
    printf("ERROR: 1 untracked free!\n");
    _SEE_RESET();
  } else {
    _SEE_RED();
    printf("ERROR: %d untracked frees detected!\n", _SEE_free_untracked);
    _SEE_RESET();
  }
  if (leaks > 0 || _SEE_free_untracked > 0) {
    _SEE_BLUE();
    printf("NOTE: Use valgrind or debugger to find file and line number of "
           "memory issues.\n");
    _SEE_RESET();
  }
  return leaks + _SEE_free_untracked;
}

#define SEE_DEBUG_MALLOC _SEE_debug_malloc
#define SEE_DEBUG_CALLOC _SEE_debug_calloc
#define SEE_DEBUG_FREE _SEE_debug_free

static bool _SEE_test_failed = false;
static char *_SEE_current_test = NULL;

#define _SEE_STR_ARGS(...) #__VA_ARGS__

#define SEE_assert(cond, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      _SEE_test_failed = true;                                                 \
      _SEE_RED();                                                              \
      printf("ASSERTION IN TEST \"%s\" FAILED ("__FILE__                       \
             ": %d): ",                                                        \
             _SEE_current_test, __LINE__);                                     \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      printf("     \"%s\" is False!\n", #cond);                                \
      _SEE_RESET();                                                            \
    }                                                                          \
  } while (0)

#define SEE_assert_eq(exp1, exp2, cond, type, ...)                             \
  do {                                                                         \
    if (!(cond)) {                                                             \
      _SEE_test_failed = true;                                                 \
      _SEE_RED();                                                              \
      printf("  ASSERTION IN TEST \"%s\" FAILED ("__FILE__                     \
             ": %d): ",                                                        \
             _SEE_current_test, __LINE__);                                     \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      printf("     \"%s == %s\" but not equal (%" type " != %" type ")!\n",    \
             #exp1, #exp2, (exp1), (exp2));                                    \
      _SEE_RESET();                                                            \
    }                                                                          \
  } while (0)

#define SEE_assert_neq(exp1, exp2, cond, type, ...)                            \
  do {                                                                         \
    if ((cond)) {                                                              \
      _SEE_test_failed = true;                                                 \
      _SEE_RED();                                                              \
      printf("  ASSERTION IN TEST \"%s\" FAILED ("__FILE__                     \
             ": %d): ",                                                        \
             _SEE_current_test, __LINE__);                                     \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      printf("     \"%s != %s\" but equal (%" type " == %" type ")!\n", #exp1, \
             #exp2, (exp1), (exp2));                                           \
      _SEE_RESET();                                                            \
    }                                                                          \
  } while (0)

#define SEE_assert_eq_fun(exp1, exp2, fun, type, ...)                          \
  SEE_assert_eq(exp1, exp2, fun(), type, __VA_ARGS__)

#define SEE_assert_eqi(exp1, exp2, ...)                                        \
  SEE_assert_eq(exp1, exp2, (exp1) == (exp2), "d", __VA_ARGS__)

#define SEE_assert_eqlu(exp1, exp2, ...)                                       \
  SEE_assert_eq(exp1, exp2, (exp1) == (exp2), "lu", __VA_ARGS__)

#define SEE_assert_eqp(exp1, exp2, ...)                                        \
  SEE_assert_eq((void *)(exp1), (void *)(exp2), (exp1) == (exp2), "p",         \
                __VA_ARGS__)

#define SEE_assert_neqp(exp1, exp2, ...)                                       \
  SEE_assert_neq((void *)(exp1), (void *)(exp2), (exp1) == (exp2), "p",        \
                 __VA_ARGS__)

#define SEE_assert_eqstr(str1, str2, ...)                                      \
  SEE_assert_eq(str1, str2, strcmp((str1), (str2)) == 0, "s", __VA_ARGS__)

#define SEE_EPSILON 1.e-8
#define SEE_assert_eqf(exp1, exp2, ...)                                        \
  SEE_assert_eq(exp1, exp2, fabs((exp1) - (exp2)) < SEE_EPSILON, "f",          \
                __VA_ARGS__)

#define SEE_assert_eqf_eps(exp1, exp2, eps, ...)                               \
  SEE_assert_eq(exp1, exp2, fabs((exp1) - (exp2)) < (eps), "f", __VA_ARGS__)

static inline int _SEE_run_tests(void (*tests[])(void), const unsigned long n,
                                 const char *test_names_str) {
  if (!n) {
    _SEE_YELLOW();
    printf("NO TESTS TO EXECUTE!\n");
    _SEE_RESET();
    return 0;
  }

  // Extract names
  unsigned long test_names_str_len = strlen(test_names_str) + 1;
  char *test_names_str_buffer = SEE_MALLOC(test_names_str_len * sizeof(char));
  if (!test_names_str_buffer) {
    fprintf(stderr, "Could not run tests. Could not allocate memory to "
                    "stringify all the test names. Out of memory.");
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
    _SEE_test_failed = false;
    _SEE_current_test = test_names[i];
    tests[i]();
    if (_SEE_test_failed) {
      _SEE_RED();
      printf("Test \"%s\" FAILED\n", _SEE_current_test);
      _SEE_RESET();
      failed_tests[num_failed_tests++] = _SEE_current_test;
    } else {
      _SEE_GREEN();
      printf("Test \"%s\" PASSED\n", _SEE_current_test);
      _SEE_RESET();
    }
  }
  printf("---------------------------------------------------------------------"
         "--\n");
  if (num_failed_tests) {
    _SEE_RED();
    printf("FAILURE: %lu of %lu tests failed!\n", num_failed_tests, n);
    for (unsigned int i = 0; i < num_failed_tests; ++i)
      printf("  TEST \"%s\" FAILED\n", failed_tests[i]);
    _SEE_RESET();
  } else {
    _SEE_GREEN();
    if (n == 1)
      printf("SUCCESS: Tests passed!\n");
    else
      printf("SUCCESS: All %lu tests passed!\n", n);
    _SEE_RESET();
  }

  SEE_FREE(test_names_str_buffer);
  return num_failed_tests;
}

#define SEE_RUN_TESTS(...)                                                         \
  int main(void) {                                                             \
    void (*tests[])(void) = {__VA_ARGS__};                                     \
    const unsigned long n = sizeof(tests) / sizeof(tests[0]);                  \
    int failed_test = _SEE_run_tests(tests, n, _SEE_STR_ARGS(__VA_ARGS__));    \
    int memory_issues = _SEE_check_memory();                                   \
    return failed_test + memory_issues;                                        \
  }

#endif // SEE_H
