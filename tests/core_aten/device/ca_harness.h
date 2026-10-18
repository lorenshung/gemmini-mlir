// ca_harness.h - bare-metal checks for compiler-emitted Core ATen Gemmini kernels.
// The linked kernels perform integer device computation and exact CPU requantization.
// The driver fills sentinels and compares every output bit.  Protocol: PASS <check> / FAIL <check> <idx> <got> <exp> /
// INFO <check> mismatches=<n>/<total> / PERF <label> cycles=<n> / HARNESS_DONE pass=<n> fail=<m>.
#ifndef CA_HARNESS_H
#define CA_HARNESS_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

static int ca_npass = 0, ca_nfail = 0;

static inline void ca_fence(void) { asm volatile("fence" ::: "memory"); }
static inline uint64_t ca_rdcycle(void) { uint64_t x; asm volatile("rdcycle %0" : "=r"(x)); return x; }

static void ca_fill(void *p, size_t bytes, uint8_t v) {
  volatile uint8_t *q = (volatile uint8_t *)p;
  for (size_t i = 0; i < bytes; i++) q[i] = v;
}

// Correctness checks require exact equality.
static int ca_check_i8(const char *name, const int8_t *got, const int8_t *exp, size_t n, int tol) {
  ca_fence();
  size_t bad = 0, exact = 0, first = 0; int fg = 0, fe = 0;
  for (size_t i = 0; i < n; i++) {
    int d = (int)got[i] - (int)exp[i];
    if (d < 0) d = -d;
    if (d == 0) exact++;
    if (d != 0) { if (!bad) { first = i; fg = got[i]; fe = exp[i]; } bad++; }
  }
  if (tol != 0) { printf("FAIL nonzero_tolerance %s\n", name); ca_nfail++; return 0; }
  if (!bad) { printf("PASS %s\n", name); ca_npass++; return 1; }
  printf("FAIL %s %lu %d %d\n", name, (unsigned long)first, fg, fe);
  printf("INFO %s mismatches=%lu/%lu\n", name, (unsigned long)bad, (unsigned long)n);
  ca_nfail++;
  return 0;
}

static int ca_check_i32(const char *name, const int32_t *got, const int32_t *exp, size_t n, int tol) {
  if (tol != 0) { printf("FAIL nonzero_tolerance %s\n", name); ca_nfail++; return 0; }
  ca_fence();
  size_t bad = 0, first = 0; long fg = 0, fe = 0;
  for (size_t i = 0; i < n; i++) {
    long d = (long)got[i] - (long)exp[i];
    if (d < 0) d = -d;
    if (d != 0) { if (!bad) { first = i; fg = got[i]; fe = exp[i]; } bad++; }
  }
  if (!bad) { printf("PASS %s\n", name); ca_npass++; return 1; }
  printf("FAIL %s %lu %ld %ld\n", name, (unsigned long)first, fg, fe);
  printf("INFO %s mismatches=%lu/%lu\n", name, (unsigned long)bad, (unsigned long)n);
  ca_nfail++;
  return 0;
}

static int ca_finish(void) {
  ca_fence();
  printf("HARNESS_DONE pass=%d fail=%d\n", ca_npass, ca_nfail);
  return ca_nfail ? 1 : 0;
}
#endif
