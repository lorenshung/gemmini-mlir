"""Freestanding Gemmini instruction emitter support for the selected DIM=16 target.

Only host-side data movement and loop/address calculation appear in this runtime.
The dot products, accumulation, output scaling, saturation and activation use RoCC.
The packed register offsets are those in rtl_facts/facts.json.
"""

RUNTIME = r'''
#include <stdint.h>
#include <stddef.h>

#define GM_ACC       0x80000000u
#define GM_ACC_ADD   0xc0000000u
#define GM_ACC_FULL  0xa0000000u
#define GM_GARBAGE   0xffffffffu
#define gm_cmd(f,a,b) do { \
  uintptr_t gm_arg_a=(uintptr_t)(a), gm_arg_b=(uintptr_t)(b); \
  __asm__ volatile (".insn r 0x7b, 3, " #f ", x0, %0, %1" \
                    : : "r"(gm_arg_a), "r"(gm_arg_b) : "memory"); \
} while (0)

static inline void gm_fence(void) {
  __asm__ volatile ("fence" ::: "memory");
}
static inline uint32_t gm_float_bits(float value) {
  union { float f; uint32_t u; } bits;
  bits.f=value;
  return bits.u;
}
static inline uint64_t gm_shape(uint32_t addr, int rows, int cols) {
  return (uint64_t)addr | ((uint64_t)(uint16_t)cols<<32)
                         | ((uint64_t)(uint16_t)rows<<48);
}
static inline void gm_ld_config(int stride_bytes, int id, float scale, int shrink) {
  /* CONFIG_LD, block stride = DIM; three independently configured load states. */
  uint64_t rs1=((uint64_t)gm_float_bits(scale)<<32) | (16ull<<16)
               | ((uint64_t)(id&3)<<3) | ((uint64_t)(shrink&1)<<2) | 1;
  gm_cmd(0,rs1,(uint64_t)(uint32_t)stride_bytes);
}
static inline void gm_st_config(int stride_bytes, float scale, int relu) {
  uint64_t rs1=2ull | ((uint64_t)(relu!=0)<<2);
  uint64_t rs2=((uint64_t)gm_float_bits(scale)<<32) | (uint32_t)stride_bytes;
  gm_cmd(0,rs1,rs2);
}
static inline void gm_ex_config(int dataflow, int relu) {
  uint64_t rs1=((uint64_t)gm_float_bits(1.0f)<<32) | (1ull<<16)
               | ((uint64_t)(relu!=0)<<3) | ((uint64_t)(dataflow&1)<<2);
  gm_cmd(0,rs1,1ull<<48);
}
static inline void gm_mvin(const void *ptr, uint32_t addr, int rows, int cols) {
  gm_cmd(2,ptr,gm_shape(addr,rows,cols));
}
static inline void gm_mvout(void *ptr, uint32_t addr, int rows, int cols) {
  gm_cmd(3,ptr,gm_shape(addr,rows,cols));
}

/* The buffers are static to keep each generated kernel's stack frame bounded. */
static int8_t gm_tile_a[16*16] __attribute__((aligned(64)));
static int8_t gm_tile_b[16*16] __attribute__((aligned(64)));
static int32_t gm_tile_seed[16*16] __attribute__((aligned(64)));
static int32_t gm_tile_c32[16*16] __attribute__((aligned(64)));
static int8_t gm_tile_c8[16*16] __attribute__((aligned(64)));

static void gm_matmul(const int8_t *A, const int8_t *B, void *C,
                      int M, int K, int N, int out_bytes,
                      float scale, int relu, const int32_t *bias) {
  gm_fence();
  /* WS consumes row-major A and B with the non-transpose configuration. */
  gm_ex_config(1,0);
  for (int mi=0; mi<M; mi+=16) {
    int mr=M-mi<16 ? M-mi : 16;
    for (int nj=0; nj<N; nj+=16) {
      int nc=N-nj<16 ? N-nj : 16;
      /* Bias is a vector along output columns, expanded by address/pack loops. */
      for (int r=0;r<16;r++)
        for (int c=0;c<16;c++)
          gm_tile_seed[r*16+c]=(bias && c<nc) ? bias[nj+c] : 0;
      gm_ld_config(16*4,0,1.0f,0);
      gm_mvin(gm_tile_seed,GM_ACC,16,16);
      gm_fence();
      gm_ld_config(16,0,1.0f,0);
      for (int kk=0;kk<K;kk+=16) {
        int kr=K-kk<16 ? K-kk : 16;
        for (int r=0;r<16;r++)
          for (int c=0;c<16;c++) {
            gm_tile_a[r*16+c]=(r<mr && c<kr) ? A[(mi+r)*K+kk+c] : 0;
            gm_tile_b[r*16+c]=(r<kr && c<nc) ? B[(kk+r)*N+nj+c] : 0;
          }
        gm_mvin(gm_tile_a,0,16,16);
        gm_mvin(gm_tile_b,16,16,16);
        /* Preload weights; write product into the seeded accumulator with add. */
        gm_cmd(6,gm_shape(16,16,16),gm_shape(GM_ACC_ADD,16,16));
        gm_cmd(4,gm_shape(0,16,16),gm_shape(GM_GARBAGE,16,16));
        /* CPU packs the same DMA buffers next iteration only after completion. */
        gm_fence();
      }
      if (out_bytes==4) {
        gm_st_config(16*4,1.0f,0);
        gm_mvout(gm_tile_c32,GM_ACC_FULL,mr,nc);
        gm_fence();
        for (int r=0;r<mr;r++)
          for (int c=0;c<nc;c++)
            ((int32_t*)C)[(mi+r)*N+nj+c]=gm_tile_c32[r*16+c];
      } else {
        gm_st_config(16,scale,relu);
        gm_mvout(gm_tile_c8,GM_ACC,mr,nc);
        gm_fence();
        for (int r=0;r<mr;r++)
          for (int c=0;c<nc;c++)
            ((int8_t*)C)[(mi+r)*N+nj+c]=gm_tile_c8[r*16+c];
      }
    }
  }
  gm_fence();
}
'''
