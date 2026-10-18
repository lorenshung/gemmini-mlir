"""Direct-DMA matrix lowering for large contractions.

Append FAST_RUNTIME after device_runtime.RUNTIME.  Complete input tiles are
already row-major logical slices, so their row strides can be described to DMA
without repacking them.  Boundary tiles retain the padded baseline schedule.
"""

FAST_RUNTIME = r'''
static void gm_matmul_fast(const int8_t *A, const int8_t *B, void *C,
                           int M, int K, int N, int out_bytes,
                           float scale, int relu, const int32_t *bias) {
  gm_fence();
  gm_ex_config(1,0);
  for (int mi=0; mi<M; mi+=16) {
    int mr=M-mi<16 ? M-mi : 16;
    for (int nj=0; nj<N; nj+=16) {
      int nc=N-nj<16 ? N-nj : 16;
      for (int r=0;r<16;r++)
        for (int c=0;c<16;c++)
          gm_tile_seed[r*16+c]=(bias && c<nc) ? bias[nj+c] : 0;
      gm_ld_config(16*4,0,1.0f,0);
      gm_mvin(gm_tile_seed,GM_ACC,16,16);
      gm_fence();
      for (int kk=0;kk<K;kk+=16) {
        int kr=K-kk<16 ? K-kk : 16;
        if (mr==16 && kr==16 && nc==16) {
          /* Each command still uses LOAD_CMD/state zero.  Drain that state
             before changing its row stride for the other operand. */
          gm_ld_config(K,0,1.0f,0);
          gm_mvin(A+(size_t)mi*K+kk,0,16,16);
          gm_fence();
          gm_ld_config(N,0,1.0f,0);
          gm_mvin(B+(size_t)kk*N+nj,16,16,16);
        } else {
          for (int r=0;r<16;r++)
            for (int c=0;c<16;c++) {
              gm_tile_a[r*16+c]=(r<mr && c<kr) ? A[(size_t)(mi+r)*K+kk+c] : 0;
              gm_tile_b[r*16+c]=(r<kr && c<nc) ? B[(size_t)(kk+r)*N+nj+c] : 0;
            }
          gm_ld_config(16,0,1.0f,0);
          gm_mvin(gm_tile_a,0,16,16);
          gm_mvin(gm_tile_b,16,16,16);
        }
        gm_cmd(6,gm_shape(16,16,16),gm_shape(GM_ACC_ADD,16,16));
        gm_cmd(4,gm_shape(0,16,16),gm_shape(GM_GARBAGE,16,16));
        gm_fence();
      }
      /* Store rows use the logical destination stride directly, including
         edge tiles; no host readback, conversion or output-copy loop. */
      if (out_bytes==4) {
        gm_st_config(N*4,1.0f,0);
        gm_mvout((int32_t*)C+(size_t)mi*N+nj,GM_ACC_FULL,mr,nc);
      } else {
        gm_st_config(N,scale,relu);
        gm_mvout((int8_t*)C+(size_t)mi*N+nj,GM_ACC,mr,nc);
      }
      gm_fence();
    }
  }
  gm_fence();
}
'''
