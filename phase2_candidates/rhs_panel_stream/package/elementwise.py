"""Gemmini movement and elementwise C emission helpers.

These routines deliberately use the accumulator's DMA accumulation path for
addition, including row broadcast for a bias vector.  C loops schedule tiles
and compute addresses; every tensor value is handled by the accelerator.
"""

ELEMENTWISE_RUNTIME = r'''
static void gm_move(const signed char *src, void *dst,
                    long rows, long cols, int out_bytes) {
  gm_fence();
  gm_ld_config((unsigned long)cols, 0, 1.0f, out_bytes == 4);
  gm_st_config((unsigned long)cols * (unsigned long)out_bytes, 1.0f, 0);
  for (long r = 0; r < rows; r += 16) {
    long nr = rows - r < 16 ? rows - r : 16;
    for (long c = 0; c < cols; c += 16) {
      long nc = cols - c < 16 ? cols - c : 16;
      unsigned int address = out_bytes == 4 ? GM_ACC : 0;
      gm_mvin(src + r * cols + c, address, nr, nc);
      gm_mvout((unsigned char *)dst + (r * cols + c) * out_bytes,
               out_bytes == 4 ? GM_ACC_FULL : 0, nr, nc);
    }
  }
  gm_fence();
}

static void gm_bias(const int *src, const int *bias, int *dst,
                    long rows, long cols) {
  gm_fence();
  gm_st_config((unsigned long)cols * 4, 1.0f, 0);
  for (long r = 0; r < rows; r += 16) {
    long nr = rows - r < 16 ? rows - r : 16;
    for (long c = 0; c < cols; c += 16) {
      long nc = cols - c < 16 ? cols - c : 16;
      gm_ld_config((unsigned long)cols * 4, 0, 1.0f, 0);
      gm_mvin(src + r * cols + c, GM_ACC, nr, nc);
      /* A zero external stride repeats the bias row across the tile. */
      gm_ld_config(0, 0, 1.0f, 0);
      gm_mvin(bias + c, GM_ACC_ADD, nr, nc);
      gm_mvout(dst + r * cols + c, GM_ACC_FULL, nr, nc);
    }
  }
  gm_fence();
}

static void gm_residual(const signed char *lhs, const signed char *rhs,
                        signed char *dst, long rows, long cols,
                        float lhs_scale, float rhs_scale, int relu) {
  gm_fence();
  gm_st_config((unsigned long)cols, 1.0f, relu);
  for (long r = 0; r < rows; r += 16) {
    long nr = rows - r < 16 ? rows - r : 16;
    for (long c = 0; c < cols; c += 16) {
      long nc = cols - c < 16 ? cols - c : 16;
      gm_ld_config((unsigned long)cols, 0, lhs_scale, 1);
      gm_mvin(lhs + r * cols + c, GM_ACC, nr, nc);
      /* Complete the overwrite before changing the load scale or adding. */
      gm_fence();
      gm_ld_config((unsigned long)cols, 0, rhs_scale, 1);
      gm_mvin(rhs + r * cols + c, GM_ACC_ADD, nr, nc);
      /* Read only after the accumulator's DMA add has completed. */
      gm_fence();
      gm_mvout(dst + r * cols + c, GM_ACC, nr, nc);
      /* The next tile reuses these same accumulator rows. */
      gm_fence();
    }
  }
  gm_fence();
}
'''
