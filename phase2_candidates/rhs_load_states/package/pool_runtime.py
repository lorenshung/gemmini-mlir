"""Native Gemmini STORE max-pooling with DMA-only window gathering.

All tensor comparisons and reductions are performed by the store datapath.  A
2x2 reduction tree also handles rectangular and larger logical pooling windows,
without relying on a single full image fitting in the accumulator SRAM.
"""

# Precomputed byte constants let DMA materialize arbitrary signed padding values
# without running scalar stores over a tensor during the kernel invocation.
_PAD_ROWS = ',\n'.join(
    '{' + ','.join([str(i if i < 128 else i - 256)] * 16) + '}'
    for i in range(256)
)

POOL_RUNTIME = (
    'static const int8_t gm_pool_pad[256][16] '
    '__attribute__((aligned(64))) = {\n' + _PAD_ROWS + '\n};\n'
) + r'''

static void gm_pool_reduce_four(int8_t *dst, int channels) {
  /* CONFIG_MVOUT, using the selected RTL's ConfigMvoutRs1 field offsets:
     pool_stride=2, pool_size=2, upad=lpad=0, pool_out_dim=1,
     porows=pocols=1, orows=ocols=2.  One pooled spatial pixel is stored. */
  uint64_t cfg=2ull | (2ull<<4) | (2ull<<6)
       | (1ull<<24) | (1ull<<32) | (1ull<<40)
       | (2ull<<48) | (2ull<<56);
  uint64_t stride_scale=((uint64_t)gm_float_bits(1.0f)<<32) | 16ull;
  gm_cmd(0,cfg,stride_scale);
  /* With pooling enabled, spatial geometry is taken from CONFIG_MVOUT;
     the native pooling mvout convention uses a zero ordinary row count. */
  gm_mvout(dst,GM_ACC,0,channels);
  gm_fence();
}

static void gm_pool(const int8_t *src, int8_t *dst,
                    int batch, int H, int W, int N,
                    int ph, int pw, int sh, int sw,
                    int pt, int pl, int pb, int pr, int pad_value) {
  gm_fence();
  int Ho=(H+pt+pb-ph)/sh+1;
  int Wo=(W+pl+pr-pw)/sw+1;
  int window=ph*pw;
  int initial_groups=(window+3)/4;
  /* Only Gemmini DMA accesses this reduction-tree storage. */
  int8_t partial[initial_groups*16] __attribute__((aligned(64)));
  gm_ld_config(16,0,1.0f,1);
  for (int b=0;b<batch;b++)
    for (int oh=0;oh<Ho;oh++)
      for (int ow=0;ow<Wo;ow++)
        for (int c=0;c<N;c+=16) {
          int nc=N-c<16 ? N-c : 16;
          for (int g=0;g<initial_groups;g++) {
            for (int p=0;p<4;p++) {
              int wi=g*4+p;
              /* Duplicate an existing leaf for a short final group.  This
                 preserves a max without inventing a padding identity. */
              if (wi>=window) wi=window-1;
              int ih=oh*sh-pt+wi/pw;
              int iw=ow*sw-pl+wi%pw;
              const int8_t *leaf=(ih>=0 && ih<H && iw>=0 && iw<W)
                  ? src+((b*H+ih)*W+iw)*N+c
                  : gm_pool_pad[(uint8_t)pad_value];
              gm_mvin(leaf,GM_ACC+(uint32_t)p,1,nc);
            }
            int8_t *result=initial_groups==1
                ? dst+((b*Ho+oh)*Wo+ow)*N+c : partial+g*16;
            gm_pool_reduce_four(result,nc);
          }
          int count=initial_groups;
          while (count>1) {
            int groups=(count+3)/4;
            for (int g=0;g<groups;g++) {
              for (int p=0;p<4;p++) {
                int index=g*4+p;
                if (index>=count) index=count-1;
                gm_mvin(partial+index*16,GM_ACC+(uint32_t)p,1,nc);
              }
              int8_t *result=groups==1
                  ? dst+((b*Ho+oh)*Wo+ow)*N+c : partial+g*16;
              gm_pool_reduce_four(result,nc);
            }
            count=groups;
          }
        }
  gm_st_config(N,1.0f,0);
  gm_fence();
}
'''
