"""Generate direct logical-tensor DMA schedules with no host tensor arithmetic."""

from math import isfinite
import re


def _resident_geometry(K, N, base_row):
    K, N, base_row = int(K), int(N), int(base_row)
    if min(K, N) < 0:
        raise ValueError('resident dimensions must be nonnegative')
    if base_row < 32 or base_row % 16 or base_row >= 16384:
        raise ValueError('resident cache base must be a 16-row aligned scratchpad row in [32,16384)')
    ntiles = (N + 15)//16
    rows = 16 * ((K + 15)//16) * ntiles
    if base_row + rows > 16384:
        raise ValueError('resident RHS exceeds the selected 16384-row scratchpad capacity')
    return K, N, base_row, ntiles


def _resident_row(kk, nj, base_row, ntiles):
    if str(kk).isdecimal() and str(nj).isdecimal():
        return str(base_row + 16*((int(kk)//16)*ntiles + int(nj)//16))
    return f'({base_row}+16*((({kk})/16)*{ntiles}+({nj})/16))'


def emit_resident_pack(b, K, N, base_row=32, tag='resident', unroll_limit=512):
    """Return declarations/statements loading an i8 RHS into scratchpad.

    Each logical 16-by-16 tile owns 16 scratchpad rows, including at a
    boundary.  Partial DMA dimensions identify its valid rows and columns;
    the matching preload dimensions keep the padded lanes out of the math.
    No CPU reads, copies, or tensor arithmetic are generated.
    """
    K, N, base_row, ntiles = _resident_geometry(K, N, base_row)
    if not K or not N:
        return [], []
    ident = 'gm_resident_' + re.sub(r'[^A-Za-z0-9_]', '_', str(tag))
    statements = ['gm_fence();', f'gm_ld_config({N},0,1.0f,0);']
    unrolled = ((K + 15)//16)*ntiles <= int(unroll_limit)

    def axis(extent, variable, consume):
        if unrolled:
            for start in range(0, extent, 16):
                consume(str(start), min(16, extent-start))
        else:
            full = extent//16*16
            if full:
                statements.append(f'for (int {variable}=0; {variable}<{full}; {variable}+=16) {{')
                consume(variable, 16)
                statements.append('}')
            if full != extent:
                consume(str(full), extent-full)

    def tile(kk, kr, nj, nc):
        bp = f'((const int8_t*)({b}))+((size_t)({kk})*{N}+({nj}))'
        row = _resident_row(kk, nj, base_row, ntiles)
        statements.append(f'gm_mvin({bp},{row},{kr},{nc});')

    axis(K, ident+'_k', lambda kk, kr:
         axis(N, ident+'_n', lambda nj, nc: tile(kk, kr, nj, nc)))
    statements.append('gm_fence();')
    return [], statements


def emit_direct_matmul(a, b, c, M, K, N, out_bytes=4, scale=1.0,
                       relu=False, bias=None, tag='direct', unroll_limit=512,
                       resident_base=None, resident_panel_base=None):
    """Return ``(C declarations, C statements)`` for a static i8 contraction.

    ``a``, ``b``, ``c`` and optional ``bias`` are C pointer expressions.  Dense
    logical inputs need no host packing: partial load/compute dimensions tell
    Gemmini which lanes are live.  Small schedules are expanded in Python;
    larger schedules have constant-trip-count full-tile loops and separate
    static tails, without shape tests in the emitted C.  ``resident_base``
    reuses the RHS tile layout loaded by :func:`emit_resident_pack` and emits
    no RHS DMA commands.  Its storage must remain live until this schedule
    completes; rows 0 through 31 remain reserved for ordinary temporaries.
    """
    M, K, N = int(M), int(K), int(N)
    if min(M, K, N) < 0:
        raise ValueError('contraction dimensions must be nonnegative')
    if resident_base is not None:
        _, _, resident_base, resident_ntiles = _resident_geometry(K, N, resident_base)
    if resident_panel_base is not None:
        if resident_base is not None:
            raise ValueError('whole RHS and panel RHS residency are mutually exclusive')
        _, _, resident_panel_base, _ = _resident_geometry(K, min(16,N), resident_panel_base)
    if out_bytes not in (1, 4):
        raise ValueError('Gemmini direct readout requires i8 or i32')
    scale = float(scale)
    if not isfinite(scale):
        raise ValueError('Gemmini direct readout scale must be finite')
    if M == 0 or N == 0:
        return [], []
    ident = 'gm_direct_' + re.sub(r'[^A-Za-z0-9_]', '_', str(tag))
    declarations = []
    if bias is None:
        # A zero row can be broadcast by DMA's zero source-row stride.
        declarations.append(
            f'static const int32_t {ident}_zero[16] '
            '__attribute__((aligned(64))) = {0};')
    statements = ['gm_fence();', 'gm_ex_config(1,0);']
    # Store state is independent of load/execute state and invariant throughout
    # this contraction. Configure once, before the first output tile; a later
    # contraction establishes its own complete state.
    statements.append(f'gm_st_config({N*4},1.0f,0);' if out_bytes == 4 else
                      f'gm_st_config({N},{scale.hex()}f,{int(bool(relu))});')
    # The selected LoadController has three independently configured states.
    # Keep state 0 for seed/panel loading, state 1 for A, and state 2 for B.
    # LOAD2/LOAD3 select those states without changing their row strides.
    statements.append(f'gm_ld_config({K},1,1.0f,0);')
    if resident_base is None and resident_panel_base is None:
        statements.append(f'gm_ld_config({N},2,1.0f,0);')
    tile_count = ((M + 15)//16) * ((N + 15)//16) * max(1, (K + 15)//16)
    unrolled = tile_count <= int(unroll_limit)

    def axis(extent, variable, consume):
        if unrolled:
            for start in range(0, extent, 16):
                consume(str(start), min(16, extent-start))
        else:
            full = extent//16*16
            if full:
                statements.append(f'for (int {variable}=0; {variable}<{full}; {variable}+=16) {{')
                consume(variable, 16)
                statements.append('}')
            if full != extent:
                consume(str(full), extent-full)

    # Only immutable DRAM inputs/seed rows and accelerator-owned outputs are
    # accessed inside this direct schedule. The selected reservation station
    # enforces local-address RAW/WAR/WAW hazards and per-class command order.
    # Keep entry/exit fences for caller memory visibility; do not drain between
    # tiles that the hardware can dependency-schedule. Host-packing paths retain
    # their original fences.
    def output_tile(mi, mr, nj, nc):
        seed = (f'((const int32_t*)({bias}))+({nj})'
                if bias is not None else ident+'_zero')
        statements.extend([
            'gm_ld_config(0,0,1.0f,0);',
            f'gm_mvin({seed},GM_ACC,{mr},{nc});',
        ])

        def reduction_tile(kk, kr):
            ap = f'((const int8_t*)({a}))+((size_t)({mi})*{K}+({kk}))'
            bp = f'((const int8_t*)({b}))+((size_t)({kk})*{N}+({nj}))'
            if resident_base is not None or resident_panel_base is not None:
                row = (_resident_row(kk, 0, resident_panel_base, 1)
                       if resident_panel_base is not None else
                       _resident_row(kk, nj, resident_base, resident_ntiles))
                # With RHS above row 31, both ordinary DIM-row slots are free
                # for A. Alternate them so the next DMA need not overwrite the
                # slot consumed by the preceding compute. Address dependencies
                # still guard wraparound; no overlap is assumed in the bound.
                a_row = str(16*((int(kk)//16)%2)) if str(kk).isdecimal() else f'(16*((({kk})/16)%2))'
                statements.extend([
                    f'gm_cmd(1,{ap},gm_shape({a_row},{mr},{kr}));',
                    f'gm_cmd(6,gm_shape({row},{kr},{nc}),gm_shape(GM_ACC_ADD,{mr},{nc}));',
                    f'gm_cmd(4,gm_shape({a_row},{mr},{kr}),gm_shape(GM_GARBAGE,16,16));',
                ])
                return
            statements.extend([
                f'gm_cmd(1,{ap},gm_shape(0,{mr},{kr}));',
            ])
            statements.extend([
                f'gm_cmd(14,{bp},gm_shape(16,{kr},{nc}));',
                f'gm_cmd(6,gm_shape(16,{kr},{nc}),gm_shape(GM_ACC_ADD,{mr},{nc}));',
                f'gm_cmd(4,gm_shape(0,{mr},{kr}),gm_shape(GM_GARBAGE,16,16));',
            ])

        axis(K, ident+'_k', reduction_tile)
        if out_bytes == 4:
            cp = f'((int32_t*)({c}))+((size_t)({mi})*{N}+({nj}))'
            statements.extend([
                f'gm_mvout({cp},GM_ACC_FULL,{mr},{nc});',
            ])
        else:
            cp = f'((int8_t*)({c}))+((size_t)({mi})*{N}+({nj}))'
            statements.extend([
                f'gm_mvout({cp},GM_ACC,{mr},{nc});',
            ])

    if resident_panel_base is None:
        axis(M, ident+'_m', lambda mi, mr:
             axis(N, ident+'_n', lambda nj, nc: output_tile(mi, mr, nj, nc)))
    else:
        def panel(nj, nc):
            # One K-by-DIM weight panel fits even when the full K-by-N RHS
            # does not. Keep it live through all M tiles, then reuse its rows.
            # Caller proves the interval lies above every live explicit RHS.
            statements.extend(['gm_fence();', f'gm_ld_config({N},0,1.0f,0);'])
            def load_panel_tile(kk, kr):
                bp = f'((const int8_t*)({b}))+((size_t)({kk})*{N}+({nj}))'
                row = _resident_row(kk, 0, resident_panel_base, 1)
                statements.append(f'gm_mvin({bp},{row},{kr},{nc});')
            axis(K, ident+'_panel_k', load_panel_tile)
            statements.append('gm_fence();')
            axis(M, ident+'_m', lambda mi, mr: output_tile(mi,mr,nj,nc))
        axis(N, ident+'_n', panel)
    statements.append('gm_fence();')
    return declarations, statements
