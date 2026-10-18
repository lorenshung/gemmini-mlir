"""NHWC convolution lowering through an accelerator matrix contraction.

Small convolutions gather windows directly through Gemmini DMA. Larger
convolutions retain an address-only im2col transform. Arithmetic and readout
are performed by the Gemmini runtime.
"""
import math
import re
from epilogue_validation import validate_acc_scale
from bias_validation import validate_bias_epilogue


def _attrs(command):
    return command.get('attrs', command.get('attributes', {}))


def _ints(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f'{name} must have {count} entries')
    return tuple(int(v) for v in value)


def _float_literal(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('convolution scale must be finite')
    literal = repr(value)
    if '.' not in literal and 'e' not in literal.lower():
        literal += '.0'
    return literal + 'f'


def emit_conv(cb, command, ptrs, uid):
    """Return global declarations and C statements for one CONV2D command.

    ``ptrs`` maps declared tensor names to C pointer expressions. Resident
    weight handles are resolved by the caller before entering this function.
    """
    attrs = _attrs(command)
    validate_acc_scale(attrs)
    operands = command.get('operands', {})
    tensors = cb['tensors']
    ifm, weight, dst = (operands[k] for k in ('ifm', 'weight', 'dst'))
    shape = tuple(int(x) for x in tensors[ifm]['shape'])
    if len(shape) != 4:
        raise ValueError('CONV2D requires a rank-four NHWC activation')
    batch, height, width, channels = shape
    kh, kw, ci, co = _ints(attrs['kernel'], 4, 'kernel')
    sh, sw = _ints(attrs.get('stride', [1, 1]), 2, 'stride')
    dh, dw = _ints(attrs.get('dilation', [1, 1]), 2, 'dilation')
    pt, pl, pb, pr = _ints(attrs.get('padding', [0, 0, 0, 0]), 4, 'padding')
    if attrs.get('layout', 'nhwc').lower() != 'nhwc':
        raise ValueError('CONV2D only supports nhwc layout')
    if ci != channels or min(batch, height, width, ci, co, kh, kw, sh, sw, dh, dw) <= 0:
        raise ValueError('invalid CONV2D kernel or activation geometry')
    if min(pt, pl, pb, pr) < 0:
        raise ValueError('CONV2D padding must be nonnegative')
    oh = (height + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    ow = (width + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    if min(oh, ow) <= 0:
        raise ValueError('CONV2D has empty output geometry')
    rows, reduction = batch * oh * ow, kh * kw * ci
    wshape = tuple(int(x) for x in tensors[weight]['shape'])
    if math.prod(wshape) != reduction * co:
        raise ValueError('CONV2D weight does not match kernel geometry')
    if tensors[ifm]['dtype'] != 'i8' or tensors[weight]['dtype'] != 'i8':
        raise NotImplementedError('CONV2D requires signed int8 operands')
    epilogue = list(attrs.get('epilogue', []))
    allowed = {'bias_add', 'bias', 'requant', 'acc_scale', 'relu', 'maxpool'}
    if any(stage not in allowed for stage in epilogue):
        raise ValueError('unsupported CONV2D epilogue')
    output_dtype = attrs.get('output_dtype', tensors[dst]['dtype'])
    if output_dtype not in ('i8', 'i32'):
        raise NotImplementedError('CONV2D readout supports i8 and i32')
    output_bytes = 1 if output_dtype == 'i8' else 4
    scale = 1.0
    if 'acc_scale' in epilogue:
        scale = float(attrs['acc_scale'])
    if 'requant' in epilogue:
        scale *= math.ldexp(1.0, -int(attrs['requant_shift']))
    bias = '0'
    if 'bias_add' in epilogue or 'bias' in epilogue:
        bias_name = validate_bias_epilogue(attrs, operands, tensors, co)
        if bias_name not in ptrs:
            raise ValueError('CONV2D bias stage needs a declared bias operand')
        bias = f'(const int32_t *)({ptrs[bias_name]})'
    if 'maxpool' in epilogue:
        # The existing native pool engine consumes signed-byte convolution
        # readout. Keep every preceding conv stage on the device, then let
        # STORE pooling reduce that materialized native readout without any
        # scalar tensor arithmetic or comparisons.
        from pool_codegen import validate_pool
        if epilogue.count('maxpool') != 1 or epilogue[-1] != 'maxpool':
            raise ValueError('CONV2D requires maxpool exactly once as its final epilogue stage')
        if tensors[dst]['dtype'] != output_dtype:
            raise ValueError('CONV2D output_dtype disagrees with destination dtype')
        geometry = validate_pool(
            attrs, rows, co, tensors[dst]['shape'], output_dtype,
            expected_in_dims=(oh, ow))
        pool_tag = 'conv_' + re.sub(r'[^A-Za-z0-9_]', '_', str(uid)) + '_prepool'
        intermediate = pool_tag
        while intermediate in tensors:
            intermediate += '_'
        inner_tensors = dict(tensors)
        inner_tensors[intermediate] = {
            'shape': [rows, co], 'dtype': output_dtype, 'role': 'intermediate',
        }
        inner_cb = {**cb, 'tensors': inner_tensors}
        inner_attrs = {**attrs, 'epilogue': epilogue[:-1]}
        inner_command = {
            **command,
            'operands': {**operands, 'dst': intermediate},
            'attributes': inner_attrs,
        }
        inner_command.pop('attrs', None)
        inner_ptrs = {**ptrs, intermediate: pool_tag}
        declarations, statements = emit_conv(inner_cb, inner_command, inner_ptrs, uid)
        declarations.insert(0,
            f'static int8_t {pool_tag}[{rows * co}] __attribute__((aligned(64)));')
        statements.append(
            f'gm_pool({pool_tag}, (int8_t *)({ptrs[dst]}), '
            + ', '.join(str(v) for v in (*geometry[:3], co, *geometry[3:])) + ');')
        return declarations, statements
    if math.prod(tensors[dst]['shape']) != rows * co:
        raise ValueError('CONV2D destination shape disagrees with geometry')
    tag = 'conv_' + re.sub(r'[^A-Za-z0-9_]', '_', str(uid))
    col = tag + '_col'
    fast = rows >= 1024
    # Gather each convolution window directly into scratchpad rows. Geometry
    # and padding are resolved in Python; the emitted host code only supplies
    # addresses and instruction operands and never reads activation elements.
    if not fast and rows * reduction <= 32768:
        if not math.isfinite(scale):
            raise ValueError('convolution scale must be finite')
        zero8, zero32 = tag + '_zero8', tag + '_zero32'
        declarations = [
            f'static const int8_t {zero8}[16] __attribute__((aligned(64))) = {{0}};',
            f'static const int32_t {zero32}[16] __attribute__((aligned(64))) = {{0}};',
        ]
        schedule = ['gm_fence();', 'gm_ex_config(1,0);']
        tile_count = ((rows + 15) // 16) * ((co + 15) // 16) * kh * kw * ((ci + 15) // 16)
        if tile_count > 512:
            # Keep the static spatial gather, but share wide output/channel
            # schedules through constant-trip loops. Full tiles and tails
            # have separate statically known DMA and compute dimensions.
            def axis(extent, variable, consume):
                full = extent // 16 * 16
                if full:
                    schedule.extend([
                        '#pragma clang loop unroll(disable)',
                        f'for (int {variable} = 0; {variable} < {full}; {variable} += 16) {{',
                    ])
                    consume(variable, 16)
                    schedule.append('}')
                if full != extent:
                    consume(str(full), extent - full)

            def output_tile(mi, mr, nj, nc):
                seed = zero32 if bias == '0' else f'({bias}) + ({nj})'
                schedule.extend([
                    'gm_ld_config(0,0,1.0f,0);',
                    f'gm_mvin({seed},GM_ACC,{mr},{nc});',
                    'gm_fence();',
                ])

                def reduction_tile(ky, kx, channel, kr):
                    schedule.append(f'gm_ld_config({ci},0,1.0f,0);')
                    for ri in range(mr):
                        bn, spatial = divmod(mi + ri, oh * ow)
                        oy, ox = divmod(spatial, ow)
                        iy = oy * sh + ky * dh - pt
                        ix = ox * sw + kx * dw - pl
                        if 0 <= iy < height and 0 <= ix < width:
                            source = ((bn * height + iy) * width + ix) * ci
                            address = f'((const int8_t *)({ptrs[ifm]})) + {source} + ({channel})'
                        else:
                            address = zero8
                        schedule.append(f'gm_mvin({address},{ri},1,{kr});')
                    weight_base = (ky * kw + kx) * ci * co
                    weight_address = f'((const int8_t *)({ptrs[weight]})) + {weight_base} + ({channel}) * {co} + ({nj})'
                    schedule.extend([
                        'gm_fence();',
                        f'gm_ld_config({co},0,1.0f,0);',
                        f'gm_mvin({weight_address},16,{kr},{nc});',
                        f'gm_cmd(6,gm_shape(16,{kr},{nc}),gm_shape(GM_ACC_ADD,{mr},{nc}));',
                        f'gm_cmd(4,gm_shape(0,{mr},{kr}),gm_shape(GM_GARBAGE,16,16));',
                        'gm_fence();',
                    ])

                for ky in range(kh):
                    for kx in range(kw):
                        axis(ci, tag + '_channel', lambda channel, kr:
                             reduction_tile(ky, kx, channel, kr))
                output_offset = f'{mi * co} + ({nj})'
                if output_bytes == 4:
                    output_address = f'((int32_t *)({ptrs[dst]})) + {output_offset}'
                    schedule.extend([
                        f'gm_st_config({co * 4},1.0f,0);',
                        f'gm_mvout({output_address},GM_ACC_FULL,{mr},{nc});',
                    ])
                else:
                    output_address = f'((int8_t *)({ptrs[dst]})) + {output_offset}'
                    schedule.extend([
                        f'gm_st_config({co},{scale.hex()}f,{int("relu" in epilogue)});',
                        f'gm_mvout({output_address},GM_ACC,{mr},{nc});',
                    ])
                schedule.append('gm_fence();')

            for mi in range(0, rows, 16):
                mr = min(16, rows - mi)
                axis(co, tag + '_output_channel', lambda nj, nc:
                     output_tile(mi, mr, nj, nc))
            schedule.append('gm_fence();')
            return declarations, ['\n'.join(['{', *schedule, '}'])]
        for mi in range(0, rows, 16):
            mr = min(16, rows - mi)
            for nj in range(0, co, 16):
                nc = min(16, co - nj)
                seed = zero32 if bias == '0' else f'({bias}) + {nj}'
                schedule.extend([
                    'gm_ld_config(0,0,1.0f,0);',
                    f'gm_mvin({seed},GM_ACC,{mr},{nc});',
                    'gm_fence();',
                ])
                for ky in range(kh):
                    for kx in range(kw):
                        for channel in range(0, ci, 16):
                            kr = min(16, ci - channel)
                            schedule.append(f'gm_ld_config({ci},0,1.0f,0);')
                            for ri in range(mr):
                                bn, spatial = divmod(mi + ri, oh * ow)
                                oy, ox = divmod(spatial, ow)
                                iy = oy * sh + ky * dh - pt
                                ix = ox * sw + kx * dw - pl
                                if 0 <= iy < height and 0 <= ix < width:
                                    source = ((bn * height + iy) * width + ix) * ci + channel
                                    address = f'((const int8_t *)({ptrs[ifm]})) + {source}'
                                else:
                                    address = zero8
                                schedule.append(f'gm_mvin({address},{ri},1,{kr});')
                            weight_offset = ((ky * kw + kx) * ci + channel) * co + nj
                            weight_address = f'((const int8_t *)({ptrs[weight]})) + {weight_offset}'
                            schedule.extend([
                                'gm_fence();',
                                f'gm_ld_config({co},0,1.0f,0);',
                                f'gm_mvin({weight_address},16,{kr},{nc});',
                                f'gm_cmd(6,gm_shape(16,{kr},{nc}),gm_shape(GM_ACC_ADD,{mr},{nc}));',
                                f'gm_cmd(4,gm_shape(0,{mr},{kr}),gm_shape(GM_GARBAGE,16,16));',
                                'gm_fence();',
                            ])
                output_offset = mi * co + nj
                if output_bytes == 4:
                    output_address = f'((int32_t *)({ptrs[dst]})) + {output_offset}'
                    schedule.extend([
                        f'gm_st_config({co * 4},1.0f,0);',
                        f'gm_mvout({output_address},GM_ACC_FULL,{mr},{nc});',
                    ])
                else:
                    output_address = f'((int8_t *)({ptrs[dst]})) + {output_offset}'
                    schedule.extend([
                        f'gm_st_config({co},{scale.hex()}f,{int("relu" in epilogue)});',
                        f'gm_mvout({output_address},GM_ACC,{mr},{nc});',
                    ])
                schedule.append('gm_fence();')
        schedule.append('gm_fence();')
        return declarations, ['\n'.join(['{', *schedule, '}'])]
    declarations = [f'static int8_t {col}[{rows * reduction}] __attribute__((aligned(64)));']
    copy_width = 8 if fast and ci % 8 == 0 else (4 if fast and ci % 4 == 0 else 1)
    copy_type = {1:'int8_t',4:'uint32_t',8:'uint64_t'}[copy_width]
    kernel_function = 'gm_matmul_fast' if fast else 'gm_matmul'
    contraction = f'''{kernel_function}({col}, (const int8_t *)({ptrs[weight]}), (void *)({ptrs[dst]}),
            {rows}, {reduction}, {co}, {output_bytes}, {_float_literal(scale)},
            {int('relu' in epilogue)}, {bias});'''
    if fast and output_dtype == 'i32' and not epilogue:
        from direct_codegen import emit_direct_matmul
        extra, schedule = emit_direct_matmul(col, ptrs[weight], ptrs[dst], rows, reduction, co, tag=tag)
        declarations.extend(extra)
        contraction = '\n'.join(schedule)
    if copy_width > 1:
        destination = f'__builtin_assume_aligned(&{col}[p], {copy_width})'
        source = f'__builtin_assume_aligned(&{tag}_input[((bn * {height} + iy) * {width} + ix) * {ci} + ch], {copy_width})'
        movement = f'if (iy >= 0 && iy < {height} && ix >= 0 && ix < {width}) __builtin_memcpy({destination}, {source}, {copy_width}); else __builtin_memset({destination}, 0, {copy_width});'
    else:
        movement = f'{col}[p] = (iy >= 0 && iy < {height} && ix >= 0 && ix < {width}) ? {tag}_input[((bn * {height} + iy) * {width} + ix) * {ci} + ch] : 0;'
    statements = [f'''{{
  const int8_t *{tag}_input = (const int8_t *)({ptrs[ifm]});
  for (int bn = 0; bn < {batch}; ++bn)
    for (int oy = 0; oy < {oh}; ++oy)
      for (int ox = 0; ox < {ow}; ++ox)
        for (int ky = 0; ky < {kh}; ++ky)
          for (int kx = 0; kx < {kw}; ++kx) {{
            int iy = oy * {sh} + ky * {dh} - {pt};
            int ix = ox * {sw} + kx * {dw} - {pl};
            for (int ch = 0; ch < {ci}; ch += {copy_width}) {{
              int p = ((bn * {oh} + oy) * {ow} + ox) * {reduction}
                    + (ky * {kw} + kx) * {ci} + ch;
              {movement}
            }}
          }}
  {contraction}
}}''']
    return declarations, statements
