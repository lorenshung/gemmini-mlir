"""Fence-correct overlap of private full-i32 readout DMA and host requant.

Two compact DRAM slots alternate. A fence completes tile i, tile i+1 is issued,
then the CPU consumes tile i before any command may reuse its slot. Device
accumulator/scratchpad scheduling remains the recipe's responsibility.
"""
from dataclasses import replace
import numpy as np
from .stream import DIM, Ins, CompactInstructions


def prepare(stream, meta):
    if meta.get('readout_pipeline', {}).get('status') == 'enabled':
        return
    def decline(reason):
        meta['readout_pipeline'] = dict(status='serial', reason=reason)
    if isinstance(stream.ins, CompactInstructions):
        return decline('structural DMA loops retain the serial exact readout boundary')
    if stream.bitwise or len(stream.readouts) != 1:
        return decline('requires one independent quantized result')
    readout = stream.readouts[0]
    if readout.after is not None or readout.indices is not None or readout.placement is not None:
        return decline('existing host dependency boundaries retained')
    args = {a.name: a for a in stream.args}
    size = int(np.prod(args[readout.out].shape))
    writers = {}
    for i, ins in enumerate(stream.ins):
        if ins.kind == 'mvout':
            writers.setdefault(stream.args[ins.arg].name, []).append(i)
    candidates = [(positions[-1], name) for name, _ in readout.terms
                  if (positions := writers.get(name))]
    if not candidates:
        return decline('no asynchronous int32 producer')
    _, name = max(candidates)
    source = args[name]
    if (source.role != 'scratch' or source.dtype != 'i32' or
        int(np.prod(source.shape)) != size or
        name in (readout.permutations or {}) or name in (readout.broadcasts or {})):
        return decline('last producer is not a private identity-layout int32 tile')
    index = stream._index[name]
    if any(i.kind == 'mvin' and i.arg == index for i in stream.ins):
        return decline('device consumes this accumulator buffer')
    positions = writers[name]
    if any(pos >= positions[0] for other, _ in readout.terms if other != name
           for pos in writers.get(other, [])):
        return decline('other rational components are not ready before the first tile')
    if any(i.kind != 'fence' for i in stream.ins[positions[-1]+1:]):
        return decline('later device side effects require the serial boundary')
    # Group consecutive full-width stores issued by a recipe's output block.
    groups = []
    stride = None
    for i, ins in enumerate(stream.ins):
        if ins.kind == 'config_st':
            stride = ins.a('stride')
        if ins.kind != 'mvout' or ins.arg != index:
            continue
        if not ins.a('local') & (1 << 29) or stride is None or stride % 4 or ins.off % 4:
            return decline('store lacks a full-int32 stride proof')
        points = [ins.off//4+r*(stride//4)+c for r in range(ins.a('rows')) for c in range(ins.a('cols'))]
        if not points or min(points) < 0 or max(points) >= size:
            return decline('store does not map inside the logical result')
        if groups and groups[-1]['end'] == i:
            groups[-1]['stores'].append(i)
            groups[-1]['points'].extend(points)
            groups[-1]['end'] = i+1
        else:
            groups.append(dict(stores=[i], points=points, end=i+1))
    if len(groups) < 2:
        return decline('one tile has no successor to overlap')
    coverage = np.zeros(size, np.int64)
    for g in groups:
        if len(set(g['points'])) != len(g['points']):
            return decline('overlapping stores inside an output block')
        coverage[g['points']] += 1
    if not np.all(coverage == 1):
        return decline('output tiles do not partition the exact result')
    extent = max(max(g['points'])-min(g['points'])+1 for g in groups)
    capacity = ((extent+DIM-1)//DIM)*DIM
    # A compact buffer must actually reduce scratch, apart from small tails.
    # This is a resource rule, independent of op names and provenance IDs.
    if capacity*2 > size:
        return decline('two readout slots would exceed the original scratch extent')
    original = list(stream.ins)
    new, readouts, receipt = [], [], []
    start = 0
    pending = None
    for ordinal, g in enumerate(groups):
        base = min(g['points'])
        slot = ordinal % 2
        offsets = {i: (slot*capacity-base)*4 for i in g['stores']}
        for i in range(start, g['end']):
            ins = original[i]
            new.append(ins if i not in offsets else Ins(ins.kind, ins.attrs, ins.arg, ins.off+offsets[i]))
        # The previous slot was fenced before this block was issued. There is
        # deliberately no fence between this block and the preceding epilogue.
        if pending is not None:
            readouts.append(replace(pending, after=len(new)))
        new.append(Ins('fence', (), None, 0))
        from .requant import expanded
        points = np.asarray(sorted(g['points']), dtype=np.int64)
        def compact(value):
            a = expanded(value, args[readout.out].shape, readout.axis)[points]
            return a[0] if all(v == a[0] for v in a) else a
        pending = replace(readout, indices=tuple(map(int, points)), parameters_compact=True,
                          terms=[(n,compact(f)) for n,f in readout.terms],
                          constant=compact(readout.constant), zero_point=compact(readout.zero_point),
                          quant_min=compact(readout.quant_min), quant_max=compact(readout.quant_max),
                          source_offsets={**(readout.source_offsets or {}), name:slot*capacity-base})
        receipt.append(dict(tile=ordinal, slot=slot, logical_min=base,
                            logical_max=max(g['points']), elements=len(g['points']),
                            ready_after=len(new), host_after=None))
        if ordinal:
            receipt[ordinal-1]['host_after'] = len(new)-1
        start = g['end']
    readouts.append(replace(pending, after=len(new)))
    receipt[-1]['host_after'] = len(new)
    new.extend(original[start:])
    # All legality checks above precede mutations. No partially transformed
    # stream can escape on a declined plan.
    stream.args[index] = replace(source, shape=(capacity*2,))
    stream.ins[:] = new
    stream.readouts[:] = readouts
    meta['readout_pipeline'] = dict(status='enabled', buffers=2, source=name,
        slot_elements=capacity, slot_alignment_elements=DIM, original_elements=size, tiles=receipt,
        ordering='MVOUT(i); fence; issue(i+1); host(i); fence; reuse(i)',
        host_arithmetic='requant-A-v1', completion='final tile fenced before final epilogue')
