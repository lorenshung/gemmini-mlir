"""Preserve original source ordinals while delegating pure tensor DCE to Merlin."""


def pure_tensor_dce(function):
    from merlin.llvmlower.entry_weight_projection import _pure_dce
    # These frontend value operations currently parse as unregistered xDSL
    # ops, so the shared effect query cannot see their pure Q/DQ semantics.
    qdq = {'quant_ext.quantize_per_tensor', 'quant_ext.dequantize_per_tensor',
           'quant_ext.quantize_per_channel', 'quant_ext.dequantize_per_channel'}
    while True:
        _pure_dce(function)
        dead = [op for op in function.body.block.ops
                if getattr(getattr(op, 'op_name', None), 'data', op.name) in qdq
                and not op.regions and op.results and all(not r.uses for r in op.results)]
        if not dead:
            return
        for op in dead:
            op.parent_block().erase_op(op)


def live_source_ops(module):
    clone = module.clone()
    original = list(module.walk())
    copied = list(clone.walk())
    for function in list(clone.walk()):
        if function.name == 'func.func' and len(function.body.blocks) == 1:
            pure_tensor_dce(function)
    alive = set(clone.walk())
    return {op for op, copy in zip(original, copied, strict=True) if copy in alive}


def live_bindings(matches, live):
    """Drop target calls made dead by the target's own source replacements.

    A zero initializer, for example, is a source operand of a from-zero GEMM,
    but is not an argument of its target kernel. This is binding ABI liveness,
    not a second model DCE implementation.
    """
    replaced = {child for m in matches for op in m.consumed if op is not None for child in op.walk()}
    owners = {m.result[1]: i for i, m in enumerate(matches)}
    pending = [i for i, m in enumerate(matches) if any(
        use.operation in live and use.operation not in replaced for use in m.result[1].uses)]
    kept = set()
    while pending:
        i = pending.pop()
        if i in kept:
            continue
        kept.add(i)
        pending.extend(owners[value] for _, value in matches[i].operands if value in owners)
    return [m for i, m in enumerate(matches) if i in kept]
