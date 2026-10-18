"""Preserve original source ordinals while delegating pure tensor DCE to Merlin."""


def live_source_ops(module):
    from merlin.llvmlower.entry_weight_projection import _pure_dce
    clone = module.clone()
    original = list(module.walk())
    copied = list(clone.walk())
    for function in list(clone.walk()):
        if function.name == 'func.func' and len(function.body.blocks) == 1:
            _pure_dce(function)
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
