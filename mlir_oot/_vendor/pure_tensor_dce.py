# Vendored verbatim functions from Merlin src/merlin/llvmlower/entry_weight_projection.py
# commit 5eaa5aed838d86e4a1a17d56fa957c807e1cb08b. No entry ABI/projection machinery is included.
from collections import Counter
from xdsl.dialects.builtin import TensorType
from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
from xdsl.dialects.linalg.ops import YieldOp
from xdsl.traits import IsTerminator, SymbolOpInterface, get_effects

def _pure_result_operation(operation) -> bool:
    if (
        not operation.results
        or operation.has_trait(IsTerminator, value_if_unregistered=False)
        or operation.has_trait(SymbolOpInterface, value_if_unregistered=False)
    ):
        return False
    # These value casts have no effects, although some supported xDSL versions
    # omit their Pure trait. Do not retain dead integer component subgraphs just
    # because that trait is absent.
    value_casts = {"arith.extsi", "arith.extui", "arith.trunci", "arith.bitcast"}
    if operation.name in value_casts:
        return True
    # Tensor structured operations can report unknown generic memory effects,
    # while their tensor results and fully pure scalar body are value-only.
    if isinstance(operation, LinalgStructuredOperation):
        if any(not isinstance(value.type, TensorType) for value in operation.results):
            return False
        from xdsl.dialects.builtin import MemRefType

        if any(isinstance(value.type, MemRefType) for value in operation.operands):
            return False
        for region in operation.regions:
            for child in region.walk():
                if isinstance(child, YieldOp) or child.name in value_casts:
                    continue
                if get_effects(child) != set():
                    return False
        return True
    # Tensor empties/views are value-only IR operations. Physical memref
    # allocations, reads/writes/frees and unknown effects remain even when the
    # result is unused: allocator failures/resource/address effects are not
    # granted by the arithmetic contract.
    return get_effects(operation) == set()

def _pure_dce(entry):
    erased = Counter()
    while True:
        changed = False
        for operation in reversed(tuple(entry.body.block.ops)):
            if all(not result.uses for result in operation.results) and _pure_result_operation(operation):
                erased[operation.name] += 1
                operation.parent.erase_op(operation)
                changed = True
        if not changed:
            return tuple(sorted(erased.items()))
