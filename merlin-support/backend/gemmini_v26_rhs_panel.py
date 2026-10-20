"""Prospective V26 full-K stripe adapter; no execution/timing qualification.

The selected resource view below is an explicit compatibility snapshot, not a
capacity guess or mutable hardware grant. Original complete-source/actual-byte
owners must validate the final ELF, aliases and lifetimes before admission.
Shared dense planning and block scheduling own all schedule choices/checks.
"""

from dataclasses import replace

from merlin.compile.scheduling.block_schedule import (
    CONTIGUOUS,
    NEST,
    Compute,
    Contraction,
    Geometry,
    K,
    Knobs,
    Load,
    M,
    N,
    Preload,
    Store,
    check_residency,
    schedule_contraction,
)
from merlin.transforms.dense_planning import (
    DenseProblem,
    DenseResources,
    choose_dense_block,
)

FACTS_SHA256 = "b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af"
# Independently extracted selected address-space view: 4*4096 operand rows,
# 2*512 addressable accumulator rows, 16 i8/i32 lanes. No overlap/port claim.
GEOMETRY = Geometry(16, 16384, 4096, 1024, True, {"facts": FACTS_SHA256})
STATIC_BUDGET = 4096


def _fresh(schedule):
    """Lower shared stationary choices to explicit original fresh PRELOADs."""
    weight = None
    operations = []
    for op in schedule.ops:
        if isinstance(op, Preload):
            if op.weight_row is not None:
                weight = op.weight_row, op.weight_block
            if weight is None:
                raise ValueError("panel preload has no independently scheduled weight")
            op = replace(op, weight_row=weight[0], weight_block=weight[1])
        elif isinstance(op, Compute):
            op = replace(op, fresh_weights=True)
        operations.append(op)
    result = replace(schedule, ops=tuple(operations))
    check_residency(result)
    return result


def stripe_schedule(rows, k, cols, geometry=GEOMETRY):
    return _fresh(
        schedule_contraction(
            Contraction(rows, k, cols),
            geometry,
            Knobs(
                load_on_index_change=True,
                lookahead_steps=0,
                load_grouping=NEST,
                placement=CONTIGUOUS,
                loop_order=(N, K, M),
            ),
        )
    )


def choose_group(m, k, n, geometry=GEOMETRY):
    if any(type(v) is not int or v <= 0 for v in (m, k, n)):
        return None
    dim = geometry.block
    if dim != 16 or geometry.separate_accumulator_space is not True:
        return None
    mt, kt = ((v + dim - 1) // dim for v in (m, k))
    if mt < 2 or max(m * k, k * n, m * n, n * 4) >= 2**31:
        return None
    problem = DenseProblem(m, k, n, dim, dim, dim, 1, 1, 4)
    resources = DenseResources(
        geometry.operand_rows * dim,
        geometry.accumulator_rows * dim * 4,
        max_m_blocks=geometry.accumulator_rows // dim,
        max_n_blocks=1,
        max_k_blocks=kt,
        partial_output_tiles=True,
        partial_output_blocks=True,
    )

    def price(block):
        if block.m < 2 or block.k != kt:
            return None  # No K spill, partial reduction or unproved reload.
        if 4 * block.m * kt + 4 * block.m + kt + 4 > STATIC_BUDGET:
            return None  # Bound proposal expansion before materializing ops.
        operations = stripe_schedule(
            min(m, block.m * dim), k, min(n, dim), geometry
        ).ops
        # Each shared op lowers to at most one raw command, except stores add
        # CONFIG_ST. Explicit seeds/configs are bounded separately here.
        return (
            len(operations)
            + sum(isinstance(op, Store) for op in operations)
            + block.m
            + 4
        )

    return choose_dense_block(
        problem,
        resources,
        static_budget=STATIC_BUDGET,
        price=price,
        admits=lambda block: block.k == kt and block.m >= 2,
    )


def try_emit(
    b,
    a,
    w,
    out,
    m,
    k,
    n,
    attrs,
    *,
    private_rows_owned=False,
    transpose_rhs=False,
    resident=None,
):
    """Return False without emission on unsupported semantics/resource views.

    The original caller must own the complete private row epoch (in particular
    no cached cross-operation RHS may be live) and disjoint external buffers.
    Conv/bias/scaled/i8/transpose routes are deliberately untouched.
    """
    if (
        private_rows_owned is not True
        or transpose_rhs
        or resident is not None
        or not isinstance(attrs, dict)
        or set(attrs) - {"epilogue", "output_dtype"}
        or attrs.get("epilogue", []) != []
        or attrs.get("output_dtype", "i32") != "i32"
        or len({a, w, out}) != 3
    ):
        return False
    choice = choose_group(m, k, n)
    if choice is None:
        return False
    from contraction import (
        ACC,
        ACC_ADD,
        ACC_FULL,
        DIM,
        GARBAGE,
        ONE,
        _address,
        _matrix,
        _mvin_config,
        _mvout_config,
    )

    zero = b.alloc(DIM, "i32")
    b.rocc(0, 4 | (1 << 16) | (ONE << 32), 1 << 48)
    _mvin_config(b, 0, k)
    _mvin_config(b, 1, n)
    _mvin_config(b, 2, 0)
    group_rows = choice.block.m * DIM

    def group(mi, mr, nj, nr):
        schedule = stripe_schedule(mr, k, nr)
        for row in range(0, mr, DIM):
            b.fence()
            b.rocc(14, _address(b, zero), _matrix(b, ACC | row, min(DIM, mr - row), nr))
        pending = None
        for operation in schedule.ops:
            if isinstance(operation, Load):
                if operation.role == "lhs":
                    index = b.op(
                        "add",
                        b.op("mul", b.op("add", mi, operation.dram_row), k),
                        operation.dram_col,
                    )
                    pointer, funct = b.gep(a, index), 2
                else:
                    index = b.op(
                        "add",
                        operation.dram_row * n,
                        b.op("add", nj, operation.dram_col),
                    )
                    pointer, funct = b.gep(w, index), 1
                b.rocc(
                    funct,
                    _address(b, pointer),
                    _matrix(b, operation.row, operation.rows, operation.cols),
                )
            elif isinstance(operation, Preload):
                pending = operation
                b.rocc(
                    6,
                    _matrix(
                        b, operation.weight_row, operation.weight_rows, operation.cols
                    ),
                    _matrix(
                        b,
                        ACC_ADD | operation.accumulator_row,
                        operation.rows,
                        operation.cols,
                    ),
                )
            elif isinstance(operation, Compute):
                if pending is None:
                    raise ValueError("panel compute lost its scheduled descriptor")
                b.rocc(
                    4,
                    _matrix(b, operation.input_row, operation.rows, operation.cols),
                    _matrix(b, GARBAGE, pending.rows, pending.cols),
                )
                b.fence()
                pending = None
            elif isinstance(operation, Store):
                index = b.op(
                    "add",
                    b.op("mul", b.op("add", mi, operation.dram_row), n),
                    b.op("add", nj, operation.dram_col),
                )
                _mvout_config(b, n * 4, 1.0, False)
                b.rocc(
                    3,
                    _address(b, b.gep(out, index, "i32")),
                    _matrix(
                        b,
                        ACC_FULL | operation.accumulator_row,
                        operation.rows,
                        operation.cols,
                    ),
                )
                b.fence()
            else:
                raise TypeError("panel renderer encountered unknown shared effects")

    def columns(nj, nr):
        full = (m // group_rows) * group_rows
        if full:
            b.loop(0, full, group_rows, lambda mi: group(mi, group_rows, nj, nr))
        if full != m:
            group(full, m - full, nj, nr)

    full = (n // DIM) * DIM
    if full:
        b.loop(0, full, DIM, lambda nj: columns(nj, DIM))
    if full != n:
        columns(full, n - full)
    b.fence()
    return True
