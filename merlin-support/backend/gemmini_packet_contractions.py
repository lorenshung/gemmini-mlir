"""Independent bounded state induction over actual issued WS operands.

No selected plan establishes this check. Loads label actual private cells,
PRELOAD retains its descriptor across fences, COMPUTE consumes current cells,
and every readout must contain each exact source product once. This is a
functional source-effect checker, not an asynchronous or hardware timing model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Contraction:
    ordinal: int
    lhs: str
    rhs: str
    output: str
    m: int
    k: int
    n: int


def check_packets(
    stream,
    *,
    contractions,
    dim,
    spad_rows,
    acc_rows,
    address_bits,
    data_bits,
    accumulator_flag,
    accumulate_flag,
    full_width_flag,
    memory_cell,
    zero_cell,
    execute_contract,
    weight_shape,
    compute_shape,
    preload_shape,
    load_configuration,
):
    """Check a complete observed stream under original source/effect callbacks.

    memory_cell(address,width) returns the current logical tensor and element
    index from original linked ABI bindings. zero_cell independently proves a
    linked immutable zero element. Missing values or unsupported effects refuse.
    The callbacks and complete trace/source custody belong to the original owner.
    """
    if (
        not contractions
        or len(contractions) > 1024
        or not 0 < dim <= 256
        or any(
            not isinstance(c, Contraction) or min(c.m, c.k, c.n) < 1
            for c in contractions
        )
    ):
        raise ValueError("packet checker requires bounded exact source contractions")
    if len({(c.lhs, c.rhs) for c in contractions}) != len(contractions):
        raise ValueError(
            "repeated identical operand owners need a separate invocation binding"
        )
    if sum(c.m * c.n for c in contractions) > 2_000_000:
        raise ValueError("packet checker exceeds its independent output-cell budget")
    pairs = {(c.lhs, c.rhs): c for c in contractions}
    spad, acc, outputs = {}, {}, set()
    pending, stationary, ex, load, store = None, None, False, None, None
    loads = computes = products = fences_with_pending = 0
    address_mask = (1 << address_bits) - 1
    data_mask = (1 << data_bits) - 1

    def local(raw, *, accumulator, extra=0):
        allowed = data_mask | (accumulator_flag if accumulator else 0) | extra
        if raw & ~allowed or bool(raw & accumulator_flag) != accumulator:
            raise ValueError(
                "actual local address has unsupported flags/normalization/garbage"
            )
        base = raw & data_mask
        if base >= (acc_rows if accumulator else spad_rows):
            raise ValueError("actual local row exceeds selected private storage")
        return base

    def extent(rows, cols):
        if any(type(x) is not int or not 0 < x <= dim for x in (rows, cols)):
            raise ValueError(
                "actual primitive shape is absent or exceeds the physical panel"
            )

    for index, command in enumerate(stream):
        if index >= 1_000_000:
            raise ValueError("actual packet stream exceeds independent command budget")
        kind, fields = command["class"], command.get("decoded", {})
        raw1, raw2 = (command.get(key, {}).get("raw", 0) for key in ("rs1", "rs2"))
        if kind == "FENCE":
            fences_with_pending += pending is not None
            continue
        if kind == "FLUSH":
            if loads or computes or pending is not None:
                raise ValueError("flush changes live arithmetic ownership")
            continue
        if kind == "CONFIG_EX":
            if pending is not None:
                raise ValueError("EX change cannot retire a pending PRELOAD")
            execute_contract(raw1, raw2)
            ex = True
        elif kind == "CONFIG_LD":
            load_configuration(command)
            load = fields["stride"]
        elif kind == "CONFIG_ST":
            forbidden = (
                "acc_act",
                "pool_stride",
                "pool_size",
                "upad",
                "lpad",
                "pool_out_dim",
                "porows",
                "pocols",
                "orows",
                "ocols",
            )
            if (
                any(fields.get(name) != 0 for name in forbidden)
                or fields.get("acc_scale") != 1.0
            ):
                raise ValueError(
                    "actual readout has unsupported activation/scale/pooling effects"
                )
            store = fields["out_stride_bytes"]
        elif kind == "MVIN":
            if load is None:
                raise ValueError("actual load lacks complete selected configuration")
            rows, cols = fields["rows"], fields["cols"]
            extent(rows, cols)
            is_acc = fields["is_acc_addr"]
            base = local(fields["addr"], accumulator=is_acc)
            if base + rows > (acc_rows if is_acc else spad_rows):
                raise ValueError("actual transfer crosses private storage boundary")
            for row in range(rows):
                for col in range(cols):
                    address = raw1 + row * load + col * (4 if is_acc else 1)
                    if is_acc:
                        if load != 0 or not zero_cell(address, 4):
                            raise ValueError(
                                "accumulator initialization is not independently proven zero"
                            )
                        acc[base + row, col] = None
                    else:
                        spad[base + row, col] = memory_cell(address, 1)
            loads += 1
        elif kind == "PRELOAD":
            if not ex or pending is not None:
                raise ValueError(
                    "PRELOAD lacks EX state or overwrites a pending descriptor"
                )
            wr, wc = weight_shape(raw1)
            cr, cc = preload_shape(raw2)
            extent(wr, wc)
            extent(cr, cc)
            if wc != cc:
                raise ValueError("actual PRELOAD weight/output columns disagree")
            wb = (
                None
                if raw1 & address_mask == address_mask
                else local(raw1 & address_mask, accumulator=False)
            )
            cb = local(raw2 & address_mask, accumulator=True, extra=accumulate_flag)
            pending = (wb, cb, wr, wc, cr, bool(raw2 & accumulate_flag))
        elif kind in {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"}:
            if (
                not ex
                or fields.get("garbage") is not True
                or fields.get("bd") != address_mask
            ):
                raise ValueError(
                    "actual WS compute lacks exact zero incoming spatial partial sum"
                )
            if pending is None:
                raise ValueError(
                    "actual compute has no pending PRELOAD/output ownership"
                )
            wb, cb, wr, wc, cr, continuation = pending
            ar, ak = compute_shape(raw1)
            extent(ar, ak)
            if (ar, ak) != (cr, wr):
                raise ValueError("actual A/B/C primitive domains disagree")
            ab = local(raw1 & address_mask, accumulator=False)
            if kind == "COMPUTE_ACCUMULATE":
                if stationary is None or (wr, wc) != stationary[:2]:
                    raise ValueError(
                        "retained weights lack a matching independently observed preload"
                    )
                weights = stationary[2]
            else:
                if wb is None:
                    raise ValueError(
                        "zero preload has no exact source operand ownership"
                    )
                weights = tuple(
                    tuple(spad.get((wb + k, col)) for col in range(wc))
                    for k in range(wr)
                )
                stationary = (wr, wc, weights)
            for row in range(ar):
                for col in range(wc):
                    a = [spad.get((ab + row, k)) for k in range(ak)]
                    b = [weights[k][col] for k in range(ak)]
                    if None in a or None in b:
                        raise ValueError(
                            "compute reads unowned or clobbered operand cells"
                        )
                    pair = (a[0][0], b[0][0])
                    source = pairs.get(pair)
                    if source is None:
                        raise ValueError(
                            "actual compute operand owners have no exact typed source contraction"
                        )
                    m, k0 = divmod(a[0][1], source.k)
                    kb, n = divmod(b[0][1], source.n)
                    if (
                        m >= source.m
                        or n >= source.n
                        or kb != k0
                        or k0 + ak > source.k
                        or a != [(source.lhs, m * source.k + k0 + k) for k in range(ak)]
                        or b
                        != [(source.rhs, (k0 + k) * source.n + n) for k in range(ak)]
                    ):
                        raise ValueError(
                            "actual product cells differ from complete source matrix geometry"
                        )
                    location = cb + row, col
                    if location not in acc:
                        raise ValueError("compute writes an uninitialized accumulator")
                    previous = acc[location]
                    expected = None if k0 == 0 else (source.ordinal, m, n, k0)
                    if previous != expected or (
                        previous is not None and not continuation
                    ):
                        raise ValueError(
                            "actual continuation duplicates/omits K or changes accumulator ownership"
                        )
                    if not continuation and k0 != 0:
                        raise ValueError(
                            "actual nonaccumulating write loses a reduction prefix"
                        )
                    acc[location] = (source.ordinal, m, n, k0 + ak)
                    products += ak
            computes += 1
            pending = None
        elif kind == "MVOUT":
            if pending is not None or store is None or fields.get("readout") != "i32":
                raise ValueError(
                    "actual readout has pending work or unsupported numerical state"
                )
            rows, cols = fields["rows"], fields["cols"]
            extent(rows, cols)
            base = local(fields["addr"], accumulator=True, extra=full_width_flag)
            for row in range(rows):
                for col in range(cols):
                    tensor, element = memory_cell(raw1 + row * store + 4 * col, 4)
                    found = [
                        source for source in contractions if source.output == tensor
                    ]
                    if len(found) != 1:
                        raise ValueError(
                            "actual readout destination lacks unique source output ownership"
                        )
                    source = found[0]
                    m, n = divmod(element, source.n)
                    key = source.ordinal, m, n
                    if (
                        m >= source.m
                        or key in outputs
                        or acc.get((base + row, col)) != (*key, source.k)
                    ):
                        raise ValueError(
                            "actual readout duplicates, omits or misroutes a complete source reduction"
                        )
                    outputs.add(key)
        else:
            raise ValueError(
                "actual packet has unsupported arithmetic/state/memory effects: " + kind
            )
    if pending is not None:
        raise ValueError("final fence does not establish lone PRELOAD retirement")
    if (
        not computes
        or products != sum(c.m * c.k * c.n for c in contractions)
        or len(outputs) != sum(c.m * c.n for c in contractions)
    ):
        raise ValueError(
            "actual accelerator work/outputs do not cover every required source product"
        )
    return {
        "status": "checked_actual_source_product_coverage",
        "computes": computes,
        "products": products,
        "raw_i32_values": len(outputs),
        "fences_preserving_pending_preload": fences_with_pending,
        "hardware_qualification": False,
        "timing_authority": "NONE",
    }
