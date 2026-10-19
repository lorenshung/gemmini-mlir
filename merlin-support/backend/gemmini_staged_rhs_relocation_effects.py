"""Independent actual packet relocation and immutable-RHS liveness proof.

No candidate plan or placement implementation is imported. Complete original
source/native/output checking and the original whole-program private epoch are
prerequisites; this comparison issues neither a compiler capability nor timing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from gemmini_arithmetic_scope import IntegerArithmeticOwner
from gemmini_cleanup_final_guard import _logical
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_source_execution import SourceExecutionOwner
from gemmini_staged_dense_capability import _digest, _leaf
from gemmini_staged_rhs_window import ClosedScratchEpochOwner, check_original_footprint
from gemmini_staged_tile_effects import check_output_tile_permutation
from gemmini_timing_resources import _pin, _revalidate_sources
from gemmini_zero_initializer_effects import canonical_issue_stream

from merlin.common.paths import module_source_path


def canonical_relocated_stream(stream, *, lhs, rhs, problem, dim, data_bits, shape_fields, baseline):
    """Name only independently proved RHS cells by their canonical private row.

    The fixed caller first proves all actual source products, runtime, zero and
    full outputs. ``shape_fields`` are derived selected ISA bitfields, not prices
    or a policy. A pending descriptor reads the cells present at consumption.
    """
    if type(baseline) is not bool or lhs == rhs or not 0 < data_bits <= 32:
        raise ValueError("actual RHS relocation lacks exact owners or selected address fields")
    mask = (1 << data_bits) - 1
    result, cells, pending, stride = [], {}, None, None

    def field(name, raw):
        offset, width = shape_fields[name]
        if type(offset) is not int or type(width) is not int or offset < data_bits or not 0 < width <= 16:
            raise ValueError("actual RHS relocation has unsupported selected shape fields")
        return raw >> offset & ((1 << width) - 1)

    for original, raw_command in stream:
        event = copy.deepcopy(list(original))
        command = copy.deepcopy(raw_command)
        kind = event[0]
        fields = {} if command is None else command["decoded"]
        index = len(result)
        if kind in {"CONFIG_EX", "FLUSH"}:
            if pending is not None:
                raise ValueError("actual RHS relocation crosses pending descriptor ownership")
            cells.clear()
        elif kind == "CONFIG_LD":
            stride = fields["stride"]
        elif kind == "MVIN" and fields.get("is_acc_addr") is False:
            role = event[1][0]
            offset = event[1][1]
            if role not in {lhs, rhs} or type(stride) is not int or stride != (problem.k if role == lhs else problem.n):
                raise ValueError("actual RHS relocation has unknown source ownership or load stride")
            row, col = divmod(offset, stride)
            address = fields["addr"]
            if not 0 < fields["rows"] <= dim or not 0 < fields["cols"] <= dim:
                raise ValueError("actual RHS relocation has invalid physical transfer geometry")
            required = 0 if role == lhs else dim + (0 if baseline else row)
            if address != required or role == rhs and (row % dim or col % dim):
                raise ValueError("actual RHS relocation has an unadmitted physical operand placement")
            for r in range(fields["rows"]):
                for c in range(fields["cols"]):
                    cells[address + r, c] = (role, row + r, col + c)
            if role == rhs:
                address = dim + row
                event[2] = (event[2] & ~mask) | address
                command["rs2"]["raw"] = event[2]
                fields["addr"] = address
        elif kind == "PRELOAD":
            if pending is not None:
                raise ValueError("actual RHS relocation overwrites a pending descriptor")
            pending = index, event[1] & mask, field("rows", event[1]), field("cols", event[1])
        elif kind == "COMPUTE_PRELOADED":
            if pending is None or event[1] & mask:
                raise ValueError("actual RHS relocation loses original compute/LHS ownership")
            descriptor, address, rows, cols = pending
            first = cells.get((address, 0))
            if first is None or first[0] != rhs or not 0 < rows <= dim or not 0 < cols <= dim:
                raise ValueError("actual RHS relocation lacks consumed immutable RHS source cells")
            k0, n0 = first[1:]
            a = cells.get((event[1] & mask, 0))
            if (
                k0 % dim
                or n0 % dim
                or address != (dim if baseline else dim + k0)
                or a is None
                or a[0] != lhs
                or a[2] != k0
                or any(cells.get((event[1] & mask, k)) != (lhs, a[1], k0 + k) for k in range(rows))
                or any(cells.get((address + r, c)) != (rhs, k0 + r, n0 + c) for r in range(rows) for c in range(cols))
            ):
                raise ValueError("actual RHS relocation reads mixed, clobbered or unowned panel cells")
            before, decoded = result[descriptor]
            before[1] = (before[1] & ~mask) | (dim + k0)
            decoded["rs1"]["raw"] = before[1]
            pending = None
        elif kind == "COMPUTE_ACCUMULATE":
            raise ValueError("actual RHS relocation has no retained-weight qualification")
        result.append((event, command))
    if pending is not None:
        raise ValueError("actual RHS relocation leaves a pending descriptor")
    return result


class RhsPlacementRealizationChecker:
    def __init__(self, *, selected, arithmetic, nm):
        if type(selected) is not SelectedSchedulingTarget or type(arithmetic) is not IntegerArithmeticOwner:
            raise ValueError("RHS relocation requires its original live selected staged source/effect owners")
        self.selected, self.arithmetic, self.nm = selected, arithmetic, Path(nm)
        self._selected = selected.report()["selected_inputs"]
        self._arithmetic = arithmetic.record()
        self._dim = selected.report()["geometry"]["array_rows"]
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_staged_rhs_relocation_effects",
                "gemmini_staged_rhs_window",
                "gemmini_staged_rhs_placement",
                "gemmini_zero_initializer_effects",
                "gemmini_staged_tile_effects",
                "gemmini_staged_rhs_effects",
                "gemmini_source_execution",
                "gemmini_staged_source_pair",
            )
        }
        self._identity = _digest([self._selected, self._arithmetic, self._dim, self._sources])
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.rhs_placement_realization.v1:" + self._identity

    def revalidate(self):
        _revalidate_sources(self._sources)
        if (
            self.selected.report()["selected_inputs"] != self._selected
            or self.arithmetic.record() != self._arithmetic
            or self._arithmetic["selected_inputs"] != self._selected
            or _digest([self._selected, self._arithmetic, self._dim, self._sources]) != self._identity
        ):
            raise ValueError("original RHS relocation selection/checks changed")

    def record(self):
        self.revalidate()
        return dict(
            identity=self.identity,
            sources=copy.deepcopy(self._sources),
            selected_inputs=copy.deepcopy(self._selected),
            arithmetic=copy.deepcopy(self._arithmetic),
            scope="current complete functional program/row placement/source/products/output checks only",
            compiler_capability_issued=False,
            hardware_execution_qualified=False,
            timing_authority="NONE",
        )

    def check_pair_artifacts(self, before, after, *, execution_domain):
        self.revalidate()
        if execution_domain != "functional_spike":
            raise ValueError("RHS placement has no qualified hardware/timing domain")
        p = self
        native = json.loads(_leaf(before, "native_result").read_bytes())
        source_owner = SourceExecutionOwner(
            arithmetic=p.arithmetic,
            selected=p.selected,
            result=_leaf(before, "native_result"),
            command_buffer=_leaf(before, "command_buffer_artifact"),
            lowered=_leaf(before, "lowered"),
            trace_invocation=_leaf(before, "issue_invocation"),
            nm=p.nm,
        )
        epoch = ClosedScratchEpochOwner(
            source_owner, capsule_dir=Path(native["source"]["capsule_declaration"]["path"]).parent
        )
        base = epoch.check_source_arm(before)
        candidate = epoch.check_source_arm(after)
        if _logical(base[3]) != _logical(candidate[3]):
            raise ValueError("RHS placement changed complete source/task/ABI/work accountability")
        from merlin.compile.scheduling.staged_dense import StagedDenseProblem

        problem = StagedDenseProblem(**base[2]["source_geometry"])
        dim = self._dim
        required = dim + ((problem.k + dim - 1) // dim) * dim
        original = epoch.record()
        footprint = original["footprint"]
        try:
            check_original_footprint(
                candidate[0],
                lhs=footprint["lhs_abi_owner"],
                rhs=candidate[1],
                dim=dim,
                data_bits=footprint["address_data_bits"],
                physical_rows=footprint["maximum_private_physical_rows"],
            )
            relocated = False
        except ValueError:
            relocated = True
        if relocated and required > footprint["maximum_private_physical_rows"]:
            raise ValueError("RHS placement exceeds independently owned padded full-K window")
        fields = p.selected._documents()[2]["layouts"]["fields"]
        shape = {}
        for name, key in [("rows", "num_rows"), ("cols", "num_cols")]:
            field = next(row for row in fields if row["bundle"] == "PreloadRs" and row["name"] == key)
            if field["status"] != "derived":
                raise ValueError("RHS relocation lacks selected ISA shape fields")
            shape[name] = (field["offset"], field["width"])

        def logical(physical, cb):
            return "abi/" + cb["kernel_abi"]["args"][int(physical.removeprefix("tensor_"))]["tensor"]

        lhs = logical(original["footprint"]["lhs_abi_owner"], base[3])
        rhs = logical(base[1], base[3])
        if not relocated:
            effects = check_output_tile_permutation(
                canonical_issue_stream(base[0], base[3]),
                canonical_issue_stream(candidate[0], candidate[3]),
                rhs=rhs,
                operand_rows=2 * dim,
                dim=dim,
            )
            self.revalidate()
            epoch.revalidate()
            return dict(
                status="checked_original_private_panel_fallback",
                original_epoch=original,
                original_source_execution=[base[2], candidate[2]],
                actual_issue_effects=effects,
                assigned_operand_rows=2 * dim,
                full_k_placement_applied=False,
                compiler_capability_issued=False,
                hardware_execution_qualified=False,
                timing_authority="NONE",
            )
        streams = []
        for arm, is_baseline in [(base, True), (candidate, False)]:
            streams.append(
                canonical_relocated_stream(
                    canonical_issue_stream(arm[0], arm[3]),
                    lhs=lhs,
                    rhs=rhs,
                    problem=problem,
                    dim=dim,
                    data_bits=original["footprint"]["address_data_bits"],
                    shape_fields=shape,
                    baseline=is_baseline,
                )
            )
        effects = check_output_tile_permutation(*streams, rhs=rhs, operand_rows=required, dim=dim)
        self.revalidate()
        epoch.revalidate()
        return dict(
            status="checked_original_private_rhs_relocation_and_actual_issue",
            original_epoch=original,
            original_source_execution=[base[2], candidate[2]],
            actual_issue_effects=effects,
            assigned_operand_rows=required,
            full_k_placement_applied=True,
            compiler_capability_issued=False,
            hardware_execution_qualified=False,
            timing_authority="NONE",
        )
