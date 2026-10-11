"""Gemmini-owned source catalog and Spike extension selection for Core ATen."""

import hashlib
from pathlib import Path

from merlin.llvmlower import toolchain
from merlin.llvmlower.device_build import DeviceRouting
from merlin.targetgen.plugins import load_module

TARGET = "gemmini"


def prepare_bundle(directory, *, target, facts):
    from merlin.llvmlower.modular_contractions import prepare_bundle as prepare
    from merlin.system.offload import device_dtype_triples

    return prepare(Path(directory), device_dtype_triples(target))


def routing(directory, *, target, package, facts, eligible):
    from merlin.system.offload import device_dtype_triples

    triples = {tuple(shape.dtypes) for _, shape in eligible}
    if len(triples) != 1 or not triples.issubset(set(device_dtype_triples(target))):
        raise ValueError("ambiguous or underivable device datapath")
    operand, weight, accum = next(iter(triples))
    if operand != weight:
        raise ValueError("catalog requires identical operand storage types")
    # Immutable snapshots at different roots must not share cached Python modules.
    identity = hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest()
    backend = load_module(package, "mlir_oot.golden_device_catalog", package_name=f"core_aten_backend_{identity}")
    # The provider owns semantic matching. Unsupported operations remain host work.
    source = (Path(directory) / "model.mlir").read_text()
    _, inventory = backend.build_catalog(source)
    if not inventory["covered_contractions"]:
        return None
    builder = backend.merlin_builder(toolchain.llvm_install() / "bin")
    admission_path = Path(directory) / "spec-int8-admission.json"
    if admission_path.exists():
        import json

        admission = json.loads(admission_path.read_text())
        numeric_policy = admission["selection"]["spec"]

        def checked_builder(prepared, workdir):
            proof = admit_numeric_schedule(Path(prepared).read_text(), package=package, numeric_policy=numeric_policy)
            manifest, obj = builder(prepared, workdir)
            actual = json.loads(Path(manifest).read_text())
            if actual["source_sha256"] != proof["source_sha256"] or any(
                kernel["schedule"] != next(p["schedule"] for p in proof["kernels"] if p["symbol"] == kernel["symbol"])
                for kernel in actual["kernels"]
            ):
                raise ValueError("compiled catalog differs from the proved numerical schedule")
            actual["numeric_schedule_proof"] = proof
            Path(manifest).write_text(json.dumps(actual, indent=2) + "\n")
            return manifest, obj

        selected_builder = checked_builder
    else:
        selected_builder = builder
    return DeviceRouting(
        device=target,
        package_dir=package,
        operand_dtype=operand,
        accum_dtype=accum,
        catalog_builder=selected_builder,
    )


def runner_options():
    from merlin.runtime.backends.spike import spike_path

    binary = Path(spike_path()).resolve()
    # Extension and tool search roots are properties of the selected tool installation.
    prefix = binary.parent.parent
    return {
        "spike_binary": binary,
        "extension": "gemmini",
        "extlib": prefix / "lib" / "libgemmini.so",
        "path_prepend": [prefix.parent / "bin"],
    }


def full_call_oracle(target):
    """Use submitted source-bound device catalogs at the full-call L2 boundary."""
    from merlin.targetgen.rtl.facts import ensure_facts
    from merlin.targetgen.spike_full_call import full_call_adapter
    from merlin.targetgen.target_experiment import load_capability_manifest

    isa = load_capability_manifest(target).contract["isa"]["march"]
    return full_call_adapter(
        target,
        isa=isa,
        runner_options=runner_options(),
        execution_provider=Path(__file__),
        rtl_facts=ensure_facts(target),
    )


def admit_numeric_schedule(source, *, package, numeric_policy):
    """Prove the actual default catalog's WS primitive and i32 aggregation bounds."""
    from merlin.targetgen.operation_numerics import integer_split_k_limit

    identity = hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest()
    backend = load_module(package, "mlir_oot.golden_device_catalog", package_name=f"core_aten_backend_{identity}")
    _, catalog = backend.build_catalog(source)
    # These are the very Shape/GoldenGemm classes used by build_catalog.
    from dataclasses import asdict
    from importlib import import_module

    kernel_module = import_module(backend.__package__ + ".golden_gemm")
    isa = kernel_module.isa
    geometry = kernel_module.F.DIM
    proofs = []
    for kernel in catalog["kernels"]:
        shape = kernel_module.Shape(**kernel["schedule"])
        bound = integer_split_k_limit(numeric_policy, tile_dim=geometry, full_k=shape.k)
        if bound is None or geometry > bound["max_padded_k"]:
            raise ValueError("provider padded mesh K exceeds selected internal MAC bound")
        emitted = kernel_module.GoldenGemm(shape).build()
        commands = list(emitted.walk())
        configs = [op for op in commands if isinstance(op, kernel_module.G.ConfigExOp)]
        if not configs or any(op.attributes["dataflow"].value.data != isa.WEIGHT_STATIONARY for op in configs):
            raise ValueError("numeric proof requires the emitted weight-stationary dataflow")
        if shape.bias or shape.output_dtype != "i32" or shape.scale != 1.0 or shape.relu:
            raise ValueError("numeric proof requires from-zero raw-i32 readout")
        computes = [op for op in commands if isinstance(op, kernel_module.G.ComputeOp)]
        if not computes:
            raise ValueError("schedule emitted no primitive computes")
        extents = []
        reset_rows, accumulation_rows = set(), set()
        for compute in computes:
            k = int(compute.attributes["a_cols"].value.data)
            if not 1 <= k <= geometry <= bound["max_padded_k"]:
                raise ValueError("emitted compute exceeds proved padded primitive K")
            siblings = list(compute.parent_block().ops)
            prior = siblings[: siblings.index(compute)]
            preload = next((op for op in reversed(prior) if isinstance(op, kernel_module.G.PreloadOp)), None)
            if preload is None or preload.attributes["bd_rows"].value.data != k:
                raise ValueError("compute lacks matching stationary weight preload")
            if compute.a("bd", isa.GARBAGE_ADDR) != isa.GARBAGE_ADDR:
                raise ValueError("WS mesh D must be garbage, not prior accumulator feedback")
            c = int(preload.attributes["c"].value.data)
            if not c & isa.ACC_ADDR_BIT:
                raise ValueError("WS partial dot product does not target accumulator SRAM")
            row = c & ~(isa.ACC_ADDR_BIT | isa.ACC_ACCUMULATE_BIT | isa.ACC_FULL_ROW_BIT)
            if c & isa.ACC_ACCUMULATE_BIT:
                if row not in reset_rows:
                    raise ValueError("i32 accumulation lacks prior zero/overwrite initialization")
                accumulation_rows.add(row)
            else:
                reset_rows.add(row)
            # In WS, PRELOAD's BD is B, D is garbage (no accumulator feedback
            # into the 20-bit mesh). C's accumulation bit addresses 32-bit SRAM.
            extents.append(k)
        if shape.k > geometry and reset_rows != accumulation_rows:
            raise ValueError("not all initialized output rows receive later K tiles")
        stores = [op for op in commands if isinstance(op, kernel_module.G.MvoutOp)]
        if not stores or any(not op.a("local") & isa.ACC_FULL_ROW_BIT for op in stores):
            raise ValueError("accumulator DMA readout is not raw full-i32")
        maximum = geometry * bound["maximum_product"]
        proofs.append(
            dict(
                symbol=kernel["symbol"],
                schedule=asdict(shape),
                status="proven",
                primitive_k=sorted(set(extents)),
                initialized_accumulator_rows=sorted(reset_rows),
                accumulated_rows=sorted(accumulation_rows),
                padded_primitive_k=geometry,
                k_split=[min(geometry, shape.k - start) for start in range(0, shape.k, geometry)],
                max_partial_sum_bound=maximum,
                mac_positive_limit=(1 << (bound["mac_result_bits"] - 1)) - 1,
                headroom=((1 << (bound["mac_result_bits"] - 1)) - 1) - maximum,
                full_result_bound=bound["full_result_bound"],
                bound=bound,
                aggregation="WS mesh independent dot products; accumulator SRAM i32 overwrite first K tile, add subsequent K tiles",
                mesh_initialization="WS D is garbage; compute accumulate preserves stationary B, not a preceding dot product",
                emitted_schedule_sha256=hashlib.sha256(str(emitted).encode()).hexdigest(),
            )
        )
    if not proofs or not catalog["coverage_complete"]:
        raise ValueError("provider did not cover every proposed integer contraction")
    return dict(status="proven", source_sha256=catalog["source_sha256"], kernels=proofs, bindings=catalog["bindings"])
