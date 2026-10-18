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
    # A closed seeded MAC + byte readout needs the region ABI: its seed and
    # result types differ from the old two-input/raw-i32 contraction ABI.
    regions = _core_aten_module(package, "integer_regions")
    parsing = _core_aten_module(package, "catalog")
    source = (Path(directory) / "model.mlir").read_text()
    if regions.has_integer_readout(parsing.parse_module(source)):
        _, inventory, matches = parsing.build(source)
        selected = [m for m in matches if m.kernel.meta.get("integer_readout")]
        if not selected:
            return None
        def region_builder(prepared, workdir):
            # Re-prove the actual prepared bytes, not a preceding capture.
            if (Path(directory) / "spec-int8-admission.json").exists():
                import json
                policy = json.loads((Path(directory) / "spec-int8-admission.json").read_text())["selection"]["spec"]
                admit_numeric_schedule(Path(prepared).read_text(), package=package, numeric_policy=policy)
            return core_aten_offload(prepared, workdir, package=package)
        audit = load_module(package, "mlir_oot.golden_device_catalog", package_name="core_aten_region_audit")
        return DeviceRouting(device=target, package_dir=package, operand_dtype=operand,
                             accum_dtype=accum, region_offload=region_builder,
                             final_elf_audit=audit.final_elf_audit)
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
        final_elf_audit=backend.final_elf_audit,
    )


def _core_aten_module(package, name):
    identity = hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest()
    return load_module(package, f"mlir_oot.core_aten.{name}", package_name=f"core_aten_backend_{identity}")


def core_aten_catalog(prepared, workdir, *, package, llvm_bin=None):
    """Bind and compile the exactly implementable Core ATen regions of ``prepared``.

    Returns (manifest path, object path).  The manifest (schema gemmini_core_aten_catalog_v1)
    lists each kernel's argument table and each binding's root ordinal, replaced ordinals and
    operand sources; regions it cannot implement exactly are listed under ``declines``.  This is
    opt-in beside ``routing``: the integer GEMM catalog and its bindings are unchanged.
    """
    catalog = _core_aten_module(package, "catalog")
    workdir = Path(workdir)
    catalog.compile_catalog(Path(prepared), workdir,
                            llvm_bin=Path(llvm_bin) if llvm_bin else toolchain.llvm_install() / "bin")
    return workdir / "core_aten_catalog.json", workdir / "core_aten.o"


def core_aten_offload(prepared, workdir, *, package, llvm_bin=None):
    """Rewrite the bound regions of ``prepared`` into C-interface calls.

    Returns (rewritten MLIR, shim C source, kernel object, manifest).  After any lowering with
    function-boundary bufferization and ``llvm.emit_c_interface`` lowering, the rewritten model calls
    ``_mlir_ciface_ca_<symbol>``, which the shim defines over the kernel object.  The host image
    must link both (and provide malloc for the call results' destinations).
    """
    import json

    manifest_path, obj = core_aten_catalog(prepared, workdir, package=package, llvm_bin=llvm_bin)
    offload = _core_aten_module(package, "offload")
    manifest = json.loads(Path(manifest_path).read_text())
    workdir = Path(workdir)
    rewritten = workdir / "core_aten_offloaded.mlir"
    rewritten.write_text(offload.rewrite(Path(prepared).read_text(), manifest) + "\n")
    shim = workdir / "core_aten_shim.c"
    shim.write_text(offload.shim_source(manifest))
    return rewritten, shim, obj, manifest_path


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

    regions = _core_aten_module(package, "integer_regions")
    catalog_module = _core_aten_module(package, "catalog")
    if regions.has_integer_readout(catalog_module.parse_module(source)):
        return _admit_integer_regions(source, package=package, numeric_policy=numeric_policy)
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


def _admit_integer_regions(source, *, package, numeric_policy):
    """Prove the emitted seeded MAC/readout stream, including its source binding."""
    from merlin.targetgen.operation_numerics import integer_split_k_limit
    catalog_module = _core_aten_module(package, "catalog")
    _, catalog, matches = catalog_module.build(source)
    proofs = []
    for binding, match in zip(catalog["bindings"], matches):
        if not match.kernel.meta.get("integer_readout"):
            continue
        dims = match.kernel.meta["dimensions"]
        bound = integer_split_k_limit(numeric_policy, tile_dim=16, full_k=dims["k"])
        if bound is None or bound["max_padded_k"] < 16:
            raise ValueError("provider padded mesh K exceeds selected internal MAC bound")
        stream = match.kernel.stream
        computes = [i for i in stream.ins if i.kind == "compute"]
        if not computes or any(i.a("a_cols") > 16 or i.a("bd") != 0xffffffff for i in computes):
            raise ValueError("seeded WS primitive differs from the proved independent dot product")
        if any(i.a("dataflow") != 1 for i in stream.ins if i.kind == "config_ex"):
            raise ValueError("seeded proof requires weight-stationary execution")
        # The bound producer is exact integer math with raw full-width DMA.
        # Source exact readout remains on the host, including bias and RNE.
        stores=[i for i in stream.ins if i.kind=='mvout']
        if not stores or any(not (i.a('local') & (1 << 29)) for i in stores):
            raise ValueError('exact producer requires full-width i32 readout')
        if any(i.a('scale') != 1 for i in stream.ins if i.kind=='config_ld') or any(
                i.a('acc_scale') != 1 for i in stream.ins if i.kind=='config_st'):
            raise ValueError('exact producer contains an unproved FP32 data-path scale')
        proofs.append(dict(symbol=binding["symbol"], status="proven", schedule=match.kernel.meta,
            primitive_k=sorted({i.a("a_cols") for i in computes}), padded_primitive_k=16,
            k_split=[min(16, dims["k"]-k) for k in range(0,dims["k"],16)],
            max_partial_sum_bound=16*bound["maximum_product"],
            mac_positive_limit=(1 << (bound["mac_result_bits"]-1))-1,
            headroom=((1 << (bound["mac_result_bits"]-1))-1)-16*bound["maximum_product"],
            bound=bound, exact_arithmetic=True, device_readout="full_i32",
            quantization="retained_source_exact_integer_host_readout", seed=match.kernel.meta["seed"],
            accumulator_min=match.kernel.meta["accumulator_min"], accumulator_max=match.kernel.meta["accumulator_max"],
            emitted_schedule_sha256=hashlib.sha256(repr(stream.ins).encode()).hexdigest()))
    if not proofs:
        raise ValueError("provider did not cover any exact seeded readout region: " + str(catalog["declines"]))
    return dict(status="proven", source_sha256=catalog["source_sha256"], kernels=proofs,
                bindings=catalog["bindings"], declines=catalog["declines"], abi="gemmini_core_aten_catalog_v1")
