"""Bind the original primitive caller without conflating software roles.

The arithmetic specification, observed effects and logical-pointer transport
contract are separate selections. Re-rendering the existing original caller is
only a byte correspondence check; it supplies no kernel or execution authority.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import MethodType

import numpy as np
from gemmini_timing_resources import _pin
from merlin.runtime import harness_render
from merlin.runtime.backends import base
from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
from merlin.targetgen import target_registry
from merlin.targetgen.contract.readback_policy import ReadbackPolicy, canonical_sha256

_METHODS = {
    name: getattr(BoundRoCCBackend, name)
    for name in ("_data", "render_harness", "build_source_paths")
}


def _original_backend():
    backend = base.get_backend("gemmini")
    if type(backend) is not BoundRoCCBackend or backend.target != "gemmini":
        raise ValueError("primitive caller requires its original data-bound backend")
    for name, original in _METHODS.items():
        callback = getattr(backend, name)
        if (
            type(callback) is not MethodType
            or callback.__self__ is not backend
            or callback.__func__ is not original
        ):
            raise ValueError("primitive caller original backend method was replaced")
    return backend


def check_original_caller(selected, arm, values):
    """Reopen exact facts/transport and require the completed harness bytes.

    This path deliberately supports only the original data-bound backend and
    its ordinary cold, full-B64 primitive pointer caller. Other callers refuse.
    """
    selected.revalidate()
    backend = _original_backend()
    data = backend._data()
    info = target_registry.resolve("gemmini")
    path = info.capability_contract_path
    contract_pin = _pin(path)
    contract = info.load_contract()
    facts, effects, _operands = selected._documents()
    if (
        data != {"contract": contract, "facts": facts}
        or contract.get("name") != effects.get("name")
        or any(
            contract.get(key) != effects.get(key)
            for key in ("rtl_source", "harness_abi", "runner")
        )
    ):
        raise ValueError(
            "primitive caller facts/runtime/ABI differ from selected effects"
        )
    harness_render.validate_contract(contract)
    sources = [_pin(path) for path in backend.build_source_paths()]
    build = arm["build_record"]
    if (
        build.get("target") != "gemmini"
        or build.get("readback_policy") != ReadbackPolicy("out_b64_v1").record()
        or any(pin not in build.get("source_pins", ()) for pin in sources)
    ):
        raise ValueError(
            "primitive completed build lacks its original caller source/transport"
        )
    cb = arm["command_buffer"]
    order = arm["caller_order"]
    if set(order) != set(cb["tensors"]) or len(order) != len(cb["tensors"]):
        raise ValueError("primitive original caller pointer order is incomplete")
    caller_cb = dict(cb, tensors={name: cb["tensors"][name] for name in order})
    blobs = {}
    rendered = backend.render_harness(
        caller_cb,
        target="gemmini",
        inputs={name: np.asarray(value).tolist() for name, value in values.items()},
        readback_policy=ReadbackPolicy("out_b64_v1"),
        blobs=blobs,
    )
    harness = Path(arm["harness"]["path"])
    if (
        blobs
        or _pin(harness) != arm["harness"]
        or rendered.encode("utf-8") != harness.read_bytes()
        or _pin(path) != contract_pin
        or any(_pin(pin["path"]) != pin for pin in sources)
        # The original registry deliberately returns a newly bound immutable
        # data wrapper on each lookup. Its selected data/methods, not Python
        # object identity, establish this correspondence.
        or _original_backend()._selection_json != backend._selection_json
        or backend._data() != data
    ):
        raise ValueError("primitive actual completed caller bytes/selection changed")
    selected.revalidate()
    return {
        "contract": contract,
        "normalized": harness_render._roster(caller_cb)[0],
        "binding": {
            "capability_contract": contract_pin,
            "facts": _pin(selected.facts_path),
            "observed_effects": _pin(selected.contract_path),
            "backend_selection_sha256": hashlib.sha256(
                backend._selection_json.encode()
            ).hexdigest(),
            "caller_sources": sources,
            "original_harness": arm["harness"],
            "original_pointer_order": list(order),
            "caller_command_buffer_sha256": canonical_sha256(caller_cb),
            "scope": "original caller byte correspondence; no execution or timing qualification",
        },
    }
