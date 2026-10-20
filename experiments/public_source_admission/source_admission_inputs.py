"""Closed operator data for the original public-source admission experiment.

These checks grant no execution, arithmetic, timing or publication authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

SCHEMA = "gemmini.public_source_admission.v1"
TARGET_SCHEMA = "gemmini.public_source_admission.v2"
FIELDS = {
    "schema",
    "readiness",
    "job_plan",
    "source",
    "roles",
    "compiler",
    "factory_inputs",
    "execution_contract",
    "purpose",
}
SOURCE_FIELDS = {
    "identity",
    "manifest",
    "ordinal",
    "attestation",
    "program",
    "source_member",
    "supplements",
    "input_binding",
    "cost_limits",
}


class Intake:
    """Bounded, mutation-sensitive input reopening; never a saved PASS cache."""

    def __init__(self):
        self.pins = {}

    def read(self, pin, *, maximum=8 << 20):
        if type(pin) is not dict or set(pin) not in (
            {"path", "sha256"},
            {"path", "bytes", "sha256"},
        ):
            raise ValueError("expected one exact ordinary input pin")
        path = Path(pin["path"])
        if (
            not path.is_absolute()
            or path.resolve() != path
            or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file()
        ):
            raise ValueError("input must be one canonical ordinary file")
        before = path.stat()
        identity = lambda st: (
            st.st_dev,
            st.st_ino,
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
        )
        if before.st_size > maximum:
            raise ValueError("input exceeds selected bounded intake")
        with path.open("rb") as stream:
            if identity(os.fstat(stream.fileno())) != identity(before):
                raise ValueError("input changed before bounded read")
            raw = stream.read(maximum + 1)
            if (
                len(raw) > maximum
                or identity(os.fstat(stream.fileno())) != identity(before)
                or identity(path.stat()) != identity(before)
                or path.resolve() != path
                or any(p.is_symlink() for p in (path, *path.parents))
            ):
                raise ValueError("input changed during bounded read")
        if hashlib.sha256(raw).hexdigest() != pin["sha256"] or (
            "bytes" in pin
            and (type(pin["bytes"]) is not int or len(raw) != pin["bytes"])
        ):
            raise ValueError("selected input bytes changed")
        self.pins[str(path)] = (dict(pin), maximum)
        return raw

    def document(self, pin):
        from merlin.common.strict_json import loads

        return loads(self.read(pin), max_bytes=8 << 20)

    def finish(self):
        for pin, maximum in tuple(self.pins.values()):
            self.read(pin, maximum=maximum)


def require_fields(value, names, owner):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(owner + " has missing or foreign fields")


def read_declaration(path, sha256, *, intake=None):
    intake = intake or Intake()
    document = intake.document({"path": str(Path(path).absolute()), "sha256": sha256})
    if (
        type(document) is not dict
        or type(document.get("schema")) is not str
        or document.get("schema")
        not in {
            SCHEMA,
            TARGET_SCHEMA,
        }
    ):
        raise ValueError("unknown source admission declaration version")
    target_publication = document["schema"] == TARGET_SCHEMA
    require_fields(
        document,
        FIELDS | ({"publication"} if target_publication else set()),
        "source admission declaration",
    )
    if target_publication:
        require_fields(
            document["publication"], {"scope"}, "selected target publication"
        )
        if document["publication"]["scope"] != "selected_target_v1":
            raise ValueError("unknown selected target publication scope")
    require_fields(document["source"], SOURCE_FIELDS, "selected source")
    require_fields(
        document["roles"], {"profile", "private_exclusion"}, "original operator roles"
    )
    require_fields(
        document["compiler"],
        {"root", "inventory", "sha256", "python"},
        "selected compiler",
    )
    require_fields(
        document["compiler"]["python"],
        {"launcher", "binary", "venv_config"},
        "compiler Python",
    )
    n = document["source"]["ordinal"]
    if type(n) is not int or n < 0:
        raise ValueError("source ordinal must be an explicit nonnegative integer")
    return document, intake


def compiler_python(declaration, intake):
    selected = declaration["compiler"]["python"]
    launcher = Path(selected["launcher"])
    if (
        not launcher.is_absolute()
        or str(launcher) != shutil.which("python3")
        or launcher.resolve() != Path(selected["binary"]["path"])
    ):
        raise ValueError("manifest python3 is not the explicitly selected interpreter")
    intake.read(selected["binary"], maximum=64 << 20)
    if Path(selected["venv_config"]["path"]) != launcher.parent.parent / "pyvenv.cfg":
        raise ValueError("compiler Python lost its selected environment configuration")
    intake.read(selected["venv_config"])


def native_loader_python_preflight():
    """Check the original frozen-loader interpreter without executing it.

    Keep the loader's exact environment/.env/default precedence and venv spelling.
    Availability is neither capture/runtime qualification nor a package probe;
    the original loader and source owners still perform their complete checks.
    """
    from merlin.targetgen.capsule_source import _m2m_python

    selected = _m2m_python()
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise ValueError(
            "selected frozen-loader Python is missing or not executable: "
            f"{selected}; configure the original MERLIN_M2M_PYTHON / "
            "MERLIN_M2M_VENV selection or provide its valid default"
        )
    return {
        "selection_owner": "merlin.targetgen.capsule_source._m2m_python",
        "python": str(selected),
        "availability": "file_and_executable",
        "runtime_qualification": "NONE",
        "framework_or_loader_execution": False,
    }


def reopen_plan(declaration, intake):
    from merlin_experiments.phase0 import development_admission_plan as P

    saved = intake.document(declaration["job_plan"])
    if saved.get(
        "schema"
    ) != "merlin.phase0.development_admission_job_plan.v1" or not saved.get("jobs"):
        raise ValueError("original complete development job plan is required")
    limits = saved["jobs"][0]["limits"]
    actual = P.plan(
        Path(declaration["readiness"]["path"]),
        sha256=declaration["readiness"]["sha256"],
        limits=limits,
    )
    if actual != saved:
        raise ValueError(
            "development plan no longer matches its complete original input owners"
        )
    rows = [
        r
        for r in actual["members"]
        if r["identity"] == declaration["source"]["identity"]
    ]
    if len(rows) != 1:
        raise ValueError("selected source is outside the original complete roster")
    return rows[0]


def select_abi(command_buffer, inputs, execution_contract):
    """Select original execution owners from emitted ABI; never modify its program."""
    requested = inputs.get("source_binding_protocol", "whole_program")
    if type(requested) is not str or requested not in {
        "whole_program",
        "merlin.native_pointer_source.v1",
        "gemmini.pointer_source_contractions.v1",
    }:
        raise ValueError("unknown requested source binding protocol")
    chosen = json.loads(json.dumps(inputs))
    chosen.pop("source_binding_protocol", None)
    chosen.pop("execution_contract", None)
    abi = command_buffer.get("kernel_abi")
    if type(abi) is dict and abi.get("kind") == "whole_program":
        if requested == "gemmini.pointer_source_contractions.v1":
            raise ValueError(
                "selected pointer contractions protocol requires the emitted pointer ABI"
            )
        chosen["source_binding_protocol"] = "whole_program"
        return "whole_program", chosen
    pointer = command_buffer.get("compiler_pointer_abi")
    if (
        abi is None
        and type(pointer) is dict
        and type(pointer.get("version")) is int
        and pointer["version"] == 1
    ):
        protocol = (
            "gemmini.pointer_source_contractions.v1"
            if inputs.get("source_binding_protocol")
            == "gemmini.pointer_source_contractions.v1"
            else "merlin.native_pointer_source.v1"
        )
        chosen.update(
            source_binding_protocol=protocol,
            execution_contract=execution_contract,
        )
        return protocol, chosen
    raise ValueError("emitted ABI has no original checked source execution route")


def copy_compiler(declaration, intake, destination):
    root = Path(declaration["compiler"]["root"])
    if (
        not root.is_absolute()
        or root.resolve() != root
        or any(p.is_symlink() for p in (root, *root.parents))
    ):
        raise ValueError("compiler package root must be canonical")
    inventory = intake.document(declaration["compiler"]["inventory"])
    files = inventory.get("files")
    if type(files) is not list or not files:
        raise ValueError("original complete compiler inventory is required")
    actual = []
    for n, path in enumerate(root.rglob("*")):
        if n >= 256 or path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("compiler namespace exceeds original ordinary file bounds")
        if path.is_file():
            actual.append(path.relative_to(root).as_posix())
    actual.sort()
    if actual != sorted(row["path"] for row in files):
        raise ValueError(
            "compiler package differs from its complete original inventory"
        )
    digest = hashlib.sha256()
    payload = []
    for row in sorted(files, key=lambda r: r["path"]):
        relative = Path(row["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != row["path"]
        ):
            raise ValueError("compiler inventory leaf escapes selected package")
        raw = intake.read(
            {
                "path": str(root / relative),
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
        )
        digest.update(row["path"].encode() + b"\0" + raw + b"\0")
        payload.append((relative, raw))
    if digest.hexdigest() != declaration["compiler"]["sha256"]:
        raise ValueError("complete original compiler identity changed")
    if destination is not None:
        destination.mkdir()
        for relative, raw in payload:
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    return digest.hexdigest()
