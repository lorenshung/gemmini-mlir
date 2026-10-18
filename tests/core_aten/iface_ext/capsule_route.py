"""Put the extension-opcode buffers through Merlin's capsule route: ``capsule_common.lower_interface``.

That is the function capsule grading and the model-group route both call: it runs the package's own
manifest entrypoints (parse, lower_interface_to_target, emit_command_buffer, emit_target_artifact),
validates the buffer against Merlin's JSON schema and the interface dtypes, and refuses unfenced reads.
Merlin's sandboxed runner needs the external ``aet`` package, so the entrypoints are run directly here
with the manifest's own argv.

    python capsule_route.py WORKDIR [--rep]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "device"))

import ext_cases as C  # noqa: E402
import run_ext as R  # noqa: E402


def package(work: Path) -> Path:
    pkg = work / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    for f in (REPO / ".merlin").iterdir():
        if f.is_file():
            shutil.copy(f, pkg / f.name)
    link = pkg / "gemmini-opt"
    if not link.exists():
        link.symlink_to(REPO / "gemmini-opt")
    return pkg


def invoke(pkg, command, inp, out=None, *, timeout):
    if command not in pkg.manifest["commands"] and command == "emit_target_artifact":
        command = "lower_target_to_llvm"          # the resolver's legacy alias
    argv = [a.replace("{tool}", str(pkg.tool)).replace("{input_mlir}", str(inp)).replace("{output_json}", str(out))
            for a in pkg.manifest["commands"][command]["argv"]]
    return subprocess.run([sys.executable] + argv, capture_output=True, text=True, timeout=timeout)


def main() -> int:
    from merlin.targetgen.capsule_common import lower_interface
    from merlin.targetgen.package_runtime import BackendDeclined, load_package

    work = Path(sys.argv[1])
    pkg = load_package(package(work))
    rows = {}
    for name, cb in C.cases("--rep" in sys.argv).items():
        d = work / "cap" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "interface.mlir").write_text(R.render(cb))
        try:
            buf, art = lower_interface(pkg, d / "interface.mlir", d / "gen", contract=None, timeout=600, invoke=invoke)
            rows[name] = dict(status="ok", commands=[c["opcode"] for c in buf["commands"]], artifact_bytes=len(art))
        except BackendDeclined as exc:
            rows[name] = dict(status="declined", reason=str(exc)[:300])
        except Exception as exc:  # noqa: BLE001  (a CertFailure's category import needs `aet`)
            cause = exc.__context__ if isinstance(exc, ModuleNotFoundError) and exc.__context__ else exc
            rows[name] = dict(status="refused", reason=f"{type(cause).__name__}: {cause}"[:300])
        print(f"{name:28s} {rows[name]['status']} {rows[name].get('reason', '')}", flush=True)
    (work / "capsule_route.json").write_text(json.dumps(rows, indent=1) + "\n")
    return 0 if all(r["status"] == "ok" for r in rows.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
