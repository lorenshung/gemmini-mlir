"""Run every captured case as its own ELF, in parallel, and aggregate a per-op status table.

    python run_all.py CAPTURE_DIR --engine spike|rtl [-j 8] [--only name ...] [--out summary.json]

Each case runs ``run_e2e.py --only <case>``; the summary records, per case, the bound Core ATen
ops, the engine verdict, the model-vs-eager error and the device cycle count.
"""

from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def one(case: Path, engine: str, tag: str) -> dict:
    out = Path(os.environ.get("GEMMINI_TEST_WORK", ".")) / f"{tag}_{case.name}.json"
    argv = [sys.executable, str(HERE / "run_e2e.py"), str(case.parent), "--only", case.name,
            "--batch", f"{tag}_{case.name}", "--json", str(out)]
    if engine == "rtl":
        argv += ["--rtl", "--no-spike"]
    p = subprocess.run(argv, capture_output=True, text=True)
    if not out.exists():
        return dict(case=case.name, status="ERROR", error=(p.stdout + p.stderr)[-1500:])
    res = json.loads(out.read_text())
    out.unlink()
    eng = res.get(engine, {})
    cs = res["cases"][case.name]
    if cs.get("status") == "REFUSED":
        return dict(case=case.name, status="REFUSED", reason=cs["reason"], ops=[])
    return dict(case=case.name, ops=sorted(set(cs["ops"])), status=eng.get("status", "ERROR"),
                failed=eng.get("failed"), info=eng.get("info"), cycles=next(iter(eng.get("perf", {}).values()), None),
                seconds=eng.get("seconds"), model_vs_eager_max=cs["model_vs_eager_max"],
                model_vs_eager_exact=cs["model_vs_eager_exact"], instructions=cs["instructions"],
                fences=cs["fences"], tail=eng.get("tail"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--engine", choices=("spike", "rtl"), default="spike")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    cases = sorted(p for p in a.capture.iterdir() if (p / "model.mlir").exists())
    if a.only:
        cases = [c for c in cases if c.name in a.only]
    # Results are appended as each case completes (and a rerun resumes from them): an RTL run lasts
    # hours, and one case raising must not discard the others.
    log = a.out.with_suffix(".jsonl") if a.out else None
    done = {}
    if log and log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") in ("PASS", "FAIL", "REFUSED"):
                done[r["case"]] = r
    lock = threading.Lock()

    def guarded(c):
        if c.name in done:
            return done[c.name]
        try:
            r = one(c, a.engine, f"{a.tag}_{a.engine}")
        except Exception as exc:  # noqa: BLE001
            r = dict(case=c.name, status="ERROR", error=f"{type(exc).__name__}: {exc}"[-1500:])
        if log:
            with lock, log.open("a") as f:
                f.write(json.dumps(r) + "\n")
        print(f"{r['case']:32s} {r['status']}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=a.j) as pool:
        rows = list(pool.map(guarded, cases))
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1) + "\n")
    for r in rows:
        print(f"{r['case']:32s} {r['status']:6s} cycles={r.get('cycles')} model_vs_eager max={r.get('model_vs_eager_max')}"
              f" exact={r.get('model_vs_eager_exact')}")
    return 0 if all(r["status"] in ("PASS", "REFUSED") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
