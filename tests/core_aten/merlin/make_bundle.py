"""Build one Merlin Core ATen bundle (inputs, semantic I/O contract) from Merlin's fp32 captures.

usage (Merlin's interpreter, live Merlin on PYTHONPATH):
    python make_bundle.py CORPUS CAPTURES OUTDIR OVERLOAD
"""
import sys
from pathlib import Path
import json

from merlin.targetgen.core_aten_batch import build_core_aten_batch

corpus, captures, out, overload = sys.argv[1:5]
shard = build_core_aten_batch(json.loads(Path(corpus).read_text()), Path(captures), Path(out),
                              selected_overloads={overload})
if shard["bundled_count"] != 1:
    raise SystemExit(f"{overload}: not bundled: {[c.get('reason') for c in shard['cases']]}")
print(out)
