#!/usr/bin/env python3
"""Drive Merlin's own spike_model build/run with the provider's region offload on one bundle dir."""
import json, os, sys
from pathlib import Path
import numpy as np
from merlin.runtime.backends import spike_model
from merlin.targetgen.core_aten_device import load_execution_provider, region_routing_for_bundle, executed_device_instructions
d = Path(sys.argv[1])
prov = load_execution_provider("gemmini", Path(os.environ["MERLIN_CORE_ATEN_EXECUTION_PROVIDER"]))
routing = region_routing_for_bundle("gemmini", Path(os.environ["GMLIR"]), prov.core_aten_offload)
build = spike_model.build(d, d / "spike-build", inputs_npz=d / "inputs.npz", arena_mb=64, dump_all_outputs=True,
                          backend="scalar", device=routing)
run = spike_model.run(build["elf"], mem_bytes=build["mem_bytes"], timeout=900, vlen=build.get("vlen"),
                      trace_path=d / "trace.log", **prov.runner_options())
ev = executed_device_instructions(d / "trace.log", Path(build["elf"]), "gemmini")
n_exp = np.load(d / "eager_q.npy").size
exp_dtype = np.load(d / "eager_q.npy").dtype
out = np.concatenate([np.frombuffer(b, exp_dtype) for b in run["output_bytes"]])[:n_exp]
exp = np.load(d / "eager_q.npy").reshape(-1)
meta = json.loads((d / "eager_meta.json").read_text()) if (d / "eager_meta.json").exists() else None
per = []
if meta:
    o = 0
    for so, n in zip(meta["s_out"], meta["sizes"]):
        e = np.rint(np.abs(out[o:o + n] - exp[o:o + n]) / so)
        per.append(dict(max_lsb=int(e.max()), exact=round(float((e == 0).mean()), 5)))
        o += n
print(json.dumps(dict(executed_device_instructions=ev["executed_instructions"], match=bool(np.array_equal(out, exp)),
                      maxdiff=float(np.abs(out - exp).max()), outputs=per)))
