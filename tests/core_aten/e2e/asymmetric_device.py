"""Real centered-i32 captures: byte-plane products on Spike, exact source readout."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from xdsl.dialects.builtin import ModuleOp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'device'))
from asymmetric_capture import KINDS, SCHEMES, capture, check
import runner
from mlir_oot.core_aten.emit import compile_to_object
from mlir_oot.golden_device_catalog import final_elf_audit


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    combined,symbols = ModuleOp([]),set()
    buffers,calls,checks,outputs,records = {},[],[],set(),[]
    cases = [(kind,scheme,False) for kind in KINDS for scheme in SCHEMES]
    cases += [('mm',scheme,True) for scheme in ('asym8','uint8')]
    for kind,scheme,wide in cases:
        source,inputs,expected = capture(kind,scheme,wide)
        label = kind+'_'+scheme+('_wide' if wide else '')
        (out/(label+'.mlir')).write_text(source)
        device,manifest,replays = check(source,inputs,expected)
        (out/(label+'.json')).write_text(json.dumps(manifest,indent=2))
        for index,(match,feed,raw) in enumerate(replays):
            bindings = [b for b in manifest['bindings'] if b['numeric'].get('integer_readout')]
            binding = bindings[index]
            argv=[]
            for arg in match.kernel.args:
                name=f'{label}_{index}_{arg.name}';argv.append(name)
                buffers[name]=feed.get(arg.name,np.zeros(arg.shape,np.int32 if arg.dtype=='i32' else np.int8))
                if arg.role=='output':
                    outputs.add(name);checks.append(runner.Check(name,raw,0,label+f'_{index}'))
            calls.append((binding['symbol'],argv))
        for op in list(device.body.block.ops):
            if op.sym_name.data not in symbols:
                symbols.add(op.sym_name.data);op.detach();combined.body.block.add_op(op)
        records.append(dict(kind=kind,scheme=scheme,wide=wide,producers=len(replays),fraction_readout=True))
        print(label,len(replays),flush=True)
    receipt = compile_to_object(combined,out/'build')
    test = runner.DeviceTest('asymmetric',[Path(receipt['object'])],buffers,calls,checks,outputs)
    elf = runner.build_elf(test,out/'build');final_elf_audit(elf)
    result=runner.run_spike(elf)
    (out/'spike.log').write_text(result.log)
    (out/'results.json').write_text(json.dumps(dict(elf=str(elf.resolve()),cases=records,
        spike=dict(status=result.status,passed=len(result.passed),failed=result.failed)),indent=2))
    print(result.status,len(result.passed),result.failed)
    return result.ok


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    raise SystemExit(0 if run(p.parse_args().out) else 1)
