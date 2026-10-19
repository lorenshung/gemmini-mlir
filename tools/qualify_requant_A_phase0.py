"""Rebuild frozen Phase 0 prepared shards; grade new Spike outputs exactly.

No recapture, routing expansion, numeric-policy changes or simulator substitution.
The frozen capture/reference records remain immutable; outputs live in --out.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def one(source, dest, package):
    from merlin.runtime.backends import spike_model
    from merlin.targetgen.core_aten_device import load_execution_provider, selected_facts, region_routing_for_bundle, routing_for_bundle
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch
    from merlin.llvmlower.device_build import load_region_offload
    from merlin.llvmlower.device_offload import load_sidecar
    dest.mkdir(parents=True,exist_ok=True)
    report=json.loads((source/'core_aten_batch_map.json').read_text())
    case=next(c for c in report['cases'] if c['status']=='bundled')
    rec=dict(case=case.get('case_id',case['overload']),source=str(source),status='error')
    try:
        for p in source.iterdir():
            if p.is_file() and p.suffix not in ('.log','.txt'):
                shutil.copy2(p,dest/p.name)
        routing=None
        with selected_facts('gemmini',Path(os.environ['MERLIN_RTL_FACTS'])) as facts:
            provider=load_execution_provider('gemmini',package/'merlin-support/core_aten/execution_provider.py')
            lane=case['routing']['lane']
            if lane=='device':
                if 'region offload' in case['routing']['reason']:
                    routing=region_routing_for_bundle('gemmini',package,provider.core_aten_offload)
                else:
                    routing,_=routing_for_bundle(dest,'gemmini',package,provider,facts)
                if routing is None:raise ValueError('original device lane was lost')
            build=spike_model.build(dest,dest/'spike-build',inputs_npz=dest/'inputs.npz',arena_mb=16,
                                    dump_all_outputs=True,backend='scalar',device=routing,dram_bytes=256 << 20)
            run=spike_model.run(build['elf'],mem_bytes=build['mem_bytes'],timeout=900,vlen=build.get('vlen'),**provider.runner_options())
        if run.get('metrics',{}).get('build_hash')!=build.get('build_hash'):raise ValueError('ELF build hash mismatch')
        (dest/'spike-console.txt').write_text(run['console'])
        output=[v.hex() for v in run['output_bytes']]
        expected=json.loads((source/'spike-output-bytes.json').read_text())
        (dest/'spike-output-bytes.json').write_text(json.dumps(output))
        verdict=grade_core_aten_batch(report,run['output_bytes'],output_shapes=run.get('output_shapes') or None,
                                      semantic_readback=run.get('semantic_readback'))
        cv=verdict['cases'][rec['case']]
        sidecar=(load_region_offload(dest/'spike-build') or load_sidecar(dest/'spike-build') or {})
        if routing and not sidecar.get('routed'):raise ValueError('device routing emitted no source-bound calls')
        rec.update(status='pass' if output==expected and cv['status']=='pass' else 'fail',
                   byte_exact_vs_frozen=output==expected,exact_reference_verdict=cv,lane=lane,
                   elf=str(build['elf']),build_hash=build.get('build_hash'),metrics=run.get('metrics'),
                   routed=sidecar.get('routed'),outputs=len(output))
    except Exception as exc:
        rec.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    (dest/'requant-A-result.json').write_text(json.dumps(rec,indent=2,default=str)+'\n')
    return rec


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--one',action='store_true')
    a=ap.parse_args()
    if a.one:
        print(json.dumps(one(a.source,a.out,a.package),default=str));return
    shards=sorted(a.source.glob('part*/shards/*/core_aten_batch_map.json'))
    assert len(shards)==193,len(shards)
    a.out.mkdir(parents=True,exist_ok=True)
    def work(p):
        dest=a.out/p.parent.name
        dest.mkdir(exist_ok=True)
        with (dest/'worker.log').open('w') as log:
            subprocess.run([sys.executable,__file__,'--one','--source',str(p.parent),'--out',str(dest),'--package',str(a.package)],stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads((dest/'requant-A-result.json').read_text())
    results=[]
    with ThreadPoolExecutor(a.workers) as pool:
        for r in pool.map(work,shards):
            results.append(r)
            print(r['case'],r['status'],r.get('error',''),flush=True)
            (a.out/'summary.json').write_text(json.dumps(dict(cases=results,passed=sum(r['status']=='pass' for r in results),total=len(shards)),indent=2,default=str)+'\n')
    if any(r['status']!='pass' for r in results):sys.exit(1)

if __name__=='__main__':main()
