#!/usr/bin/env python3
"""Vendor the canonical model2MLIR quant schema and its exact revision receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--revision',default='HEAD')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    revision=subprocess.check_output(['git','-C',str(a.source),'rev-parse',a.revision],text=True).strip()
    path='m2m/ir/quant_schema.json'
    data=subprocess.check_output(['git','-C',str(a.source),'show',revision+':'+path])
    schema=json.loads(data)
    assert schema['schema_version']>0 and schema['unknown_keys']=='reject'
    root=Path(__file__).resolve().parents[1]
    receipt=dict(repository='model2mlir',revision=revision,path=path,sha256=hashlib.sha256(data).hexdigest())
    if a.check:
        assert (root/'quant_schema.json').read_bytes()==data,'vendored quant schema drift'
        assert json.loads((root/'quant_schema.lock.json').read_text())==receipt,'quant schema pin drift'
    else:
        (root/'quant_schema.json').write_bytes(data)
        (root/'quant_schema.lock.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__': main()
