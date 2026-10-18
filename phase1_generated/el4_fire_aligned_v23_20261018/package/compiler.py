#!/usr/bin/env python3
"""Gemmini out-of-tree compiler driver."""
import argparse
import json
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-diagnostics', action='store_true')
    ap.add_argument('--convert-iface-to-gemmini', action='store_true')
    ap.add_argument('--emit-target-artifact', action='store_true')
    ap.add_argument('--emit-command-buffer')
    ap.add_argument('input_mlir')
    ns = ap.parse_args()
    from frontend import parse
    text = Path(ns.input_mlir).read_text()
    mixed_source=None
    task_functions=None
    if 'merlin_iface.' in text:
        cb = parse(text)
    else:
        from linalg_frontend import read_module
        if ns.verify_diagnostics:
            read_module(text)
            return
        from mixed_codegen import emit_mixed
        cb,mixed_source,task_functions=emit_mixed(text)
    if any('requant' in c.get('attributes',{}).get('epilogue',[]) for c in cb['commands']):
        cb['commands']=[]
        cb['declined']={'op':'requant','reason':'selected readout supports acc_scale, relu and maxpool; no declared route performs integer requant'}
    if mixed_source is not None and ns.emit_command_buffer and not ns.emit_target_artifact:
        from emitter import compile_source
        compile_source(mixed_source,task_functions,cb['params']['global_program_plan'])
    if ns.emit_target_artifact:
        from emitter import emit,compile_source
        if cb.get('declined'):
            print('module {}')
        else:
            print(compile_source(mixed_source,task_functions,cb["params"]["global_program_plan"]) if mixed_source is not None else emit(cb))
    elif not ns.verify_diagnostics and not ns.emit_command_buffer:
        print('module {')
        for i, cmd in enumerate(cb['commands']):
            payload = json.dumps(json.dumps(cmd))
            print(f'  "gemmini.command"() {{index = {i} : i64, payload = {payload}}} : () -> ()')
        print('}')
    if ns.emit_command_buffer:
        Path(ns.emit_command_buffer).write_text(json.dumps(cb, indent=2))
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'gemmini compiler: {e}', file=sys.stderr)
        sys.exit(1)
