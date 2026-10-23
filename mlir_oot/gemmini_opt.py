"""Host-facing compiler CLI, including optional golden export orchestration.

The submitted compiler is compiler_cli: the same Pipeline and four public
entrypoints, without host-owned export/build orchestration.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from mlir_oot.compiler_cli import Pipeline, _ExtensionPlan, _print
from mlir_oot.compiler_cli import run_interface

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gemmini-opt", add_help=True)
    ap.add_argument("--verify-diagnostics", action="store_true")
    ap.add_argument("--convert-iface-to-gemmini", action="store_true")
    ap.add_argument("--emit-target-artifact", action="store_true")
    ap.add_argument("--emit-command-buffer", default=None, metavar="PATH")
    ap.add_argument("--export-golden-contraction", action="store_true")
    ap.add_argument("--optimize-golden-contraction", action="store_true")
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--export-golden-capture", action="store_true")
    ap.add_argument("--emit-golden-inventory", metavar="PATH")
    ap.add_argument("--region")
    ap.add_argument("--llvm-bin", type=Path)
    ap.add_argument("--workdir", type=Path)
    ap.add_argument("--large-n", action="store_true")
    ap.add_argument("--prefetch-b", action="store_true")
    ap.add_argument("--dense-input-policy", choices=("banked_command_cost","resident_a_command_cost","transfer_command_cost","resident_a_prefetch"))
    ap.add_argument("--dense-b-slot-policy", choices=("remaining_rows",))
    ap.add_argument("--flat-spatial", action="store_true")
    ap.add_argument("--virtual-padding", action="store_true")
    ap.add_argument("--exact-integer-readout", action="store_true")
    ap.add_argument("--banked-prefetch", action="store_true")
    ap.add_argument("--grouped-b", action="store_true")
    ap.add_argument("--separate-b-bank", action="store_true")
    ap.add_argument("--resident-input-policy", choices=("compact_channel_planes","compact_channel_planes_prefetch_b"))
    ap.add_argument("--resident-stripes", action="store_true")
    ap.add_argument("--source-stride-resident", action="store_true")
    ap.add_argument("--source-stride-row-residue", action="store_true")
    ap.add_argument("--resident-a-load-coalescing", action="store_true")
    ap.add_argument("--spatial-command-loops", action="store_true")
    ap.add_argument("--dense-accumulator-stripes", action="store_true")
    ap.add_argument("--dense-cached-b-capacity", action="store_true")
    ap.add_argument("--resident-weight-issue-tiles", type=int, choices=(1,2,3,4))
    ap.add_argument("--flat-resident-planes", action="store_true")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("input", nargs="?", default="-")
    args = ap.parse_args(argv)

    exports=(args.export_golden_contraction,args.optimize_golden_contraction,
             args.export_golden_capture,bool(args.emit_golden_inventory))
    if sum(exports)>1:
        ap.error('select one golden export command')
    compilation_options=(args.region,args.llvm_bin,args.workdir,args.large_n,args.prefetch_b,
                         args.dense_input_policy,args.dense_b_slot_policy,args.calibration)
    capture_options=(args.flat_spatial,args.virtual_padding,args.exact_integer_readout,
                     args.banked_prefetch,args.grouped_b,args.separate_b_bank,
                     args.resident_input_policy,args.resident_stripes,args.source_stride_resident,args.source_stride_row_residue,args.resident_a_load_coalescing,args.spatial_command_loops,args.dense_accumulator_stripes,args.dense_cached_b_capacity,args.resident_weight_issue_tiles,args.flat_resident_planes)
    if not any(exports) and any((*compilation_options,*capture_options)):
        ap.error('golden options require an explicit golden export command')
    if any(exports) and any((args.verify_diagnostics,args.convert_iface_to_gemmini,
                             args.emit_target_artifact,args.emit_command_buffer,args.output)):
        ap.error('golden export commands cannot mix with interface pipeline options')
    if args.emit_golden_inventory and (args.input!='-' or any((*compilation_options,*capture_options))):
        ap.error('inventory export accepts no compilation options or input')
    if (args.export_golden_contraction or args.optimize_golden_contraction) and any(capture_options):
        ap.error('capture schedule options require --export-golden-capture')
    if args.calibration is not None and not args.optimize_golden_contraction:
        ap.error('calibration requires --optimize-golden-contraction')
    if args.optimize_golden_contraction and any((args.large_n,args.prefetch_b,
                                                args.dense_input_policy,args.dense_b_slot_policy)):
        ap.error('optimized alternatives derive options from their pinned exports')
    if args.export_golden_capture and any((args.region,args.large_n,args.prefetch_b)):
        ap.error('contraction identity and schedule options require --export-golden-contraction')
    if any(exports):
        import json
        from .golden_compiler_export import export_inventory,export_contraction,export_capture
        try:
            if args.emit_golden_inventory:
                result=export_inventory(_HERE.parent)
                Path(args.emit_golden_inventory).write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(dict(surfaces=len(result['package_inventory']['surfaces']),
                                      inventory=args.emit_golden_inventory)))
            else:
                if args.input=='-' or args.llvm_bin is None or args.workdir is None:
                    raise ValueError('golden compilation requires an input file, --llvm-bin and --workdir')
                if args.export_golden_contraction or args.optimize_golden_contraction:
                    if not args.region:raise ValueError('contraction export requires --region identity')
                    if args.optimize_golden_contraction:
                        from .golden_calibrated_plan import optimize_contraction
                        if args.calibration is None:raise ValueError('optimized contraction requires --calibration')
                        result=optimize_contraction(Path(args.input),args.region,args.llvm_bin,
                                                   args.workdir,args.calibration)
                    else:
                        result=export_contraction(Path(args.input),args.region,args.llvm_bin,args.workdir,
                            large_n=args.large_n,prefetch_b=args.prefetch_b,
                            dense_input_policy=args.dense_input_policy,dense_b_slot_policy=args.dense_b_slot_policy)
                    print(json.dumps(dict(kernel_symbol=result['kernel_symbol'],
                        object_sha256=result['compilation']['object_sha256'],
                        selected_plan_controls_emitted_code=result['selected_plan_controls_emitted_code'],
                        receipt=str(args.workdir/('golden_optimized_plan.json'
                            if args.optimize_golden_contraction else 'golden_export.json')))))
                else:
                    result=export_capture(Path(args.input),args.llvm_bin,args.workdir,
                        flat_spatial=args.flat_spatial,virtual_padding=args.virtual_padding,
                        exact_integer_readout=args.exact_integer_readout,
                        banked_prefetch=args.banked_prefetch,grouped_b=args.grouped_b,
                        separate_b_bank=args.separate_b_bank,resident_input_policy=args.resident_input_policy,
                        resident_stripes=args.resident_stripes,source_stride_resident=args.source_stride_resident,source_stride_row_residue=args.source_stride_row_residue,dense_input_policy=args.dense_input_policy,
                        dense_b_slot_policy=args.dense_b_slot_policy,resident_a_load_coalescing=args.resident_a_load_coalescing,
                        spatial_command_loops=args.spatial_command_loops,dense_accumulator_stripes=args.dense_accumulator_stripes,dense_cached_b_capacity=args.dense_cached_b_capacity,resident_weight_issue_tiles=args.resident_weight_issue_tiles,flat_resident_planes=args.flat_resident_planes)
                    print(json.dumps(dict(routes=len(result['bundle']['routes']),
                        object_sha256=result['bundle']['object_sha256'],
                        shared_solver_selected=False,receipt=str(args.workdir/'requant.json'))))
            return 0
        except Exception as exc:
            print(f'golden export: {type(exc).__name__}: {exc}',file=sys.stderr)
            return 1

    return run_interface(args)


if __name__ == "__main__":
    raise SystemExit(main())
