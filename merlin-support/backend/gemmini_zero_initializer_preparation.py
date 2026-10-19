"""Connect closed source seed groups to the preserved V23 direct emitter.

This proposes edits to an owned candidate, never the Phase 1 package. Generic
source recognition lives in Merlin and is copied byte-for-byte into an approved
candidate pass component. Original source/ELF/packet/full-output checks remain
mandatory. Availability here is no target qualification or timing claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("initializer fusion lost the exact original generated compiler boundary")
    return text.replace(old, new)


def rewrite_mixed_codegen(text):
    """Keep all original source operations while virtualizing a closed zero seed."""
    text = _replace_once(
        text,
        "    _, fn = read_module(text)",
        "    source_module, fn = read_module(text)",
    )
    text = _replace_once(
        text,
        "    returned=list(block.ops.last.operands)",
        """    from zero_initializer_policy import choose_zero_initializer_groups
    from pathlib import Path
    import json
    public = json.loads(Path(__file__).with_name('staged-tile-public.json').read_bytes())
    contract = public.get('contract', {})
    supported = (public.get('schema') == 'gemmini.staged_tile_public.v1'
                 and contract.get('readout') == 'raw_accumulator_identity'
                 and contract.get('operand_bits') == 8
                 and contract.get('accumulator_bits') == 32
                 and contract.get('external_buffers_disjoint') is True
                 and contract.get('fence_completion') == 'data_commands_preserve_pending_descriptor')
    init_groups = choose_zero_initializer_groups(source_module) if supported else ()
    grouped = {group.contraction_ordinal: group for group in init_groups}
    absorbed = {index for group in init_groups for index in group.source_op_indices
                if index != group.contraction_ordinal}
    virtual = {operations[index].results[result] for group in init_groups
               for index, result in group.virtual_result_positions}
    returned=list(block.ops.last.operands)""",
    )
    text = _replace_once(
        text,
        "        for j,value in enumerate(op.results):\n            if isinstance(value.type,TensorType):",
        (
            "        for j,value in enumerate(op.results):\n"
            "            if value in virtual: continue\n"
            "            if isinstance(value.type,TensorType):"
        ),
    )
    text = _replace_once(
        text,
        "    for i,op in enumerate(operations):\n        # Include captures",
        ("    for i,op in enumerate(operations):\n        if i in absorbed: continue\n        # Include captures"),
    )
    text = _replace_once(
        text,
        "        needed=list(dict.fromkeys(op.operands))",
        "        needed=list(dict.fromkeys(value for value in op.operands if value not in virtual))",
    )
    text = _replace_once(
        text,
        "        operands=[local_ptrs[v] for v in op.operands]",
        "        operands=[local_ptrs.get(v, '((void*)0)') for v in op.operands]",
    )
    text = _replace_once(
        text,
        "        geometry=_contraction(op)\n        if geometry:",
        (
            "        geometry=_contraction(op)\n"
            "        if i in grouped and not geometry:\n"
            "            raise ValueError('closed initializer lost original d"
            "irect contraction lowering')\n"
            "        if geometry:"
        ),
    )
    text = _replace_once(
        text,
        "        func=f'task_{i}'\n        task_functions[i]=func",
        "        task_id=len(tasks)\n        func=f'task_{task_id}'\n        task_functions[task_id]=func",
    )
    text = _replace_once(
        text,
        (
            "        task={'task_index':i,'kind':'device' if device else 'hos"
            "t','source_op_indices':[i],\n"
            "              'instruction_start':i,'instruction_end':i+1,"
        ),
        (
            "        source_indices=list(grouped[i].source_op_indices) if i i"
            "n grouped else [i]\n"
            "        task={'task_index':task_id,'kind':'device' if device els"
            "e 'host','source_op_indices':source_indices,\n"
            "              'instruction_start':task_id,'instruction_end':task"
            "_id+1,"
        ),
    )
    text = _replace_once(
        text,
        (
            "        placements.append({'source_op_index':i,'operation':kind,"
            "'lane':'accelerator' if device else 'host',\n"
            "                           'reason':'integer Gemmini contraction"
            "' if device else 'source layout, initializer, quantization, or f"
            "32 scalar work'})"
        ),
        (
            "        for source_index in source_indices:\n"
            "            placements.append({'source_op_index':source_index,'o"
            "peration':_name(operations[source_index]),\n"
            "                               'lane':'accelerator' if device el"
            "se 'host',\n"
            "                               'reason':'proved zero seed consum"
            "ed by direct accumulator initialization' if source_index != i el"
            "se\n"
            "                                        'integer Gemmini contrac"
            "tion' if device else\n"
            "                                        'source layout, initiali"
            "zer, quantization, or f32 scalar work'})"
        ),
    )
    return text


def rewrite_provenance(text):
    """Use actual source group ordinals instead of relabeling a fused task ID."""
    text = _replace_once(
        text,
        "    from xdsl.dialects.builtin import Builtin, IntegerAttr, StringAttr",
        "    from xdsl.dialects.builtin import Builtin, DenseArrayBase, IntegerAttr, StringAttr, i64",
    )
    text = _replace_once(
        text,
        "            nested.attributes['merlin.source_op_index']=IntegerAttr(task,64)",
        (
            "            indices=plan['tasks'][task]['source_op_indices'] if "
            "plan is not None else [task]\n"
            "            if len(indices)==1:\n"
            "                nested.attributes['merlin.source_op_index']=Inte"
            "gerAttr(indices[0],64)\n"
            "            else:\n"
            "                nested.attributes['merlin.source_op_indices']=De"
            "nseArrayBase.from_list(i64,indices)"
        ),
    )
    text = _replace_once(
        text,
        "                node.attributes.pop('merlin.source_op_index',None)",
        (
            "                node.attributes.pop('merlin.source_op_index',Non"
            "e)\n"
            "                node.attributes.pop('merlin.source_op_indices',N"
            "one)"
        ),
    )
    return text


def prepare_compiler(source, destination, *, enabled, generic_source):
    """Prepare an owned policy experiment, with no new executable capability.

    Both arms must start from the same already-selected staged compiler. The
    generic transformation is candidate-owned source, not an SDK export or an
    independent checker. The returned provenance cannot qualify an execution.
    """
    from merlin.targetgen.compiler_pass_selection import (
        _pin,
        _tree,
        copy_compiler_source,
    )

    if type(enabled) is not bool:
        raise ValueError("zero initializer preparation needs an explicit Boolean policy")
    source, destination, generic_source = map(Path, (source, destination, generic_source))
    parent = _tree(source)
    generic_pin = _pin(generic_source)
    library = {}
    for name in (
        "selected-shared-library.json",
        "shared-library-sources.json",
        "staged-tile-public.json",
    ):
        path = source / name
        _pin(path)
        if path.stat().st_size > 1 << 20:
            raise ValueError("zero initializer selected source contract exceeds its bound")
        library[name] = path.read_bytes()
        json.loads(library[name])
    if (source / "zero_initializer_policy.py").exists() or (source / "passes").exists():
        raise ValueError("zero initializer preparation requires a fresh pass component")
    mixed = rewrite_mixed_codegen((source / "mixed_codegen.py").read_text())
    provenance = rewrite_provenance((source / "provenance.py").read_text())
    generic = generic_source.read_bytes()
    if hashlib.sha256(generic).hexdigest() != generic_pin["sha256"]:
        raise ValueError("generic initializer transformation changed during preparation")
    copy_compiler_source(source, destination)
    (destination / "mixed_codegen.py").write_text(mixed)
    (destination / "provenance.py").write_text(provenance)
    (destination / "passes").mkdir()
    (destination / "passes/__init__.py").write_text("")
    (destination / "passes/zero_initializer_groups.py").write_bytes(generic)
    (destination / "zero_initializer_policy.py").write_text(
        f"ENABLED = {enabled!r}\n\n"
        "def choose_zero_initializer_groups(module):\n"
        "    if not ENABLED:\n"
        "        return ()\n"
        "    from passes.zero_initializer_groups import find_zero_initializer_groups\n"
        "    return find_zero_initializer_groups(module)\n"
    )
    if (
        _tree(source) != parent
        or _pin(generic_source) != generic_pin
        or any((destination / name).read_bytes() != data for name, data in library.items())
    ):
        raise ValueError("zero initializer preparation changed its immutable source selection")
    return {
        "schema": "gemmini.zero_initializer_preparation.v1",
        "source": {"path": str(source), "sha256": parent},
        "prepared": {"path": str(destination), "sha256": _tree(destination)},
        "implementation": _pin(Path(__file__).resolve()),
        "candidate_owned_generic_transformation": generic_pin,
        "selected_library": json.loads(library["selected-shared-library.json"]),
        "enabled": enabled,
        "authority": "candidate source preparation only; independent realization required",
        "hardware_execution_qualified": False,
        "timing_authority": "NONE",
    }
