"""Fresh f32 movement captures using Merlin's selected int8 recipe and model2MLIR.

Run with the model2MLIR interpreter. The recipe is supplied by the caller;
no quantizer or host representation algorithm is duplicated in the provider.
"""
import argparse
import json
from pathlib import Path

import m2m
import numpy as np
import torch

from capture_rtl import rtl_set
from quantized_movement import MOVEMENTS
from merlin.targetgen._recipe_quantizer import apply_recipe
from merlin.capture.integerization import integerize_static_pt2e


def capture(recipe, out):
    selected = {name: (prototype.fn, tensors) for name, prototype, tensors in rtl_set()
                if name in MOVEMENTS or name in ('constant_pad_nd_p1', 'fill_Scalar')}
    selected['constant_pad_nd'] = selected.pop('constant_pad_nd_p1')
    selected['fill'] = selected.pop('fill_Scalar')
    selected['_to_copy'] = (lambda x: torch.ops.aten._to_copy.default(x, dtype=torch.float64),
                            [torch.empty(3, 19)])
    generator = torch.Generator().manual_seed(88157)
    summary = []
    for name in MOVEMENTS:
        fn, templates = selected[name]
        inputs = tuple(torch.randn(t.shape, generator=generator, dtype=torch.float32) for t in templates)
        class Model(torch.nn.Module):
            def forward(self, *xs):
                return fn(*xs)
        quantized = apply_recipe(Model().eval(), recipe, example_inputs=inputs)
        integerized, receipt = integerize_static_pt2e(quantized, inputs, numerics='integer_readout')
        result = m2m.convert(integerized, inputs, backend='fx_importer')
        assert result.ok, (name, result.diagnostics)
        wd = out / 'part0/shards' / ('aten__' + name + '__' + ('Scalar' if name == 'fill' else 'default')) / 'capture-bundle'
        wd.mkdir(parents=True, exist_ok=True)
        (wd / 'model.mlir').write_text(result.mlir_text)
        np.savez(wd / 'inputs.npz', **{f'in{i}': t.numpy() for i,t in enumerate(inputs)})
        summary.append(dict(name=name, input_shapes=[list(t.shape) for t in inputs],
                            integerization_receipt=receipt))
        print(name, 'captured', flush=True)
    (out / 'capture_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recipe',type=Path)
    ap.add_argument('out',type=Path)
    args = ap.parse_args()
    capture(json.loads(args.recipe.read_text()),args.out)
