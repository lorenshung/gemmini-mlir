"""Read archived bodies without claiming their obsolete quant provenance schema.

The original MLIR files and eager outputs remain immutable. Only trace attributes
from the pre-schema captures are removed; operation operands, properties, types,
bodies and ATen provenance are retained. Catalog bindings hash the resulting
source bytes. Real model2MLIR captures are passed through unchanged.
"""

from pathlib import Path


def source_text(text):
    if "prov.quant.qmin" not in text and "prov.quant.qmax" not in text:
        return text
    from mlir_oot.frontend.parse import parse_module

    module = parse_module(text)
    for op in module.walk():
        for key in list(op.attributes):
            if key.startswith("prov.quant."):
                del op.attributes[key]
    return str(module)


def read(path):
    return source_text(Path(path).read_text())
