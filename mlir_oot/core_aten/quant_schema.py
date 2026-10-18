"""Consumer of model2MLIR's vendored prov.quant schema; no local key allowlists."""
from fractions import Fraction
import json
from pathlib import Path
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr
from .exactness import require

SCHEMA = json.loads((Path(__file__).resolve().parents[2]/'merlin-support/quant_schema.json').read_text())


def attributes(op, kind):
    result = {k:v for k,v in op.attributes.items() if k.startswith(SCHEMA['namespace'])}
    for key,value in result.items():
        spec = SCHEMA['attributes'].get(key)
        require(spec is not None and kind in spec['op_kinds'], 'unknown prov.quant key for '+kind+': '+key)
        typ = spec['type']
        if 'array<' in typ and isinstance(value,ArrayAttr):
            if 'decimal_integer_string' in typ:
                require(all(isinstance(v,StringAttr) and v.data.lstrip('-').isdigit() for v in value), 'invalid rational array '+key)
            else:
                require(all(isinstance(v,FloatAttr) and ('array<'+str(v.type)+'>') in typ for v in value), 'invalid float array '+key)
        elif typ.startswith('string'):
            require(isinstance(value,StringAttr), 'invalid string '+key)
            if typ.startswith('string('): require(value.data in ('true','false'),'invalid boolean string '+key)
        elif typ == 'i64':
            require(isinstance(value,IntegerAttr) and str(value.type)=='i64', 'invalid integer '+key)
        elif typ.startswith('f'):
            require(isinstance(value,FloatAttr) and str(value.type) in typ.split(' | '), 'invalid float '+key)
        else:
            require(False,'invalid attribute type '+key)
    return {k.removeprefix(SCHEMA['namespace']):v for k,v in result.items()}


def ratios(attrs):
    numerators,denominators = attrs.get('scale_numerator'),attrs.get('scale_denominator')
    require(numerators is not None and denominators is not None, 'missing exact rational scale pair')
    require(len(numerators)==len(denominators) and len(numerators)>0, 'invalid rational scale pair lengths')
    result=[]
    for n,d in zip(numerators,denominators):
        require(int(d.data)>0,'nonpositive scale denominator')
        result.append(Fraction(int(n.data),int(d.data)))
    return result
