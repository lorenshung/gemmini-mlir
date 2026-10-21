"""Shared Core ATen test setup: helper-script imports and the execution provider under test."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
for helpers in ('device', 'e2e', 'iface_ext'):
    sys.path.insert(0, str(HERE / helpers))

POLICY = dict(numerical_semantics=dict(accumulator_dtype='i32', internal_arithmetic=dict(
    signed_operand_bits=8, mac_result_bits=20, mac_result_overflow='wrap_to_20_bits',
    full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))


def provider():
    """A fresh load of merlin-support/core_aten/execution_provider.py."""
    path = PACKAGE / 'merlin-support/core_aten/execution_provider.py'
    spec = importlib.util.spec_from_file_location('core_aten_execution_provider', path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api
