"""Source-relative cleanup controls; synthetic contracts do not qualify hardware."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def adapter():
    path = (
        Path(__file__).resolve().parents[1] / "backend/gemmini_quiescent_load_config.py"
    )
    spec = importlib.util.spec_from_file_location("quiescent_config_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def contract(tmp_path, adapter):
    leaf = tmp_path / "source.txt"
    leaf.write_text("public synthetic admission stimulus")
    pin = adapter._pin(leaf)
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "schema": adapter.SCHEMA,
                "selected_inputs": {
                    "facts": pin,
                    "observed_contract": pin,
                    "operand_sidecar": pin,
                },
                "runtime": pin,
                "sources": [pin],
                "semantics": adapter.SEMANTICS,
                "encoding": {"config_rs1": 1234567, "max_stride": 1023},
            }
        )
    )
    return adapter.LoadConfigurationContract.read(path)


def test_shared_cleanup_changes_real_statement_stream_and_preserves_nonconfig(
    adapter, contract
):
    before = [
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
        "gm_mvin(((const int8_t*)(A))+((size_t)(0)*17+(0)),0,3,16);",
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
        "gm_mvin(B,16,16,9);",
        "gm_fence();",
    ]
    after, report = adapter.optimize_statements(before, contract)
    assert after == before[:4] + before[5:]
    assert report["removed"] == report["configuration_check"]["removed_calls"] == 1
    assert report["target_execution_qualification"] == "not_established"
    assert before[4] == "gm_ld_config(17,0,1.0f,0);"  # original list remains unchanged


@pytest.mark.parametrize(
    "middle",
    [
        "gm_mvin(side_effect(),0,16,16);",
        "gm_mvin(A++,0,16,16);",
        "gm_mvin(A[0],0,16,16);",
        "gm_mvin(A,0,16);",
        "gm_ex_config(1,0);",
        "gm_cmd(8,0,0);",
        "gm_ld_config(17,1,1.0f,0);",
        "external();",
    ],
)
def test_unknown_or_effectful_arguments_are_barriers(adapter, contract, middle):
    before = [
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
        middle,
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
    ]
    assert adapter.optimize_statements(before, contract)[0] == before


@pytest.mark.parametrize(
    "setter",
    [
        "gm_ld_config(18,0,1.0f,0);",
        "gm_ld_config(stride,0,1.0f,0);",
        "gm_ld_config(17,0,0.5f,0);",
        "gm_ld_config(17,0,1.0f,1);",
        "gm_ld_config(1024,0,1.0f,0);",
        "gm_ld_config(17,0,1.0f,0); external();",
    ],
)
def test_dynamic_modes_and_changed_settings_are_never_removed(
    adapter, contract, setter
):
    before = ["gm_fence();", "gm_ld_config(17,0,1.0f,0);", "gm_fence();", setter]
    assert adapter.optimize_statements(before, contract)[0] == before


def test_removal_requires_original_fence_at_each_supported_configuration(
    adapter, contract
):
    before = [
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
        "gm_mvin(A,0,16,16);",
        "gm_ld_config(17,0,1.0f,0);",
    ]
    assert adapter.optimize_statements(before, contract)[0] == before


def test_loops_keep_original_route(adapter, contract):
    before = [
        "for (int i=0;i<4;i++) {",
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
        "}",
    ]
    after, report = adapter.optimize_statements(before, contract)
    assert after == before and report["status"] == "refused"


def test_source_mutation_refuses_before_rewrite(adapter, contract):
    pin = json.loads(contract.payload)["runtime"]
    Path(pin["path"]).write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        adapter.optimize_statements(["gm_fence();"], contract)


def test_contract_mutation_refuses_before_rewrite(adapter, contract):
    contract.path.write_text("{}")
    with pytest.raises(ValueError, match="contract changed"):
        adapter.optimize_statements(["gm_fence();"], contract)


def test_runtime_helper_binding_counts_definitions_without_confusing_calls(
    adapter, tmp_path
):
    path = tmp_path / "runtime.py"
    path.write_text(
        'RUNTIME = """static inline void gm_fence(void) { do_fence(); }\n'
        'static void other(void) { gm_fence(); gm_fence(); }"""\n'
    )
    assert adapter._runtime_helper(path, "gm_fence") == "do_fence();"


def test_disabled_control_receives_same_algorithm_and_retains_statements(
    adapter, contract
):
    before = [
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
    ]
    after, report = adapter.optimize_statements(before, contract, enabled=False)
    assert after == before and report["removed"] == 0
