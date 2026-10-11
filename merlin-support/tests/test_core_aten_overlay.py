from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _provider_module():
    path = Path(__file__).resolve().parents[1] / "core_aten/overlay_provider.py"
    spec = importlib.util.spec_from_file_location("gemmini_core_aten_overlay_provider", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_provider = _provider_module()
build_gemmini_overlay = _provider.build_gemmini_overlay
gemmini_overlay_digest = _provider.gemmini_overlay_digest
gemmini_profile = _provider.gemmini_profile
validate_gemmini_overlay = _provider.validate_gemmini_overlay


def test_header_macro_parser_accepts_only_exact_integer_defines() -> None:
    macro = _provider._macro
    text = "\n".join(
        [
            "#define DIM 16",
            "#define  BANK_NUM \t4  ",
            "#define BANK_ROWS 4096\r",
            "#define ACC_ROWS 1024 // trailing comment",
            " #define MAX_BYTES 64",
            "#define DIMENSION 99",
            "#define DIM_X 98",
            "#define NAME_ONLY",
            "#define MAX_BYTES (64)",
            "#define MAX_BYTES 64",
        ]
    )
    assert macro(text, "DIM") == 16
    assert macro(text, "BANK_NUM") == 4
    assert macro(text, "BANK_ROWS") == 4096
    assert macro(text, "MAX_BYTES") == 64
    with pytest.raises(RuntimeError):
        macro(text, "ACC_ROWS")
    with pytest.raises(RuntimeError):
        macro(text, "NAME_ONLY")
    with pytest.raises(RuntimeError):
        macro("#defineDIM 3", "DIM")


def test_gemmini_profile_is_evidence_derived_and_keeps_portable_cases() -> None:
    profile = gemmini_profile()
    assert profile["facts"]["tile_edge"] == 16
    assert profile["facts"]["operand_dtype"] == "int8"
    assert profile["facts"]["accumulator_dtype"] == "int32"
    assert len(profile["sources"]) == 3


def test_gemmini_overlay_is_additive_deterministic_and_exact() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("z3", reason="exact-cover certificate refresh requires z3")
    first = build_gemmini_overlay(pytorch_version="test")
    second = build_gemmini_overlay(pytorch_version="test")
    assert first == second
    assert first["portable_replacement_forbidden"] is True
    assert first["core_aten_denominator_additions"] == 0
    assert first["candidate_count"] == 10
    assert first["selected_count"] == 9
    assert first["cover"]["status"] == "optimal"
    assert first["cover"]["selected_union_matches_denominator"] is True
    assert first["overlay_sha256"] == gemmini_overlay_digest(first)
    assert validate_gemmini_overlay(first) == first["selected_cases"]
    assert {case["denominator_classification"] for case in first["selected_cases"]} == {
        "core",
        "non_core_target_bridge",
    }
    assert "aten._int_mm.default" not in {
        case["overload"] for case in first["selected_cases"] if case["denominator_classification"] == "core"
    }


def test_gemmini_overlay_covers_tile_rank_layout_conv_and_widening() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("z3", reason="exact-cover certificate refresh requires z3")
    document = build_gemmini_overlay(pytorch_version="test")
    obligations = set(document["obligations"])
    assert {
        "gemmini::cell::contraction/i8/aligned",
        "gemmini::cell::contraction/i8/partial",
        "gemmini::cell::contraction/i8/sub_tile",
        "gemmini::layout::contraction/rhs_transposed",
        "gemmini::rank::contraction/2",
        "gemmini::rank::contraction/3",
        "gemmini::rank::contraction/4",
        "gemmini::numeric::i8xi8_i32",
        "gemmini::readout::int8",
        "gemmini::readout::int32",
    } <= obligations
    selected_union = {obligation for case in document["selected_cases"] for obligation in case["covered_obligations"]}
    assert selected_union == obligations


def test_profile_geometry_is_derived_after_pin_verification(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.common import provenance

    header = tmp_path / "include/gemmini_params.h"
    header.parent.mkdir()
    header.write_text(
        "#define DIM 8\n#define BANK_NUM 2\n#define BANK_ROWS 2048\n"
        "#define ACC_ROWS 512\n#define MAX_BYTES 32\n"
        "typedef int8_t elem_t;\ntypedef int32_t acc_t;\n"
    )
    checked = []
    monkeypatch.setattr(provenance, "pin", lambda _: SimpleNamespace(commit="synthetic-pin"))
    monkeypatch.setattr(provenance, "require", lambda name, **kwargs: checked.append((name, kwargs)))
    profile = gemmini_profile(tmp_path)
    assert profile["facts"]["tile_edge"] == 8
    assert profile["facts"]["scratchpad_banks"] == 2
    assert checked == [("gemmini_isa_headers", {"checkout": tmp_path, "reads": ["include/gemmini_params.h"]})]
    header.write_text(header.read_text().replace("#define DIM 8", "#define DIM 0"))
    with pytest.raises(RuntimeError, match="nonpositive tile_edge"):
        gemmini_profile(tmp_path)


def test_profile_refuses_unverified_header(tmp_path, monkeypatch):
    from merlin.common import provenance

    def refuse(*args, **kwargs):
        raise RuntimeError("header is off the selected hardware pin")

    monkeypatch.setattr(provenance, "require", refuse)
    with pytest.raises(RuntimeError, match="off the selected hardware pin"):
        gemmini_profile(tmp_path)
