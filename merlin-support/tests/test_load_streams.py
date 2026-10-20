"""Local unsigned normalization cuts; no target timing or execution grant."""

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_load_streams")


def load_module(adapter):
    return adapter.F.parse_module(
        "Load",
        list(
            enumerate(
                """module Load :
  input cmd_q : {io : {deq : {bits : {cmd : {rs1 : UInt<16>}}}}}
  node raw = bits(cmd_q.io.deq.bits.cmd.rs1, 9, 8)
  node normalized = mux(eq(raw, UInt<2>(0)), UInt<2>(1), raw)
  wire pixel_repeats : UInt<2>[3]
  wire state_id : UInt<2>
  connect pixel_repeats[state_id], normalized
""".splitlines(),
                1,
            )
        ),
    )


def pixel_module(adapter):
    return adapter.F.parse_module(
        "Pixel",
        list(
            enumerate(
                """module Pixel :
  input io : {req : {bits : {pixel_repeats : UInt<2>}}}
  wire req : {bits : {pixel_repeats : UInt<2>}}
  node captured = sub(io.req.bits.pixel_repeats, UInt<1>(1))
  connect req.bits.pixel_repeats, captured
""".splitlines(),
                1,
            )
        ),
    )


def test_original_zero_normalization_and_one_capture_are_identity(adapter):
    load, pixel = load_module(adapter), pixel_module(adapter)
    assert adapter._one(load, "pixel_repeats[state_id]") == adapter.F.Ref("normalized")
    assert adapter._contains(
        pixel,
        adapter.F.Ref("captured"),
        adapter.F.parse_expr("io.req.bits.pixel_repeats"),
    )
    adapter._identity_pixels(
        adapter.C.ScalarCutpoints(load),
        adapter.C.ScalarCutpoints(pixel),
        {"offset": 8},
        "normalized",
        "captured",
    )


@pytest.mark.parametrize(
    "fault", ["no_normalization", "repeat_twice", "bad_capture", "wrong_offset"]
)
def test_changed_normalization_or_selected_operand_slice_refuses(adapter, fault):
    load, pixel = load_module(adapter), pixel_module(adapter)
    offset = 8
    if fault == "no_normalization":
        _, number, locator = load.nodes["normalized"]
        load.nodes["normalized"] = adapter.F.Ref("raw"), number, locator
    elif fault == "repeat_twice":
        _, number, locator = load.nodes["normalized"]
        load.nodes["normalized"] = adapter.F.Lit("UInt", 2, 2), number, locator
    elif fault == "bad_capture":
        _, number, locator = pixel.nodes["captured"]
        pixel.nodes["captured"] = (
            adapter.F.parse_expr("io.req.bits.pixel_repeats"),
            number,
            locator,
        )
    else:
        offset = 9
    with pytest.raises(ValueError, match="not identity repetition"):
        adapter._identity_pixels(
            adapter.C.ScalarCutpoints(load),
            adapter.C.ScalarCutpoints(pixel),
            {"offset": offset},
            "normalized",
            "captured",
        )


def test_ambiguous_destination_and_self_referential_capture_refuse(adapter):
    load = load_module(adapter)
    load.connects.append(replace(load.connects[0]))
    with pytest.raises(ValueError, match="unique original node"):
        adapter._one(load, "pixel_repeats[state_id]")
    pixel = pixel_module(adapter)
    _, number, locator = pixel.nodes["captured"]
    pixel.nodes["captured"] = adapter.F.Ref("captured"), number, locator
    with pytest.raises(ValueError, match="cyclic"):
        adapter._contains(pixel, adapter.F.Ref("captured"), adapter.F.Ref("foreign"))


@pytest.mark.parametrize("fault", ["source", "port", "state", "owner"])
def test_source_and_selected_mapping_custody_stay_fresh(adapter, tmp_path, fault):
    path = tmp_path / "neutral.fir"
    path.write_text("original neutral source")
    owner = object.__new__(adapter.SelectedLoadStreams)
    owner.selected = owner._selected = SimpleNamespace(revalidate=lambda: None)
    owner.ports = owner._ports = {"MVIN": 0, "MVIN2": 1}
    owner.states = owner._states = (0, 1)
    owner.sources = {"firrtl": adapter._pin(path)}
    owner.mapping = owner._mapping = SimpleNamespace(
        ports=(("MVIN", 0), ("MVIN2", 1)),
        states=(0, 1),
        config_class="CONFIG_LD",
        source_sha256=owner.sources["firrtl"]["sha256"],
    )
    assert owner.record()["timing_authority"] is False
    if fault == "source":
        path.write_text("changed source")
    elif fault == "port":
        owner.mapping.ports = (("MVIN", 1), ("MVIN2", 0))
    elif fault == "state":
        owner.mapping.states = (0, 2)
    else:
        owner.selected = SimpleNamespace(revalidate=lambda: None)
    with pytest.raises(ValueError):
        owner.revalidate()
