"""Text extents stay separate from linked tensor/data ownership."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def M(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_direct_host_scope")


TEXT = (
    "00001000 00000020 T kernel\n00001020 00000040 t main\n00002000 00000040 B tensor_0"
)
SECTIONS = [(".text", 1, 6, 128, 128, 0x1000), (".bss", 8, 3, 256, 64, 0x2000)]


def test_unique_code_extents_do_not_require_data_names(M):
    assert M.select_code_extents(TEXT, SECTIONS, "kernel") == {
        "kernel": (0x1000, 32),
        "main": (0x1020, 64),
    }


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "data",
        "zero",
        "duplicate",
        "nonexecuting",
        "outside",
        "ambiguous",
        "same_main",
    ],
)
def test_foreign_or_incomplete_code_extent_refuses(M, fault):
    text = TEXT
    sections = list(SECTIONS)
    entry = "kernel"
    if fault == "missing":
        text = text.replace("kernel", "other")
    elif fault == "data":
        text = text.replace("T kernel", "D kernel")
    elif fault == "zero":
        text = text.replace("00000020 T", "00000000 T")
    elif fault == "duplicate":
        text += "\n00001000 00000020 W kernel"
    elif fault == "nonexecuting":
        sections[0] = (".text", 1, 2, 128, 128, 0x1000)
    elif fault == "outside":
        text = text.replace("00000040 t main", "00000100 t main")
    elif fault == "ambiguous":
        sections.append(sections[0])
    else:
        entry = "main"
    with pytest.raises(ValueError):
        M.select_code_extents(text, sections, entry)
