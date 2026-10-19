"""Saved labels or matching hashes cannot construct original effect owners."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from gemmini_execute_cutpoints import SelectedExecuteCutpoints
from gemmini_source_resources import SourceResourceEffects


def test_execute_source_owner_rejects_saved_or_synthetic_receipt():
    with pytest.raises(ValueError, match="original live selected"):
        SelectedExecuteCutpoints({"status": "verified"}, {"floor": 1})


def test_source_effects_reject_saved_receipt_or_matching_hash_label():
    with pytest.raises(ValueError, match="original live checked"):
        SourceResourceEffects(
            selected={}, realization={"status": "verified"}, dma={}, execute={}
        )
