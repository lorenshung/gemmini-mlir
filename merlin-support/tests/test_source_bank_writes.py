"""Pure accounting refuses missing effects and never drops unsupported commands."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from gemmini_mvin_bank_path import _bound, _contains
from gemmini_source_bank_writes import SourceBankWriteEffects, account_stream
from merlin.targetgen.rtl import firrtl_struct as F


class PathFixture:
    """Accounting fixture, explicitly not a target/effect qualification owner."""

    def command_rows(self, command, configuration):
        if configuration is None:
            raise ValueError("missing configuration")
        if command.get("unsupported"):
            raise ValueError("unsupported broadcast")
        return command["writes"]


def write(resource="bank-one", mask=None):
    return {
        "resource": resource,
        "byte_mask": [1, 1, 0] if mask is None else mask,
        "physical_event_observed": False,
    }


def test_all_commands_preserved_and_alternative_geometry_masks_do_not_become_physical():
    stream = [
        {"class": "CONFIG_LD"},
        {
            "class": "MVIN",
            "index": 1,
            "writes": [write(), write("bank-two", [0, 0, 1])],
        },
        {"class": "COMPUTE_PRELOADED"},
        {"class": "MVOUT"},
        {"class": "FENCE"},
    ]
    commands, events = account_stream(stream, PathFixture())
    assert len(commands) == len(stream) and len(events) == 2
    assert commands[1]["status"] == "source_derived_plain_spad_writes"
    assert commands[2]["status"] == commands[3]["status"] == "UNKNOWN"
    assert all(row["compulsory"] is False for row in events)
    assert events[0]["demand"] == {"bank-one": 1} and events[1]["demand"] == {
        "bank-two": 1
    }


@pytest.mark.parametrize(
    "writes", [[], [{}], [write(mask=[0, 0, 0])], [write(mask=[True, 0])]]
)
def test_missing_or_invalid_local_effects_are_unknown_instead_of_zero(writes):
    commands, events = account_stream(
        [{"class": "CONFIG_LD"}, {"class": "MVIN", "writes": writes}], PathFixture()
    )
    assert commands[-1]["status"] == "UNKNOWN" and not events


def test_flush_invalidates_configuration_and_refused_load_does_not_drop_siblings():
    commands, events = account_stream(
        [
            {"class": "CONFIG_LD"},
            {"class": "MVIN", "unsupported": True},
            {"class": "FLUSH"},
            {"class": "MVIN", "writes": [write()]},
        ],
        PathFixture(),
    )
    assert len(commands) == 4 and not events
    assert (
        commands[1]["reason"] == "unsupported broadcast"
        and commands[-1]["reason"] == "missing configuration"
    )


def test_borrowed_or_replaced_callback_is_not_original_bound_method():
    class Owner:
        def check(self):
            return True

    a, b = Owner(), Owner()
    assert _bound(a, "check", Owner.check)() is True
    a.check = b.check
    with pytest.raises(ValueError, match="original bound"):
        _bound(a, "check", Owner.check)
    a.check = lambda: True
    with pytest.raises(ValueError, match="original bound"):
        _bound(a, "check", Owner.check)


def test_source_references_are_structural_not_repr_matching_and_json_is_no_owner():
    reference = F.parse_expr("io.req.bits.pixel_repeats")
    assert _contains(
        F.parse_expr("sub(io.req.bits.pixel_repeats, UInt<1>(1))"), reference
    )
    assert not _contains(F.parse_expr("req.bits.pixel_repeats"), reference)
    with pytest.raises(ValueError, match="original live"):
        SourceBankWriteEffects(
            source_effects={"status": "pass"}, bank_service={"status": "pass"}
        )
