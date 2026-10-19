"""Explicit observation formats; synthetic fixtures grant no target authority."""

import hashlib
import json

import pytest
from test_source_execution_issuer import fixture as fixture  # noqa: PLC0414
from test_source_execution_issuer import module as module  # noqa: PLC0414


def test_exact_protocols_preserve_v1_flags_and_select_complete_commit_logging(module):
    from gemmini_source_execution import (
        COMMIT_ONLY_PROTOCOL,
        FULL_LOG_PROTOCOL,
        source_trace_flags,
    )

    assert source_trace_flags(FULL_LOG_PROTOCOL) == ("-l", "--log-commits")
    assert source_trace_flags(COMMIT_ONLY_PROTOCOL) == ("--log-commits",)


@pytest.mark.parametrize("protocol", [None, True, 2, "v2", "gemmini.source_observation.v3"])
def test_unknown_protocol_refuses_before_any_process_or_owner(module, protocol):
    from gemmini_source_execution import source_trace_flags

    with pytest.raises(ValueError, match="unknown original source observation"):
        source_trace_flags(protocol)


def test_unknown_protocol_does_not_create_trace_or_call_owner(fixture):
    f = fixture
    destination = f.root / "unknown-protocol"
    with pytest.raises(ValueError, match="unknown original source observation"):
        f.issuer.create_source_execution_owner(
            native_result=f.result, command_buffer=f.cb, lowered=f.lowered,
            work_root=destination, remaining_s=20, observation_protocol="unknown",
        )
    assert not destination.exists() and f.calls == []


def test_nonstring_protocol_cannot_spoof_version_equality(module):
    from gemmini_source_execution import source_trace_flags

    class Spoof:
        def __eq__(self, _other):
            return True

    with pytest.raises(ValueError, match="unknown original source observation"):
        source_trace_flags(Spoof())


def test_v2_issuer_binds_actual_selected_argv_and_owner_protocol(fixture):
    from gemmini_source_execution import COMMIT_ONLY_PROTOCOL
    from merlin.common import invocation_record

    f = fixture
    owner = f.issuer.create_source_execution_owner(
        native_result=f.result, command_buffer=f.cb, lowered=f.lowered,
        work_root=f.root / "commits-only", remaining_s=20,
        observation_protocol=COMMIT_ONLY_PROTOCOL,
    )
    args = owner.original_constructor_arguments
    observed = invocation_record.verify(args["trace_invocation"])
    assert observed["returncode"] == 0
    assert observed["argv"] == [f.runtime["spike"]["path"],
        "--extlib=" + f.runtime["functional_extension"]["path"],
        "--extension=gemmini", "--log-commits", str(f.elf)]
    assert args["observation_protocol"] == COMMIT_ONLY_PROTOCOL


def test_v1_owner_digest_payload_unchanged_v2_flags_are_bound(module):
    from gemmini_source_execution import (
        COMMIT_ONLY_PROTOCOL,
        FULL_LOG_PROTOCOL,
        SourceExecutionOwner,
        source_trace_flags,
    )

    owner = object.__new__(SourceExecutionOwner)
    owner._pins, owner._arithmetic_identity, owner._selected_inputs = {"original": "pins"}, "arithmetic", {"facts": "same"}
    owner._observation_protocol = FULL_LOG_PROTOCOL
    owner._observation_flags = source_trace_flags(FULL_LOG_PROTOCOL)
    old = hashlib.sha256(json.dumps([owner._pins, owner._arithmetic_identity, owner._selected_inputs], sort_keys=True).encode()).hexdigest()
    assert owner._digest() == old
    owner._observation_protocol = COMMIT_ONLY_PROTOCOL
    owner._observation_flags = source_trace_flags(COMMIT_ONLY_PROTOCOL)
    assert owner._digest() != old
    identity = owner._digest()
    owner._observation_flags = ("-l", "--log-commits")
    assert owner._digest() != identity
    with pytest.raises(ValueError, match="observation flags changed"):
        owner.revalidate()
