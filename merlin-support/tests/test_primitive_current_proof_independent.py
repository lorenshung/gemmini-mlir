"""Neutral failed-constructor lifetime controls, not target qualification."""

from types import SimpleNamespace

import gemmini_primitive_current_proof as P
import pytest


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_first_seed_check_expires_view_even_if_exception_keeps_it_alive(
    monkeypatch, failure
):
    # An unassigned failed constructor is still reachable through its traceback.
    # Keep that object explicitly alive to distinguish revocation from weakref GC.
    captured = []

    def reject_seed():
        captured.extend(P._ACTIVE)
        raise failure("neutral seed verification failed")

    context = SimpleNamespace(revalidate_current_seed=reject_seed)
    owner = SimpleNamespace(
        identity="neutral",
        context=context,
        selected=object(),
        arithmetic=object(),
        cleanup=object(),
        nm="neutral-nm",
        _sources={},
    )
    # Only the lifetime state machine is tested. This neutral object cannot
    # satisfy the original target owner/type/semantic validation or issue a grant.
    monkeypatch.setattr(P, "_owner", lambda candidate: None)
    with pytest.raises(failure, match="seed verification failed") as retained:
        P._CurrentProofView(owner, issuer=P._ISSUER)
    assert retained.value.__traceback__ is not None
    assert len(captured) == 1
    assert captured[0] not in P._ACTIVE
