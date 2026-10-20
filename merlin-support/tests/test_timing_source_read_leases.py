"""Real source-byte custody controls; no timing or compiler qualification."""

import errno
import hashlib
import os

import gemmini_timing_resources as T
import pytest
from merlin.common import read_lease_digest as L


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    L.close()
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    yield
    L.close()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "retained-source.bin"
    path.write_bytes(b"nonconstant-source" * 32768)
    return path


def original_pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require_lease(path):
    if L.digest(path) is None:
        pytest.skip("selected filesystem does not support kernel read leases")


def break_lease(path):
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as error:
        assert error.errno in (errno.EAGAIN, errno.EWOULDBLOCK)
    else:
        os.close(fd)


def test_same_original_bytes_reuse_digest_with_live_lease(source, monkeypatch):
    expected = original_pin(source)
    require_lease(source)
    monkeypatch.setattr(
        T.hashlib, "file_digest", lambda *_a: pytest.fail("unexpected full rehash")
    )
    assert T._pin(source) == expected
    T._revalidate_sources({"source": expected})


@pytest.mark.parametrize("fault", ["writer", "replace"])
def test_broken_custody_or_replaced_original_refuses(source, fault):
    expected = T._pin(source)
    require_lease(source)
    if fault == "writer":
        break_lease(source)
    else:
        replacement = source.with_suffix(".replacement")
        replacement.write_bytes(source.read_bytes())
        replacement.replace(source)
    with pytest.raises(RuntimeError, match="custody changed"):
        T._revalidate_sources({"source": expected})


def test_early_member_lease_is_rechecked_after_other_pins(
    source, tmp_path, monkeypatch
):
    expected = T._pin(source)
    require_lease(source)
    later = tmp_path / "later.bin"
    later.write_bytes(b"small unleased metadata")
    later_pin = original_pin(later)
    original = T._pin

    def intervening(path):
        result = original(path)
        if str(path) == str(later):
            break_lease(source)
        return result

    monkeypatch.setattr(T, "_pin", intervening)
    with pytest.raises(RuntimeError, match="custody changed"):
        T._revalidate_sources({"source": expected, "later": later_pin})


@pytest.mark.parametrize("mode", ["disabled", "unsupported"])
def test_unselected_or_unavailable_leases_keep_full_byte_verification(
    source, monkeypatch, mode
):
    if mode == "disabled":
        monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "0")
    else:
        monkeypatch.setattr(L, "digest", lambda *_a: None)
    before = T._pin(source)
    source.write_bytes(b"changed-original")
    assert T._pin(source) != before
    with pytest.raises(ValueError, match="source changed"):
        T._revalidate_sources({"source": before})
