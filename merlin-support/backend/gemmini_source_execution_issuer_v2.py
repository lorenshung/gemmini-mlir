"""Versioned live issuer shared by original and reconstructed factory modules."""

import math
import time
from pathlib import Path

from gemmini_source_execution_issuer import SourceExecutionIssuer
from gemmini_staged_dense_capability import _digest
from gemmini_timing_resources import _revalidate_sources
from merlin.targetgen.compiler_pass_selection import _pin


class SourceExecutionIssuerV2:
    """Same current-artifact checks; qualification may come from sealed parent."""

    def __init__(self, qualification):
        self._original = SourceExecutionIssuer(qualification)
        self._source = _pin(Path(__file__).resolve())
        self._identity = _digest([self._original.identity, self._source])
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.original_source_execution_issuer.v2:" + self._identity

    def revalidate(self):
        if type(self._original) is not SourceExecutionIssuer:
            raise ValueError(
                "delegated issuer lost its original current-artifact verifier"
            )
        self._original.revalidate()
        _revalidate_sources({"delegation_implementation": self._source})
        if _digest([self._original.identity, self._source]) != self._identity:
            raise ValueError(
                "delegated source issuer or original qualification changed"
            )

    def source_rule(self):
        self.revalidate()
        return self._original.source_rule()

    def create_source_execution_owner(
        self, *, native_result, command_buffer, lowered, work_root, remaining_s
    ):
        if (
            type(remaining_s) not in (int, float)
            or not math.isfinite(remaining_s)
            or remaining_s <= 0
        ):
            raise ValueError("source issuer requires a positive remaining wall budget")
        deadline = time.monotonic() + remaining_s
        self.revalidate()
        result = self._original.create_source_execution_owner(
            native_result=native_result,
            command_buffer=command_buffer,
            lowered=lowered,
            work_root=work_root,
            remaining_s=deadline - time.monotonic(),
        )
        self.revalidate()
        if time.monotonic() >= deadline:
            raise ValueError(
                "source issuer budget expired during original delegated owner revalidation"
            )
        return result
