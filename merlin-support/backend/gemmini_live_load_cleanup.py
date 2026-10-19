"""Live host admission after independent paired checks; receipts are only data.

This does not authorize hardware/timing or dense execution. Original callers
obtain this owner only by running the source-pinned admission checker. Its
serialized record cannot reconstruct the live selected library/target owners.
"""

from __future__ import annotations

import copy
import hashlib
import json

from gemmini_timing_resources import _pin
from merlin.targetgen.compiler_library import CompilerLibrarySelection

_CHECKED_ADMISSION = object()


class LiveLoadCleanupOwner:
    def __init__(self, *, receipt, selected, library, checker, _checked):
        if _checked is not _CHECKED_ADMISSION:
            raise ValueError("run the original independent admission checker")
        if type(library) is not CompilerLibrarySelection:
            raise ValueError(
                "live cleanup requires the original selected library owner"
            )
        self._receipt, self._selected, self._library = receipt, selected, library
        self._binding = library.record()
        self._source = _pin(__file__)
        self._checker = _pin(checker)
        self._record = {
            "schema": "gemmini.live_fenced_load_cleanup.v1",
            "receipt_identity": receipt.identity,
            "compiler_library_selection": self._binding,
            "owner_source": self._source,
            "admission_checker": self._checker,
        }
        self._identity = hashlib.sha256(
            json.dumps(self._record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.live_fenced_load_cleanup.v1:" + self._identity

    def revalidate(self):
        if (
            self._source != _pin(__file__)
            or self._checker != _pin(self._checker["path"])
            or type(self._library) is not CompilerLibrarySelection
            or self._library.record() != self._binding
            or self._receipt.identity != self._record["receipt_identity"]
            or hashlib.sha256(
                json.dumps(self._record, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            != self._identity
        ):
            raise ValueError("original live cleanup source/library selection changed")
        self._selected.revalidate()
        row = self._receipt.revalidate()
        if row.get("compiler_library_selection") != self._binding:
            raise ValueError(
                "independent admission selected a different shared release"
            )
        if row["shared_library_sources"] != {
            member["module"]: member["sha256"]
            for member in self._binding["contract"]["members"]
            if member["module"] is not None
        }:
            raise ValueError("live cleanup sources differ from the selected release")
        if row["selected_inputs"] != self._selected.report()["selected_inputs"]:
            raise ValueError("independent admission selected a different target")
        return row

    def selected_binding(self):
        self.revalidate()
        return {
            **self._receipt.selected_binding(),
            "compiler_library_selection": self._library.record(),
        }

    def record(self):
        self.revalidate()
        return {
            **self._receipt.record(),
            "identity": self.identity,
            "live_host_admission": copy.deepcopy(self._record),
            "selected_binding": self.selected_binding(),
            "serialized_record_authority": "NONE; original live host owners required",
        }

    def rewrite(self, statements, *, enabled=True, execution_domain):
        self.revalidate()
        result = self._receipt.rewrite(
            statements, enabled=enabled, execution_domain=execution_domain
        )
        self.revalidate()
        rewritten, report = result
        return rewritten, {**report, "capability_identity": self.identity}

    def check_actual_pair(self, before, after, *, execution_domain):
        self.revalidate()
        result = self._receipt.check_actual_pair(
            before, after, execution_domain=execution_domain
        )
        self.revalidate()
        return result


def _after_independent_checks(*, receipt, selected, library, checker):
    """Only the original checker calls this after replaying complete evidence."""
    return LiveLoadCleanupOwner(
        receipt=receipt,
        selected=selected,
        library=library,
        checker=checker,
        _checked=_CHECKED_ADMISSION,
    )
