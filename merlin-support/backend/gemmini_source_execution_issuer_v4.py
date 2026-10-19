"""Original fresh-artifact trace issuer for finite full-K RHS placement."""

from gemmini_source_execution_issuer import SourceExecutionIssuer
from gemmini_staged_dense_capability import _digest
from gemmini_staged_rhs_capability import QualifiedRhsResidencyOwner
from gemmini_timing_resources import _pin, _revalidate_sources

from merlin.common.paths import module_source_path


class SourceExecutionIssuerV4:
    def __init__(self, qualification):
        if type(qualification) is not QualifiedRhsResidencyOwner:
            raise ValueError("full-K source issuer requires its original live qualified allocation/compiler owner")
        qualification.revalidate()
        self._qualification = qualification
        self._qualification_identity = qualification.identity
        self._prototype = qualification.prototype.seed.tile.prototype
        self._prototype_identity = self._prototype.identity
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_source_execution_issuer_v4",
                "gemmini_source_execution_issuer",
                "gemmini_source_execution",
                "gemmini_source_packet_artifacts",
                "merlin.common.invocation_record",
                "merlin.common.cpu_pool",
                "merlin.common.paths",
                "merlin.targetgen.native_model_execution",
            )
        }
        self._trace = SourceExecutionIssuer.create_source_execution_owner
        self._trace_code = self._trace.__code__
        self._identity = _digest([self._qualification_identity, self._prototype_identity, self._sources])
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.original_source_execution_issuer.v4:" + self._identity

    def revalidate(self):
        self._qualification.revalidate()
        _revalidate_sources(self._sources)
        if (
            type(self._qualification) is not QualifiedRhsResidencyOwner
            or self._qualification.identity != self._qualification_identity
            or self._qualification.prototype.seed.tile.prototype is not self._prototype
            or self._prototype.identity != self._prototype_identity
            or SourceExecutionIssuer.create_source_execution_owner is not self._trace
            or self._trace.__code__ is not self._trace_code
            or _digest([self._qualification_identity, self._prototype_identity, self._sources]) != self._identity
        ):
            raise ValueError("original full-K source issuer or actual trace implementation changed")

    def source_rule(self):
        self.revalidate()
        return self._prototype.arithmetic.source_rule()

    def create_source_execution_owner(self, *, native_result, command_buffer, lowered, work_root, remaining_s):
        self.revalidate()
        return self._trace(
            self,
            native_result=native_result,
            command_buffer=command_buffer,
            lowered=lowered,
            work_root=work_root,
            remaining_s=remaining_s,
        )
