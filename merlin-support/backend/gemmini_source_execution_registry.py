"""Explicit original factory adding a current-source diagnostic issuer.

The selected compiler mechanism remains unchanged. This separate live owner
can issue only per-current-artifact source checks, never preparation, timing,
hardware certification or permission inferred from a saved JSON success.
"""

from gemmini_source_execution_issuer import SourceExecutionIssuer
from gemmini_staged_tile_capability import QualifiedStagedTileOwner
from gemmini_staged_tile_capability import restore_registry as restore_tile_registry
from merlin.common.cpu_pool import shared_compute_pool
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)


def restore_registry(inputs, *, work_root):
    # Original child admission owns its leaf lease; never wrap this factory in
    # an outer CPU lease while that child waits. The underlying factory leases
    # its independent source-pair proof phase after the child returns.
    registry = restore_tile_registry(inputs, work_root=work_root)
    with shared_compute_pool(timeout_s=120):
        rows = registry.record()["capabilities"]
        if len(rows) != 1 or rows[0]["mechanism"] != "staged_dense_tile_order_raw_i32":
            raise ValueError(
                "source issuer factory requires its exact original selected tile capability"
            )
        binding = PassCapabilityBinding(**rows[0]["binding"])
        selected = registry.resolve("staged_dense_tile_order_raw_i32", binding)
        if type(selected.owner) is not QualifiedStagedTileOwner:
            raise ValueError(
                "source issuer factory cannot adopt a saved or foreign qualification"
            )
        issuer = SourceExecutionIssuer(selected.owner)
        return ReviewedPassCapabilities(
            (
                selected,
                ReviewedPassCapability("source_execution_diagnostic", binding, issuer),
            )
        )
