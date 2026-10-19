"""Unpaid normal CLI selection/preparation intake; no author or grading run.

This calls the same selection functions as authoring_cli. An actual original
factory reissues the live owner, so use an admitted resource boundary and leave
time for the recorded historical qualification cost. It does not resume or
admit a formal Phase 2 campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    from merlin.common.cpu_pool import shared_compute_pool
    from merlin_experiments.phase2 import compiler_library_state as library_state
    from merlin_experiments.phase2 import compiler_pass_state as pass_state

    parser = argparse.ArgumentParser(description=__doc__)
    library_state.add_selection_arguments(parser)
    pass_state.add_selection_arguments(parser)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args(argv)
    with shared_compute_pool():
        library = library_state.selection_from_arguments(args)
        preparation = pass_state.preparation_from_arguments(args)
    if library is None or preparation is None:
        parser.error("explicit original library, pass selection and both preparation policies are required")
    # The factory callback may launch a child which owns its own CPU lease.
    selected = pass_state.selection_from_arguments(
        args, library=library, stage_root=args.stage_root
    )
    with shared_compute_pool(), selected.metadata_operation():
        record = selected.record()
    print(
        json.dumps(
            {
                "selection": record,
                "preparation": preparation.record(),
                "formal_stage_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
