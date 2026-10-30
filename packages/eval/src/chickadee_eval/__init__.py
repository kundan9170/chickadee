"""E1/E2/E3 measurement rig. Diagnoses first, compares second.

Phase 0 lands the schema and the archive layout only: what an experiment *is*, what must
be fixed before results are seen, and where results go. The runner is Phase 1 and the E2
arms are Phase 6, so nothing here executes anything.

The package imports only `chickadee.engine` and `chickadee.schema`.
"""

from __future__ import annotations

from chickadee_eval import archive
from chickadee_eval.archive import config_sha, create, find, is_archive, run_dir
from chickadee_eval.manifest import Arm, Language, Manifest, Problem, Subset, SubsetKind
from chickadee_eval.protocol import (
    CIMethod,
    CommonPolicy,
    MissingPolicy,
    Preregistration,
    validate_arms_differ_in_one_variable,
)

__all__ = [
    "Arm",
    "CIMethod",
    "CommonPolicy",
    "Language",
    "Manifest",
    "MissingPolicy",
    "Preregistration",
    "Problem",
    "Subset",
    "SubsetKind",
    "archive",
    "config_sha",
    "create",
    "find",
    "is_archive",
    "run_dir",
    "validate_arms_differ_in_one_variable",
]
