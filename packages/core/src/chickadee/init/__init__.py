"""Repository initialisation: identity, lease, staging and atomic publication.

Submodules are re-exported as modules, not flattened. `publish.py` contains a function
called `publish`, and a package-level re-export of the function shadows the module of the
same name — so `from chickadee.init import publish` would silently hand a caller the
function and fail at the first attribute access.
"""

from __future__ import annotations

from chickadee.init import detect, lease, publish, toolchain
from chickadee.init.detect import NotAGitRepository, ProfileEvidence, RepoIdentity
from chickadee.init.lease import LeaseBusy, LeaseError, RepoLease
from chickadee.init.orchestrate import InitError, InitReport, init, require_initialised

__all__ = [
    "InitError",
    "InitReport",
    "LeaseBusy",
    "LeaseError",
    "NotAGitRepository",
    "ProfileEvidence",
    "RepoIdentity",
    "RepoLease",
    "detect",
    "init",
    "lease",
    "publish",
    "require_initialised",
    "toolchain",
]
