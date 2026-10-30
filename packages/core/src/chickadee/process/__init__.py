"""The only `Popen` in core. Isolation wrap and Executor policy sit above this."""

from chickadee.process.harness import (
    DIFF_FLAGS,
    GIT_TIMEOUT_S,
    VERSION_TIMEOUT_S,
    git,
    run_git,
    run_harness,
)
from chickadee.process.supervisor import (
    Completed,
    ProcessSession,
    ProcessSupervisor,
    scrub_env,
)

__all__ = [
    "DIFF_FLAGS",
    "GIT_TIMEOUT_S",
    "VERSION_TIMEOUT_S",
    "Completed",
    "ProcessSession",
    "ProcessSupervisor",
    "git",
    "run_git",
    "run_harness",
    "scrub_env",
]
