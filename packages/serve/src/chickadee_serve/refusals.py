"""Why an engine would not start serving, and the exit code each reason is.

The reasons are the one vocabulary for a refused start, whoever starts the engine:
`chickadee serve`, `chickadee-serve`, or a frontend hosting it in its own process
(`chickadee_wire.host`). `server.refused` is the one translation from the engine's errors
into them. Kept free of the engine so a client can name a reason without loading it.
"""

from __future__ import annotations

from typing import Final, Literal

__all__ = ["REFUSAL_EXIT", "Refusal"]

Refusal = Literal["held", "unbindable", "not_a_repository", "config"]
"""`held`: a server already serves the repository. `unbindable`: the address will not
bind (a socket path too long, a non-loopback host). `not_a_repository`: the path is in
no repository. `config`: the engine config does not load."""

REFUSAL_EXIT: Final[dict[Refusal, int]] = {
    "held": 5,
    "unbindable": 1,
    "not_a_repository": 4,
    "config": 1,
}
"""The exit codes: `5` the repository is held, `4` a precondition, `1` configuration."""
