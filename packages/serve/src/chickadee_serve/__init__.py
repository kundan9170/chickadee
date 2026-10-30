"""`chickadee_serve` — the engine facade over a socket.

Every frontend is a client of one wire: the REPL, a TUI, an editor, `--replay`. The
wire itself is `routes.py` and `frames.py`, which import nothing from `chickadee.*`, so a client
can depend on the contract without the engine. Everything else here serves it.

This module imports nothing, so importing the contract does not import the server.
"""
