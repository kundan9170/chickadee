"""Zeus's tool families: `navigate`, `act`, `talk`, `record` hold what Zeus's
tools are given — the ports and the coder's two steps. The tools themselves
are built-ins (`chickadee/tools/builtin/`), and Zeus's loop dispatches them through the one
dispatcher (`zeus/calls.py`); the read family needs nothing of Zeus's but the session's
`Reader`, so it has no module here.
"""
