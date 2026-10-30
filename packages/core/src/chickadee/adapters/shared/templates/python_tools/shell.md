Run a command in the worktree: tests, builds, linters, scripts, git reads. It runs in a sandbox:
the worktree is writable, the network is off, credentials are stripped. `cwd` is a directory
from the repository root. `network=True` asks for the network; `sandbox=False` asks to run
outside the sandbox after a command failed because of it. Some commands ask the user first.
Returns the exit code and output; long output is cut, with a handle you can pass to
read(handle=...). Never run a command only to say you are done: call finish.

Returns text that also has `.returncode`, `.stdout` and `.stderr`, as `subprocess.run` does:
    r = shell(command="python3 -m pytest -q")
    if r.returncode != 0:
        print(r)
`.stdout` and `.stderr` are None when the output was cut or redacted; `.returncode` stays.
