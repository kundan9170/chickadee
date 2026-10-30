Run a command in the worktree: tests, builds, linters, scripts, git reads. `command` is required.

{{^unsealed}}
The command runs inside a sandbox over the task's worktree. Its files are writable, so a test
run, a build or a formatter can write what it needs; the version-control directory is
read-only; credentials are stripped and secret files are unreadable; `/tmp` and a package cache
are kept for the session. Every command is recorded, and what it writes is recorded with it and
can be undone.

The network is off. Some commands ask the user first, as the user's own settings say: an
install, a push, a server, anything destructive. Approved, a command runs outside the sandbox,
as the user would run it: the network on, their environment, anywhere they can write. `network: true` asks
for a command that needs the network and does not say so itself; `sandbox: false` asks for one
that failed because of the sandbox. A refused command did not run.
{{/unsealed}}
{{#unsealed}}
No sandbox runs here: the command runs on the user's machine, in the task's worktree, with
credentials stripped from its environment and a command that names a secret file refused.
Nothing else stops it reaching the network, or writing the version-control directory or a file
outside the repository: never write either of those, and set `network: true` on a command that
needs the network. A command may need the user's approval before it runs, as the user's own
settings say, and a refused command did not run. Every command is recorded, and what it writes
in the worktree is recorded with it and can be undone.
{{/unsealed}}

Install a project's dependencies into the worktree, where they need no approval:
`python3 -m venv .venv && .venv/bin/pip install …`, `npm install`, `uv sync`. A pipe reports its
last command's exit code: `pip install x | tail -5` says `exit 0` when pip failed, so leave the
pipe off, or start with `set -o pipefail`, when the exit code matters.

`cwd` is a directory from the repository root, the root by default. Output under the inline cap
returns whole; over it, the head that fits and a reference you can pass to `read` for the rest.
Commands run one at a time.
