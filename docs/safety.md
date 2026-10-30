# Safety

What Chickadee guarantees, in plain words, and where each guarantee is enforced.

**Models never write or run anything directly.** A model proposes tool calls. Harness code
performs every effect. Model-authored file writes go through one module, the edit port. Commands
go through one executor and one process supervisor. The checks the harness runs on its own
(the facts it reports, its command classifier, the edit port's refusals) never call a model.

**Every edit is a recorded, undoable step.** Each writing call is journaled before and after,
with a snapshot of the worktree. One accepted task is one commit, and `undo_task` puts the
tree back. A dropped task restores every file it changed.

**Commands run sandboxed, in a worktree.** The worker's shell writes only the session's
worktree. `.git` is read-only inside it, and the network is off. The backend is bubblewrap on
Linux, Seatbelt on macOS, or a container. `chickadee setup` probes the sandbox and tells you
when it cannot run. A run configured to require a sandbox refuses to run commands without one.

**Risky actions ask first.** Commands are classified, and the tier of a command decides whether
you are asked. Reads run without asking. Destructive git, network access, credentials, and
anything that expands the scope of the run are confirmed at dispatch, with the concrete command
shown. A destructive command is never covered by an "always allow". Unattended, anything not
authorized in advance is denied.

**Secrets stay out.** A denylist keeps secret paths out of reads and writes. API keys are read
from environment variables named in the config, never written to a file. A repository's own
config cannot repoint a model, set a price or raise a spend cap.

**Spend has a hard cap.** Every run has a finite dollar cap, set in your config. Warnings fire at
fractions of it. Reaching it ends the run with a resumable terminal state and a handoff, not a
silent continuation. You can raise or lift the cap for one session, and the spend is still
counted and shown. Caps can also be set per rung.

**The journal is the source of truth.** State is a fold over the journal. After a crash the
harness observes the real state of the repository and reconciles before retrying, and it never
blindly retries an effect that may already have happened.

**Verification is never overstated.** A run that completes says the episode ended, not that the
change is right. A report lists what ran: which files changed, whether syntax checks passed,
which tests ran and with what result. Tests written during the same run can guide the work, but
they are not independent proof. Where you supplied no acceptance tests, the handoff says
behaviour was not independently verified.

## Limits

- Chickadee is a personal prototype. It can use MCP tools under policy for trusted personal
  use. It is not hardened against prompt injection from untrusted content, and it does not yet
  have scoped credentials or stronger isolation, so it is not ready for untrusted repositories
  or multi-user deployments.
- The sandbox is only as strong as the backend. With no enforcing backend, commands run
  without isolation unless you require one.

To report a vulnerability, see [SECURITY.md](../SECURITY.md).
