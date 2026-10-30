# Configuration

This page covers where Chickadee reads its settings, how to set up a first config, and every
option. Two examples are validated against the real loader:

- [`examples/config.toml`](../examples/config.toml): the minimum, one frontier model and one local one.
- [`examples/config.full.toml`](../examples/config.full.toml): every table, with example values.

## Contents

1. [Where settings live](#where-settings-live)
2. [Set it up in five steps](#set-it-up-in-five-steps)
3. [Models, rungs and roles](#models-rungs-and-roles)
4. [Reference: every table](#reference-every-table)
5. [Limits](#limits)
6. [Approval preferences](#approval-preferences)
7. [Command rows and hooks](#command-rows-and-hooks)
8. [Environment variables](#environment-variables)
9. [Check your config](#check-your-config)

## Where settings live

Settings are read in layers. A later layer overrides an earlier one, key by key.

| # | Layer | File | Who owns it |
|---|---|---|---|
| 1 | defaults | built in | |
| 2 | user | `~/.config/chickadee/config.toml` (`$XDG_CONFIG_HOME/chickadee/config.toml`) | you |
| 3 | project | `<repo>/.chickadee/config.toml` | the repository, committed |
| 4 | local | `work.toml`, kept in the repository's local state for this clone | you, for one clone |
| 5 | environment | `CHICKADEE_*` variables | you, for one run |

**The split that matters.** Models, prices, spend caps, the sandbox and everything else that
costs money or widens what can run belong to you, in the *user* file. A repository's
`.chickadee/config.toml` may set only four things:

| Table | What a repository may say |
|---|---|
| `profiles` | which language profile applies (`"auto"` detects per file) |
| `secrets` | extra paths to keep secret |
| `commands` | extra command rows, see [command rows](#command-rows-and-hooks) |
| `hooks` | hook scripts, see [hooks](#command-rows-and-hooks) |

Anything else in a repository's file is refused at load, so cloning a repository can never
repoint a model, set a price or raise your spend cap. Executable settings a repository ships
(`commands`, `hooks`, an approval `allow`) stay inert until you accept them for that clone with
`chickadee trust`.

Two more files sit beside the config:

- `approval.toml`, your standing answers to permission questions. One in `~/.config/chickadee/`
  and one in `<repo>/.chickadee/`. See [approval preferences](#approval-preferences).
- `.chickadee/conventions.md` in a repository, your house conventions, which the roles read.

## Set it up in five steps

1. **Create the file.** `chickadee setup` writes a commented starting config to
   `~/.config/chickadee/config.toml` if none exists. It never overwrites one. It also probes
   the sandbox and lists which language servers it found.

   ```bash
   uv run chickadee setup
   ```

2. **Declare your models.** Each `[models.<name>]` is one OpenAI-compatible endpoint. The API
   key is read from the environment variable you name, never written to the file.

   ```toml
   [models.frontier]
   base_url = "https://api.example.com/v1"
   model_id = "your-frontier-model"
   api_key_env = "FRONTIER_API_KEY"
   context_window = 200000
   max_output_tokens = 32000

   [models.local]
   base_url = "http://localhost:8000/v1"
   model_id = "your-local-coder"
   context_window = 131072
   max_output_tokens = 16384
   ```

3. **Price them.** Every model needs a `[prices.<name>]` row, in USD per million tokens,
   before a call to it can be charged. Price a local model at a hosted-market equivalent so the
   spend cap and any cost comparison stay honest.

   ```toml
   [prices.frontier]
   price_in = 3.00
   price_out = 15.00

   [prices.local]
   price_in = 0.20
   price_out = 0.60
   ```

4. **Point each rung at a model.**

   ```toml
   [rungs]
   high-param = "frontier"   # Zeus and Athena
   mid-param  = "frontier"   # locate, critic, reviewer and the other helpers
   low-param  = "local"      # the worker
   ```

5. **Set the cap and export the key.**

   ```toml
   [spend]
   hard_cost_usd = 5.00
   warning_fractions = [0.5, 0.8]
   ```

   ```bash
   export FRONTIER_API_KEY=...
   ```

A run refuses to start with no finite `hard_cost_usd`. The file `chickadee setup` writes sets
it to $5.00.

## Models, rungs and roles

- A **model** is an endpoint you can call: `[models.<name>]`.
- A **rung** is a named slot a role runs on. The three default rungs are `high-param`,
  `mid-param` and `low-param`. `[rungs]` maps each rung to a model name.
- A **role** is a job with a fixed set of tools and permissions. Each has a default rung.
  `[roles]` moves a role onto another rung.

| Role | Default rung | Job |
|---|---|---|
| `zeus` | `high-param` | the conversation, designs |
| `athena` | `high-param` | plans, delegates and reviews tasks |
| `worker` | `low-param` | edits and runs commands for one task |
| `locate` | `mid-param` | finds where something lives |
| `critic` | `mid-param` | proposes checks |
| `reader`, `reviewer`, `researcher` | `mid-param` | read-only helpers Zeus can spawn |
| `residue`, `checkpoint` | `high-param`, `mid-param` | single calls that write summaries |
| `title` | `low-param` | names a session |

One model can serve several rungs. You can add rungs of your own by naming them in `[rungs]`,
then move a role onto one with `[roles]`. A role's tools and call caps are fixed by its
definition. Configuration can only choose within them.

## Reference: every table

Types: `int`, `float`, `bool`, `str`, lists written `["a", "b"]`. "no default" means required.

### `[spend]`

| Key | Default | Meaning |
|---|---|---|
| `hard_cost_usd` | no default | The run's hard cap in USD. Must be finite and above zero. |
| `warning_fractions` | `[0.8]` | Fractions of the cap at which a warning fires. Strictly increasing, each between 0 and 1. |
| `packet_warning_cost_usd` | none | A diagnostic warning for one unit of work. It never replaces the cap. |
| `[spend.rungs]` | empty | A cap per rung, beneath the total, e.g. `high-param = 3.00`. A rung not named has no rung cap. |

### `[models.<name>]`

| Key | Default | Meaning |
|---|---|---|
| `base_url` | no default | The endpoint, e.g. `https://api.example.com/v1`. |
| `model_id` | no default | The model name the endpoint expects. |
| `context_window` | no default | The model's window in tokens. |
| `max_output_tokens` | no default | The most tokens one reply may produce. |
| `provider` | `"local"` | A label for the provider. |
| `api_key_env` | none | Name of the environment variable holding the key. |
| `first_token_s` | `300` | Longest wait for a reply's first streamed event before the attempt is retried. |
| `token_gap_s` | `60` | Longest silence mid-reply before the attempt is retried. |
| `reasoning` | none | `{ level = "high" }` with `level` one of `minimal`, `low`, `medium`, `high`, and an optional `budget_tokens`. |
| `input_limit` | none | The provider's separate input cap, when it has one. Never above `context_window`. |
| `temperature` | none | Default sampling temperature. |
| `supports_temperature` | `true` | `false` if the endpoint rejects the parameter. |
| `supports_images` | `true` | Whether the model can see images. |
| `supports_documents` | `false` | Whether it reads a PDF itself. |
| `native_tools` | `true` | The endpoint takes a `tools` array. Every role with tools needs this. |
| `grammar_tools` | `false` | The endpoint can decode a tool under a grammar. |
| `forced_tools` | `false` | The endpoint enforces a forced tool call. |
| `json_mode` | `false` | The model may use the provider's JSON mode for structured output. |

### `[prices.<name>]`

USD per million tokens. The name matches a model in `[models]`.

| Key | Default | Meaning |
|---|---|---|
| `price_in` | no default | Input tokens. |
| `price_out` | no default | Output tokens. |
| `price_cached_in` | none | Cached input. If unset, cached reads are priced at `price_in`. |
| `price_cache_write` | none | Cache creation, for providers that charge for it. |
| `cache_affinity` | `false` | The provider caches prompt prefixes. |
| `cache_ttl_s` | `0` | How long a cached prefix stays warm. |

### `[rungs]` and `[roles]`

`[rungs]` maps a rung name to a model name from `[models]`. Every default rung must be set.
`[roles]` maps a role name to a rung name, and is optional.

### `[sandbox]`

These keys are yours alone. A repository cannot set them.

| Key | Default | Meaning |
|---|---|---|
| `backend` | `"auto"` | `auto`, `native` (bubblewrap on Linux, Seatbelt on macOS), `container`, or `none`. `container` is not usable yet. |
| `require_sandbox` | `true` | Refuse to run commands when no enforcing backend exists, instead of running unsandboxed. |
| `allow_unsandboxed_escapes` | `false` | Let the scout run a command where no backend enforces isolation. |
| `[sandbox.allow_unsandboxed_shell]` | all `false` | Per role (`zeus`, `worker`, `athena`): run shell commands unasked where no sandbox enforces anything. Off, every such command is confirmed first. |
| `allow_unsandboxed_python` | `false` | Let the worker's Python blocks run in a plain child process where no sandbox exists. |
| `never_ask` | `false` | Approve every shell command without asking, recording each as a preference. Your `deny` rules and secret paths still refuse. |

### `[workspace]`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `"in_place"` | `in_place` runs in your checkout, with turn snapshots as the undo point. `worktree` runs in a git worktree. |
| `require_lease` | `true` | Take a repository-wide write lease so two sessions never write one tree. |

You can also choose per session with `chickadee --worktree` or `chickadee --in-place`.

### `[work]`

| Key | Default | Meaning |
|---|---|---|
| `dir` | `"docs/chickadee"` | The folder, relative to the repository root, where designs, plans and briefs are written. Must be inside the repository. |
| `commit` | `true` | `true` commits that folder with the work. `false` keeps it out of the repository. |

### `[undo]`

| Key | Default | Meaning |
|---|---|---|
| `keep_days` | `7` | A snapshot younger than this is never removed. |
| `keep_turns` | `50` | The newest this-many snapshotted turns are kept whatever their age. |
| `untracked_max_bytes` | `2097152` | A snapshot skips an untracked file larger than this and names it. |
| `on_your_branch` | `"restore"` | After `apply` or `squash`, an undo puts files back uncommitted (`restore`) or also commits (`commit`). In place, nothing is committed. |

### `[interaction]`

| Key | Default | Meaning |
|---|---|---|
| `question_timeout_s` | `900` | A blocking question unanswered this long takes its default, so a lost frontend cannot hang a session. |
| `max_questions_per_run` | `3` | The blocking questions one run may ask. Past it, a question takes its default. |
| `auto_title` | `true` | Name a session after its first turn with one short call. |

### `[skills]`

| Key | Default | Meaning |
|---|---|---|
| `external` | `true` | Also index skills from `.claude/skills/` and `.agents/skills/` in the repository and your home directory, read-only. `false` indexes only `.chickadee/` and your own. |

### `[web.fetch]` and `[web.search]`

| Key | Default | Meaning |
|---|---|---|
| `fetch.engine` | `"auto"` | `auto` uses `lightpanda` when installed and sandboxed, else `http`. `http` reads a page as served. `lightpanda` runs its scripts first. |
| `fetch.approval` | `"never"` | `per_domain` asks the first time a session reaches each host. |
| `fetch.docs_domains` | `[]` | Extra documentation hosts where a URL nobody gave may be fetched. |
| `search.engine` | none | `tavily`, `brave`, `exa`, `mojeek` or `searxng`. Unset uses a keyless trial. |
| `search.api_key_env` | none | Variable holding the engine's key. Every engine except `searxng` needs one. |
| `search.searxng_url` | none | Your own SearXNG instance. |
| `search.github_token_env` | none | Variable holding a GitHub token, for a higher rate limit. Sent to GitHub only. |

### `[secrets]`

| Key | Default | Meaning |
|---|---|---|
| `deny_add` | `[]` | Extra path globs kept out of reads and writes. A repository may set this. |
| `deny_remove` | `[]` | Built-in denylist entries to remove, by name. A removal a repository ships stays inert until you accept it with `chickadee trust`. |
| `allow_secret_read` | `false` | Turn the secret-path protection off entirely. Same rule: inert unless you accept it. |

### `[tools.<role>]`

Chooses a role's tools, within the role's ceiling. Every key is optional. A `[tools.<role>.models."<model>"]`
section overrides the same keys for one model.

| Key | Meaning |
|---|---|
| `enabled` | The tool names the role is offered. |
| `shown_as` | Rename tools as the model sees them. |
| `line_anchors` | How a line-edit tool names a line: `number` or `hash`. |
| `edit_check` | What a syntax error an edit introduces does: `reject`, `report` or `off`. |
| `constrain` | Tools to decode under a grammar. The model needs `grammar_tools`. |
| `force_tool_call` | Ask for a forced tool call. The model needs `forced_tools`. |
| `calling` | `native` (default) or `python`, for the worker only. |

### `[agents.profiles.<name>]`

A profile of your own, a narrowing of a built-in one. Zeus can spawn it.

| Key | Meaning |
|---|---|
| `base` | `reader`, `reviewer` or `researcher`. Required. |
| `rung` | Move it to another declared rung. |
| `tools` | Fewer tools than the base. |
| `max_tool_calls`, `max_cumulative_result_tokens`, `seconds`, `max_cost_cents`, `report_reasks` | Lower limits than the base. |

### `[templates]`

| Key | Meaning |
|---|---|
| `design`, `plan`, `brief` | Path to your own Markdown template for each document, replacing the built-in one. |

### Other keys

- `profiles`: `"auto"` (default) detects language profiles per file. Any other value names one.
- `repo_trust`: `"first_party"` (default) or `"third_party"`. `third_party` tightens the
  prompt-injection policy, but it is not this prototype's mode, so leave it alone.

## Limits

`[limits]` holds one flat key per bound, named `<scope>.<name>`. You set only the ones you want
to change, and a value that does not fit the role's model window is refused at load.

```toml
[limits]
"worker.max_tool_calls" = 150     # tool calls in one worker task
"worker.task_seconds" = 1200      # wall-clock bound on one task
"zeus.max_jobs" = 2               # background jobs Zeus may hold
```

Scopes, with the kinds of bound each holds and its default:

| Scope | Examples |
|---|---|
| `worker.*` | `max_tool_calls` 200, `task_seconds` 1800, `read_bytes` 16384, `read_lines` 400, `result_max` 8000 tokens, `keep_recent` 4 |
| `athena.*` | `max_tool_calls` 100, `max_recut_depth` 2, `one_task_max_tasks` 3, `note_max` 200 tokens |
| `zeus.*` | `max_tool_calls` 64, `inline_bytes` 4096, `max_jobs` 4, `job_max_runtime_s` 1800, `tell_text_max` 4000 chars |
| `critic.*`, `locate.*` | `max_tool_calls` 50 and 40 |
| `reader.*`, `reviewer.*`, `researcher.*` | `max_tool_calls` 30 to 40, `seconds` 300 to 600, `max_cost_cents` 50 |
| `agents.*` | `max_parallel` 3, `max_follow_ups` 3 |
| `hooks.*` | `max_continues` 1, `max_retries` 1, `max_steering` 3, `recheck_wait_s` 30 |
| `web.*` | `fetch_max_chars` 40000, `fetch_timeout_s` 15, `search_results` 5 |
| `turn.*` | `attachments_max` 8, `pdf_pages_max` 100, `text_file_max_bytes` 1 MB |
| `lsp.*` | `servers_max` 8, `idle_s` 600 |
| `skills.*` | `index_max_rows` 24, `find_results` 5 |
| `all.*` | `compact_trigger` 0.8, `compact_target` 0.5 (fractions of the window) |

To print every key with its default and unit:

```bash
uv run python -c "from chickadee.limits import LIMITS; [print(l.name, l.value, l.unit) for l in LIMITS]"
```

## Approval preferences

`approval.toml` records standing answers to permission questions. It is separate from the config.
There is one at `~/.config/chickadee/approval.toml` (yours) and one at
`<repo>/.chickadee/approval.toml` (the team's, committed). The two levels do not merge. They
compose, and the stricter answer wins.

```toml
[approval]
deny = ["**/.env", "secrets/**"]     # paths the model may never write or read through a tool

[approval.commands]                  # set a command class to allow, ask or deny
lockfile_install = "allow"
push = "ask"
```

Command classes: `read_only`, `worktree_write`, `tests`, `build`, `dev_server`,
`lockfile_install`, `external_fetch`, `push`, `destructive`, `credential`, `scope_expanding`.

- `deny` always tightens and applies at once.
- A repository's `allow` lowers a command's tier for everyone who clones it, so it stays inert
  until you accept it with `chickadee trust`.
- A destructive class can never be lowered or granted "always".
- A preference cannot switch off a check. A file that tries is refused.

## Command rows and hooks

**Command rows** tell the classifier how to treat a command it does not know. A row names the
command prefix, its class and where it goes:

```toml
[[commands]]
prefix = ["make", "deploy"]
cls = "push"
destination = "remote"        # remote | url_host | index_host | path_roots | none
```

A row can never make a shipped row less strict.

**Hooks** run your own script at a lifecycle point:

```toml
[[hooks]]
event = "after_tool"          # see the list below
name = "log-edits"
command = ["./scripts/log-edit.sh"]
timeout_s = 10                # default 30
on_error = "allow"            # allow | deny, what a failing hook means
```

Hook points: `session_start`, `prompt_submitted`, `before_request`, `before_tool`,
`after_tool`, `after_step`, `after_batch`, `before_stop`, `permission_asked`, `round_ended`,
`task_reviewed`, `task_committed`, `episode_ended`, `before_compact`, `job_exited`,
`session_end`. Other optional keys are `role`, `model`, `tool` and `paths` (to narrow when it
fires), and `enabled` (default `true`). `chickadee hooks` lists the points and what your config
binds at each.

## Environment variables

Any config key can be set for one run with a `CHICKADEE_` variable. A double underscore nests
a table, and a single underscore stays part of a key name. Values `true`, `false`, integers
and floats are read as such.

```bash
CHICKADEE_SPEND__HARD_COST_USD=2 uv run chickadee run --yes "add a retry to fetch()"
CHICKADEE_WORKSPACE__MODE=worktree uv run chickadee
```

The environment layer applies on top of your user file and the repository's. It can set your
spend cap, as the user file does.

## Check your config

The loader the engine uses can validate a file on its own. It names the key when something is
wrong:

```bash
uv run python -c "from chickadee.config import EngineConfig; EngineConfig.from_toml('$HOME/.config/chickadee/config.toml')"
```

`chickadee setup` also reports the cap in force and whether the sandbox can run.
