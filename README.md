<p align="center">
  <img alt="Chickadee" src="docs/assets/logo.png" width="520">
</p>

<p align="center">
  <b>A coding assistant where an expensive model decides, a cheap model types,<br>
  and plain code records and checks everything.</b>
</p>

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB">
  <img alt="Status: pre-alpha" src="https://img.shields.io/badge/status-pre--alpha-orange">
  <img alt="License: proprietary" src="https://img.shields.io/badge/license-proprietary-lightgrey">
</p>

<p align="center">
  <a href="#getting-started">Getting started</a> ·
  <a href="docs/configuration.md">Configuration</a> ·
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <img alt="Chickadee in the middle of a run: the designer's report on the left, the live diff in the centre, the plan and the worker's progress on the right" src="docs/assets/screenshot-run.webp" width="100%">
</p>

---

## Why Chickadee

Most coding agents run one frontier model for everything: reading, planning, typing out code and
re-running tests. That is expensive, and most of those tokens are not hard thinking.

Chickadee splits the work by difficulty:

- **A frontier model designs.** You talk to the **designer**, which reads your repository, asks
  what it needs to and writes a design. The designer never edits a file.
- **A planner cuts the work.** The **planner** turns the approved design into a short plan of tasks,
  and reviews each result.
- **A small model builds.** The **worker**, which can be a cheap or local model, does each task:
  it reads code, edits files and runs tests in an isolated copy of your repository.
- **Plain code records and guards everything.** Every edit is snapshotted and every command is
  sandboxed. Every effect is written to a journal before it happens, so a run can be stopped,
  inspected, undone and resumed.

**The bet:** solve nearly as many tasks as a frontier-only harness (at most 5 percentage points
fewer) for at most half the cost. That is a hypothesis under test, not a result.

<p align="center">
  <img alt="The Chickadee home screen: the designer pane, recent sessions, and the planner and worker panes waiting" src="docs/assets/screenshot-home.webp" width="100%">
  <br><sub>The home screen. The designer, planner and worker each have a pane, and the header shows the models in use and the session's spend against its cap.</sub>
</p>

## Getting started

> [!NOTE]
> Chickadee is pre-alpha. It runs, but expect rough edges and breaking changes.

You need **Python 3.12+**, **[uv](https://docs.astral.sh/uv/)**, **git**, a sandbox
([bubblewrap](https://github.com/containers/bubblewrap) on Linux; built in on macOS) and at
least one OpenAI-compatible model endpoint.

```bash
git clone https://github.com/kundan9170/chickadee.git
cd chickadee
uv sync --all-packages
uv run chickadee setup     # writes ~/.config/chickadee/config.toml and checks the sandbox
```

Add your models to `~/.config/chickadee/config.toml`, then, in the repository you want to work
on:

```bash
uv run --project ~/path/to/chickadee chickadee init   # index the repository
uv run --project ~/path/to/chickadee chickadee tui    # open the interface
```

**[The configuration guide](docs/configuration.md)** covers every option with its default, and
[`examples/`](examples/) has a minimal config and a complete one to start from.

## Documentation

| Start with | For |
|---|---|
| [`docs/configuration.md`](docs/configuration.md) | Installing, configuring models and roles, every option |
| [`docs/architecture.md`](docs/architecture.md) | The shape of the system, its roles and the task loop |
| [`docs/safety.md`](docs/safety.md) | What the harness guarantees, and its known limits |
| [`docs/evaluation.md`](docs/evaluation.md) | How the cost and solve-rate bet is being measured |

## Repository layout

```text
packages/
  core/       chickadee: the engine (journal, edit port, executor, sandbox, context, graph, roles)
  cli/        chickadee_cli: the command line and the TUI
  serve/      the engine server every frontend talks to, over one socket
  wire/       the one client every frontend uses
  eval/       chickadee_eval: the E1/E2/E3 measurement rig
  testing/    shared test helpers: engines, repositories, sandboxes
docs/         architecture, safety, configuration, evaluation and images
examples/     a minimal config and a complete one
profiles/     per-language profiles: test runners and conventions
ui/           the inspector
scripts/      the repository's checks
```

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) explains how to set up, the checks CI runs and the design
rules every change keeps. Report security problems privately, as [`SECURITY.md`](SECURITY.md)
describes.

## License

Copyright © 2026 Kundan Kumar. All rights reserved. Chickadee is proprietary software. The source
is visible here for review, but no licence to use, copy, modify or distribute it is granted. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for third-party material.
