# Contributing

Thanks for taking a look. Chickadee is a prototype, so small, focused changes are easiest to
review.

## Setup

You need Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kundan9170/chickadee.git
cd chickadee
uv sync --all-packages
```

## The four checks

CI runs these. Please run them before you open a pull request:

```bash
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run pyright                 # types, strict on packages/core
uv run pytest                  # the test suite
```

The suite runs in parallel. `uv run pytest -n 0` runs it serially, which is easier when
debugging one test. Some tests need bubblewrap, the Polyglot corpus or network access and skip
themselves when they are unavailable. `pytest -rs` lists what was skipped and why.

## Pull requests

- Keep a change to one purpose. Say what it does and why.
- Add or update tests for behaviour you change. Do not weaken or delete a test to make a
  change pass.
- Comments explain why, not what. Match the surrounding style.
- If a change touches the model-facing prompts or tool behaviour, say how you checked it.
- Open an issue first for anything large.

## Design rules to keep

- Models never write or run anything directly: edits go through the edit port, commands through
  the executor.
- One concept has one home. Search before adding a second copy.
- Durable writes are crash-safe and journaled.
