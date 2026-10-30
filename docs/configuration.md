# Configuration

Chickadee reads one engine config, in your home directory:

```text
~/.config/chickadee/config.toml
```

`chickadee setup` writes a commented starting file there if none exists. It never overwrites
an existing one. A full working example is in [`examples/config.toml`](../examples/config.toml).

Models, prices and spend caps live in this file and nowhere else. A repository's own config
can tune a few settings but can never repoint a rung, set a price or raise a spend cap.

## The example

```toml
[spend]
hard_cost_usd = 5.00              # required, finite
warning_fractions = [0.5, 0.8]

[spend.rungs]                     # optional caps beneath the total
high-param = 3.00

[models.frontier]                 # any OpenAI-compatible endpoint
base_url = "https://api.example.com/v1"
model_id = "your-frontier-model"
api_key_env = "FRONTIER_API_KEY"  # the key is read from the environment
context_window = 200000
max_output_tokens = 32000

[models.local]
base_url = "http://localhost:8000/v1"
model_id = "your-local-coder"
context_window = 131072
max_output_tokens = 16384

[prices.frontier]                 # USD per million tokens
price_in = 3.00
price_out = 15.00

[prices.local]                    # price local models at a hosted-market equivalent
price_in = 0.20
price_out = 0.60

[rungs]
high-param = "frontier"           # Zeus and Athena
mid-param = "frontier"            # locate, critic, reviewer, helpers
low-param = "local"               # the worker
```

## Models and rungs

`[models.<name>]` is the catalog: every model you can run. `[rungs]` maps each rung to a model
from it. Each role has a default rung (see [architecture](architecture.md)): Zeus and Athena
use `high-param`, the worker uses `low-param`, and helpers use `mid-param`. One model may serve
several rungs. A model needs a `[prices.<name>]` row before a call to it can be charged.

You can declare your own rungs by naming them in `[rungs]` and move a role onto one under
`[roles]`.

## Spend

`hard_cost_usd` is required and must be finite. A run that reaches it stops with a resumable
terminal state. `warning_fractions` set the warning points. `[spend.rungs]` caps a single rung.
The `/budget` command shows spend against the cap and lets you raise or lift it for a session.

## Checking your file

The loader is the same one the engine uses. To check a file without starting a session:

```bash
uv run python -c "from chickadee.config import EngineConfig; EngineConfig.from_toml('examples/config.toml')"
```

It raises an error naming the key when something is wrong.

## Other settings

The config also has tables for the sandbox, the work folder, tool choices per role, command
rows and limits. The commented file written by `chickadee setup` is the place to start, and
`chickadee --help` lists the commands that read and write them.
