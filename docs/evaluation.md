# Evaluation

Chickadee rests on a cost hypothesis: that a frontier model deciding and a cheaper model
typing can solve nearly as many tasks as a conventional frontier-only harness, for much less
money. This is a hypothesis being measured. No result is claimed here.

## Experiments

| Experiment | Question | Status |
|---|---|---|
| **E1: reliability** | Given bounded, self-contained exercises, how reliably does the configured worker finish them? A diagnostic only. It cannot support a claim about economics. | Runs on a pinned subset of the Polyglot benchmark (34 Python and 49 JavaScript exercises) |
| **E2: product economics** | The headline. The same pinned frontier model in a conventional API harness versus Chickadee with a frontier and a cheaper model. | Defined, not yet run |
| **E3: frontier spend breakdown** | Where the frontier model's money goes, by stage of the lifecycle. | Defined, not yet run |

**The E2 pass condition** is no more than 5 percentage points of solve-rate loss at no more
than 50% of the cost, under paired task-level inference, declared repeats, and counting the
spend on failed tasks. Local models are priced at a conservative hosted-equivalent rate, so
running locally does not make the comparison look better than it is.

The rig reports failures, not only scores: failure attribution, reject reasons, before and
after diffs, and a reproducible manifest for every run.

## Running the rig

The rig lives in `packages/eval` and installs the `chickadee-eval` command.

```bash
git clone https://github.com/Aider-AI/polyglot-benchmark.git
export CKD_POLYGLOT_ROOT=$PWD/polyglot-benchmark
uv run chickadee-eval list --problems        # the committed subsets
uv run chickadee-eval run --subset e1-dev    # run a subset into a new archive
uv run chickadee-eval diff <archive-a> <archive-b>
```

Each run is written to an archive under `$CKD_EVAL_HOME`. `diff` compares two archives, which
is how a change to a prompt or a tool is checked before and after. The tests that read the real
corpus skip themselves when `CKD_POLYGLOT_ROOT` is not set.
