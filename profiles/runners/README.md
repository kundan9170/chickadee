# Test runners

One file per test runner: how its output says what ran (D236 Amendment 1). A run is read by
what it printed, whatever ran it and whatever its exit code (`executor/tests.read_test_run`). Kept
once per runner, not per language: jest serves JavaScript and TypeScript, Maven Java and Kotlin.

- `[[summary]]`: each kind of summary line the runner prints. `line` finds one (every match
  counts, summed: cargo prints one per test binary, go one per package). `passed`, `failed`,
  `errors`, `skipped` and `total` are patterns applied within that line: one with a group sums
  its numbers, one without counts its matches. With no `passed`, it is `total` less the rest.
- `failing`, `passing`: the test names the output prints, the first group of each match.
- `broken`: with no summary, a line saying the suite could not run (a compile error).
- `unit`: what the counts count, `test` or `package`.

Every pattern is a Python regex in multi-line mode, checked at load. Each runner's patterns are
tested against output captured from the real runner (`tests/executor/fixtures/runs/`).
