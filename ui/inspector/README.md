# Chickadee Inspector

A local web page that shows what each model was sent and what it said back, live while a run
goes: the system prompt each role got, every message in each request, each reply, each tool
call's arguments, and the full result the model was given back. Problems (a refused answer, a
refused tool call, a failed command, a task that did not finish) are marked red and can be
filtered to.

It only reads what the harness already records. It never writes.

| Tab | Source | What you see |
|---|---|---|
| Athena, Worker T1, T2… | the run journal (`box_message`, `model_call`, `tool_call`) and its blobs | everything, in full |
| Zeus (a session) | the session log (`prefix`, `turn`, `tool`, `spend`, `episode`) and its blobs | everything: the system message and tool list are rebuilt from the `prefix` line's part blobs (D223) |

A session begun before D223 has no part blobs until its next turn; the page then shows the
`zeus.md` template on disk and says so.

Each Athena and worker call carries the request exactly as the provider got it (D224): every
message, the tool schemas, the settings, and whether a tool call was forced. When a client
rewrote the request before sending (PR 641's Python calls do), the call is marked "differs
from the conversation" and the rewritten request is the one shown. A Python block shows as its
code, and the tool calls it made are listed under its result, with the journal's bounded
previews of their arguments and results. A run from before D224 shows the request rebuilt from
the conversation, without tools or settings.

## Run it

Once, after pulling (and again when `ui/inspector/src` changes):

```sh
uv sync --all-packages
cd ui/inspector && npm install && npm run build && cd ../..
```

Then, from inside the project you run chickadee on:

```sh
chickadee inspect                 # reads this project; open http://127.0.0.1:8765
chickadee inspect --repo <path>   # another project
chickadee inspect --state <dir>   # a state directory holding runs/ and sessions/ directly
chickadee inspect --port 9000
```

It is read-only, so it is safe beside a running session. The page refreshes by itself.

To work on the page itself: keep `chickadee inspect` running and use `npm run dev` here (hot
reload, `/api` forwarded to port 8765).

## Layout

- `chickadee.inspect` (core): the read-only reader, `Inspection`, over the harness's own
  readers (`journal.reader`, `context.store.SessionLog`); reached through
  `chickadee.engine.inspect_runs`.
- `chickadee_cli/inspector.py`: `chickadee inspect`, serving this page and the JSON it polls:
  `/api/index`, `/api/runs/<id>?after=<seq>`, `/api/sessions/<id>?after=<seq>`,
  `/api/zeus-template`. Only new lines travel.
- `src/model.js`: groups lines into one conversation per role and joins each call to its reply
  and each tool result to its call.
- `src/App.jsx`, `src/cards.jsx`, `src/parts.jsx`: the views.
