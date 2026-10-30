"""Hooks: points at the run's major events, what a hook may answer there, and the two
kinds of hook, the harness's own built-ins and the person's scripts.

One home per concept, Gemini CLI's split:

- `points.py`: the sixteen points, each one's level and the verdicts it takes;
- `verdicts.py`: the eight verdicts and the one way several combine;
- `payloads.py`: what a hook is told at each point, one typed model per point;
- `match.py`: whether a configured hook is about an event;
- `fire.py`: `Hooks`, what every fire site holds, and the `hook` lines it writes;
- `rule.py`: the tool points as dispatcher rules, never a second dispatcher;
- `stop.py`: the one stop helper every role's loop calls;
- `bind.py`: a configured set checked and fixed for an episode or a session;
- `builtins.py`: the harness's own hooks, by name;
- `protocol.py` and `script.py`: a script's input, output and run.

Nothing is imported here, so `config.py` can read `points.py` without the rest.
"""
