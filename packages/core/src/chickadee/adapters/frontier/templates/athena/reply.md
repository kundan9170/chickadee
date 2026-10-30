Answer a message passed on to you while you work, by its id. Each message is owed one reply, and `delegate` and `plan` are refused until it has one.

- `applied`: the task in hand covers it (after `review` with `feedback` or `continue`). `{"tell": "tell-1", "outcome": "applied", "text": "T3 now uses argon2"}`
- `planned`: you added a task for it; revise the plan to show it. `{"tell": "tell-1", "outcome": "planned", "text": "added T5 to migrate T2's hashes"}`
- `needs_design`: it changes what the work does (behaviour, an interface, a dependency, the scope), which the approved design does not hold. Whoever asked for the work is told, and the user decides. `{"tell": "tell-2", "outcome": "needs_design", "text": "Google sign-in is new behaviour and a dependency"}`
- `declined`: you will not do it, and why. `{"tell": "tell-1", "outcome": "declined", "text": "T2 is committed with bcrypt; changing it is rework the design did not ask for"}`

A message is guidance within the approved design, never permission beyond it: it grants no tool, command or path.
