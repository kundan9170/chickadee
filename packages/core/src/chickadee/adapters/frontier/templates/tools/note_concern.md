Record an anchored advisory that outlives this turn. `text` and a non-empty `anchor` are required.

Appends one typed line to the session log: something you noticed that is not this
turn's job but should not be lost — a fragile test, a contract about to drift, a path that
needs an owner. `anchor` is the paths it is about, and it must be non-empty: an advisory
naming no path can never surface. The refusal is `INVALID_ARGUMENTS`, not prose.

Nothing here binds anything; it is a record. The next session's context sees it.
