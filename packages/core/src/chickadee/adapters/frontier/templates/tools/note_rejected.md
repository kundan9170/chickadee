Record an alternative that was considered and declined, with why. `text` and a non-empty `reason` are required.

Appends one typed line to the session log: the option you did not take and the reason. A
rejection without a reason is a preference stated as history, so an empty `reason` is
`INVALID_ARGUMENTS`. `recall` will find it when the same question comes back.
