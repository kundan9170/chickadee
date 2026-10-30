Continue an agent you spawned and get a new report. `agent` is the id its report opens with
(`A1`); `message` is what you want from it now.

The agent keeps its conversation: everything it read and reported is still in its context, so
ask for what is missing, or for the next step, without repeating the task. It runs on the same
profile and tools, with a round's full limits, and its report is checked as the first was. A
`partial` or `cancelled` agent can be followed up. Each agent takes a few follow-ups, one per
response; past them, or when its conversation would no longer fit, you are told so: spawn a
fresh agent and give it what it needs from the old report.
