Answer the open task's report. `task` (its id) and `answer` are required; `text` is the agent's next message for `feedback`, and your reason for any other answer. `fresh` is optional, on `feedback`, `take_over` and `re_cut`: leave it out to keep the changes, `true` puts every file back to where the task started, and a list of paths puts back only those.

- `accept`: the task is committed, and you move on. `{"task": "T1", "answer": "accept"}`
- `feedback`: your text is the agent's next message in the same conversation; the answer is its next report. `{"task": "T1", "answer": "feedback", "text": "parse() drops the last field; fix it and add a test"}`, or with `"fresh": ["src/parse.py"]` to put that file back first.
- `continue`: only after a round held for a message. The agent goes on where it was held, with no message from you, and the round has what the held one had left of its calls and time; the answer is its next report. `{"task": "T1", "answer": "continue"}`. Use `feedback` instead when the message changes what the task does.
- `take_over`: the task is yours, to polish it or to do it yourself. Your edit tools, and your shell where you have one, work on it, every change recorded under it; then review it again, `accept` or `drop`. `{"task": "T1", "answer": "take_over", "text": "two lines left", "fresh": true}` starts it over yourself.
- `re_cut`: the task is replaced by the ones you delegate next; revise the plan to show them. `{"task": "T1", "answer": "re_cut", "text": "too big", "fresh": true}`. A task re-cut too many times is refused: take it over, give feedback, or drop it.
- `drop`: the task is given up and the tree goes back to where it started; carry on with your plan. `{"task": "T1", "answer": "drop", "text": "the wrong approach"}`

Read the facts, not only the report: what actually changed and what actually ran.
