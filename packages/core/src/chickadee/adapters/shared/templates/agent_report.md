## Your report

When you are done, answer with your report as your last message, with no tool call, in exactly
these sections, in this order:

{{sections}}

- `## Summary`: the answer to the task, in a few sentences.
{{^web}}
- `## Findings`: one claim per bullet, each citing the lines it rests on, as
  `path/to/file.py:40-52`. Write `- none` if there are none.
{{/web}}
{{#web}}
- `## Findings`: one claim per bullet, each citing the page it rests on and the words on it
  that say so, as `https://example.org/page "the exact words, copied"`. Write `- none` if there
  are none.
{{/web}}
- `## Not done`: what the task asked that you did not do or could not find, and why; `nothing` if
  you did it all.
