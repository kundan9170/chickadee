You check a research report against an answer key. You are not asked whether the report is good, only whether it states each fact of the key.

For each numbered fact, decide one verdict:
- `stated`: the report says it, in any words, without contradicting itself about it.
- `contradicted`: the report says something that cannot be true if the fact is.
- `missing`: the report says neither.

Judge only by what the report says. Do not use what you know about the subject, and do not give credit for a fact the report only hints at.

Answer with one JSON object and nothing else:
{"facts": [{"n": 1, "verdict": "stated", "why": "one short sentence quoting or pointing at the report"}, ...]}
with one entry for every fact, in order.

## The question
{{question}}

## The answer key
{{facts}}

## The report
{{report}}
