# residue

You are shown part of a conversation that is about to leave the model's context, and the
record already made of it. Your job is ONLY to name what the conversation holds that the
record does not: a decision the person stated, a preference, a fact they gave, a question
left open. Never summarise the turns, never restate the record, and never add anything the
turns do not say.

Rules:

- One item per line, at most {max_items} lines. No bullets, numbers or headings.
- Each item stands on its own: name the thing, not "as discussed".
- If the record already holds it, leave it out.
- If nothing is left over, answer with nothing at all.

## THE RECORD ALREADY MADE (an exclusion list, not material)

{exclusion}

## TURNS {first_seq} TO {last_seq}

{turns}
