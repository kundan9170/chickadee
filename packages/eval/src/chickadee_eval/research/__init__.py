"""The researcher's evaluation.

About twenty real research questions (`manifests/research.toml`), each asked of one researcher
round through the session's own agents (`PacketEngine.spawn_agent`), so what is measured is
the wiring Zeus spawns, without Zeus. Each answer is scored three ways:

1. **Citation accuracy**, by the harness's own check: how many of the
   report's quotes were on the pages the round fetched, how many were flagged, how many claims
   cited anything. Code, no model.
2. **Factual accuracy**, by a judge: each fact of the question's answer key, a person's, marked
   stated, contradicted or missing in the report. The judge is a model of the eval's own
   choosing (`--judge`), never the one the researcher ran on.
3. **Source quality**: the share of the report's web citations on the question's preferred
   domains (its project's own documentation, its tracker). Code, no model.

A run archives beside `chickadee-eval run`'s (`archive.py`), its kind `research`, so
`chickadee-eval diff` compares two of them. The web changes between runs, so each question is
asked `--repeats` times and the report says how far the repeats disagree.
"""
