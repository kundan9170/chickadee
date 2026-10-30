"""The calls that leave the machine on a model's behalf.

`policy` is the one checkpoint every such call passes: classified, redacted for its record,
recorded, its answer framed as untrusted. `urls` holds `web_fetch`'s guards, `fetch` its
engines, and `web` the `web` port that puts them together for one agent round. The MCP client
is the policy's second user.
"""
