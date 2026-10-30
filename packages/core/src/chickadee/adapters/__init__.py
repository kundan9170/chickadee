"""The model families' prompts and the reads every role's composition shares (`base`).

The family registry and its two adapters retired with the packet loop: no role
parses a model's reply as a patch any more, so no family needs an adapter. `frontier/templates`
holds the frontier roles' prompts and every tool's description; `shared/templates` the prompts
tied to no model family.
"""
