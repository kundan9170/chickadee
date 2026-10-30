# Security

## Reporting a vulnerability

Please report security problems privately, by email to kundan91700@gmail.com. Do not open a
public issue for a vulnerability. Include what you found, how to reproduce it, and the impact
you see. You can expect a first reply within a few days.

## Scope

Chickadee is a prototype for personal use. It is not hardened against prompt injection from
untrusted content, and it does not provide scoped credentials or strong isolation for untrusted
repositories or multiple users. See [docs/safety.md](docs/safety.md) for what it does guarantee.
Reports about those known limits are welcome, but they are not surprises.
