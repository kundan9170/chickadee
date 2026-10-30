"""Which proxy a fetch goes through, read from the environment as curl and `httpx` read it:
`<scheme>_proxy` or `all_proxy`, in either case, less the hosts `no_proxy` names. The one home
of that reading, for both engines: the `http` engine connects to the checked address only when
there is none, and `lightpanda` is handed it as `--http-proxy`."""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["proxy_for"]


def proxy_for(scheme: str, host: str, env: Mapping[str, str]) -> str | None:
    """The proxy URL a request to `host` over `scheme` goes through, or None."""
    found = {k.lower(): v for k, v in env.items() if k.lower().endswith("_proxy")}
    proxy = found.get(f"{scheme}_proxy") or found.get("all_proxy")
    if not proxy or _bypassed(host, found.get("no_proxy", "")):
        return None
    return proxy


def _bypassed(host: str, no_proxy: str) -> bool:
    """Whether `no_proxy` names `host`: `*`, the host, or a domain it is under. A port on an
    entry is ignored."""
    for raw in no_proxy.split(","):
        entry = raw.strip().lower().lstrip(".")
        entry = entry.rsplit(":", 1)[0] if entry.count(":") == 1 else entry
        if entry == "*" or (entry and (host == entry or host.endswith(f".{entry}"))):
            return True
    return False
