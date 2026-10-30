"""A route over a facade member that is missing, or marked `@not_built`, answers a typed 501
naming why, and `GET /` leaves its capability out until it lands."""

from __future__ import annotations

from starlette.testclient import TestClient

from chickadee.unbuilt import not_built, owner_of
from chickadee_serve.app import Served, build_app, capabilities
from chickadee_serve.auth import TokenClass, TokenStore


class _Facade:
    def budget(self, session_id: str) -> dict[str, str]:
        return {"session": session_id}

    @not_built("T-4.10")
    def handoff(self, session_id: str) -> object:
        raise AssertionError("never run")


def _client() -> tuple[TestClient, dict[str, str], Served]:
    tokens = TokenStore()
    served = Served(_Facade(), tokens)
    auth = {"authorization": f"Bearer {tokens.issue(TokenClass.INTERACTIVE)}"}
    return TestClient(build_app(served)), auth, served


def test_a_built_member_answers() -> None:
    client, auth, _ = _client()
    assert client.get("/session/s1/budget", headers=auth).json() == {"session": "s1"}


def test_a_marked_member_is_501_naming_its_task() -> None:
    client, auth, _ = _client()
    r = client.get("/session/s1/handoff", headers=auth)
    assert r.status_code == 501
    assert r.json() == {
        "code": "NOT_BUILT",
        "message": "handoff is not built yet; it arrives with T-4.10",
    }


def test_a_missing_member_is_501() -> None:
    client, auth, _ = _client()
    r = client.get("/session/s1/liveness", headers=auth)
    assert r.status_code == 501 and r.json()["code"] == "NOT_BUILT"


def test_hello_lists_only_what_answers() -> None:
    _, _, served = _client()
    caps = capabilities(served)
    assert "budget" in caps and "questions" in caps and "wire" in caps
    assert "handoff" not in caps and "liveness" not in caps and "replay" not in caps


def test_owner_of_reads_the_mark() -> None:
    assert owner_of(_Facade.handoff) == "T-4.10"
    assert owner_of(_Facade.budget) is None
