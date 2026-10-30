"""Every body a client sends and every answer it reads is a declared type.
A typed request a client builds from `models.py` is accepted by the server's own facade type,
whose validators are the server's; and `locate`'s two answers decode as themselves."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from chickadee.box.request import ApprovedDesign as CoreApprovedDesign
from chickadee.box.request import CoderRequest as CoreCoderRequest
from chickadee.box.request import OneTask as CoreOneTask
from chickadee.schema import sha256_hex
from chickadee_serve.bind import Uds
from chickadee_wire import calls
from chickadee_wire.client import Client
from chickadee_wire.models import (
    ApprovedDesign,
    CitationGateP,
    CoderRequest,
    Findings,
    OneTask,
    Unresolved,
)

ADDRESS = Uds(Path("/nonexistent/serve.sock"))


def _sent(call: calls.RouteCall[object]) -> dict[str, object]:
    raw = call.content()
    assert raw is not None
    return json.loads(raw)["request"]


BRIEF_SHA = sha256_hex("retry the fetch once")
DESIGN_SHA = sha256_hex("# Retry the fetcher\n")
TASK = OneTask(
    title="add a retry", brief="retry the fetch once", sha=BRIEF_SHA, template_sha="b" * 64
)
DESIGN = ApprovedDesign(text="# Retry the fetcher\n", sha=DESIGN_SHA, template_sha="d" * 64)


def test_a_typed_request_is_the_servers_request() -> None:
    one_task = CoreCoderRequest.model_validate(_sent(calls.coder("s", CoderRequest(task=TASK))))
    assert one_task.task == CoreOneTask(
        title="add a retry", brief="retry the fetch once", sha=BRIEF_SHA, template_sha="b" * 64
    )
    assert one_task.workspace == "worktree"
    design = CoreCoderRequest.model_validate(
        _sent(calls.coder("s", CoderRequest(design=DESIGN, workspace="in_place")))
    )
    assert design.design == CoreApprovedDesign(
        text="# Retry the fetcher\n", sha=DESIGN_SHA, template_sha="d" * 64
    )
    assert design.workspace == "in_place"


def test_the_servers_rules_still_judge_a_typed_request() -> None:
    """The generated type carries the fields, not the validators: a request naming no work,
    or both kinds, is built by the client and refused by the server, as a hand-written body
    would be."""
    CoreCoderRequest.model_validate(CoderRequest(task=TASK).model_dump(mode="json"))
    for wrong in (CoderRequest(), CoderRequest(task=TASK, design=DESIGN)):
        with pytest.raises(ValidationError, match="exactly one of design or task"):
            CoreCoderRequest.model_validate(wrong.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("answer", "kind"),
    [
        ({"targets": [], "facts": ["a fact"], "truncated": False}, Findings),
        ({"question": "which fetcher?", "candidates": ["a.py", "b.py"]}, Unresolved),
    ],
)
def test_locate_answers_decode_as_themselves(answer: dict[str, object], kind: type) -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, json=answer))
    with Client(ADDRESS, "t", transport=transport) as client:
        assert isinstance(client.locate("where?"), kind)


def test_a_body_is_sent_by_its_wire_names() -> None:
    """A field the wire names with a Python keyword (`pass`) is sent as the wire spells it."""
    gate = CitationGateP.model_validate({"checked": 1, "coverage": 1.0, "pass": True})
    sent = calls._sent(gate)  # pyright: ignore[reportPrivateUsage]
    assert sent["pass"] is True and "pass_" not in sent


def test_the_server_refuses_a_text_that_is_not_what_its_sha_names() -> None:
    """Invariant 15 at the seam: a host's request whose text was changed after its stop is
    refused, so the journal never names an approval the run did not run."""
    body = CoderRequest(task=TASK).model_dump(mode="json")
    body["task"]["brief"] = "retry the fetch twice"
    with pytest.raises(ValidationError, match="not what was approved"):
        CoreCoderRequest.model_validate(body)
