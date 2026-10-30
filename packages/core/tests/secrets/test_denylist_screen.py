"""The denylist's screen changes no answer: over a corpus of names and lists, the
screened verdict is the exact one, the same denial or none, for files and directories alike.
The screen exists for speed (`resolve` walks `$HOME`); this is what lets it."""

from __future__ import annotations

import itertools
from pathlib import PurePosixPath

import pytest

import chickadee.secrets.denylist as denylist_module
from chickadee.paths import GlobSet, glob_match
from chickadee.secrets.denylist import Denylist

SEGMENTS = (
    "src",
    ".ssh",
    ".SSH",
    "id_rsa",
    "id_rsa.pub",
    ".env",
    ".env.example",
    ".env.local",
    "key.pem",
    "KEY.PEM",
    ".docker",
    "config.json",
    ".aws",
    "credentials",
    "secrets",
    "my-secret.yaml",
    "a.key",
    "b.template",
    "service-account-1.json",
    ".gnupg",
    "x",
)

LISTS = (
    Denylist(),
    Denylist(patterns=(*Denylist().patterns, "*secret*", "secrets/", "/x/a.key", "[a-c].key")),
    Denylist(patterns=(*Denylist().patterns, "src/**", "**/config.json", "x/")),
)


def _tails(path: str) -> list[str]:
    parts = [p for p in PurePosixPath(path).parts if p != "/"]
    return ["/".join(parts[i:]) for i in range(len(parts))]


def _corpus() -> list[str]:
    paths: list[str] = []
    for depth in (1, 2, 3):
        paths += ["/".join(c) for c in itertools.product(SEGMENTS[:12], repeat=depth)]
    paths += ["/".join(c) for c in itertools.permutations(SEGMENTS[12:], 3)]
    return paths


@pytest.mark.parametrize("listed", LISTS, ids=["defaults", "with-project", "anchored"])
def test_the_screened_verdict_is_the_exact_one(listed: Denylist) -> None:
    exact = denylist_module._exact_denial  # pyright: ignore[reportPrivateUsage]
    screened = denylist_module._first_denial  # pyright: ignore[reportPrivateUsage]
    denied = 0
    for path in _corpus():
        tails = _tails(path)
        for is_dir in (False, True):
            want = exact(listed, tails, is_dir=is_dir)
            assert screened(listed, tails, is_dir=is_dir) == want, (path, is_dir)
            denied += want is not None
    assert denied > 100, "the corpus exercises denials, not only clearances"


def test_a_glob_set_matches_what_one_of_its_globs_matches() -> None:
    globs = ("*.pem", "src/**", "a/**/b", "docs/")
    names = ["k.pem", "x/k.pem", "src/a", "src", "a/b", "a/x/y/b", "docs/r.md", "other"]
    for name in names:
        assert GlobSet(globs).matches(name) == (glob_match(name, globs) is not None), name
    assert GlobSet(("src/**",), any_tail=True).matches("deep/in/src/a")
    assert not GlobSet(("src/**",)).matches("deep/in/src/a")
    assert not GlobSet(()).matches("anything")


ODD = (
    "/home/u/.ssh/id_rsa",
    "//.ssh/key",
    "a//b/.env",
    "./.env",
    "x/../.env",
    "x/./.docker/config.json",
    ".env\x00x",
    "src/ok\x00/.ssh",
    "/",
    "",
    ".",
    "KEY.PEM/",
)


@pytest.mark.parametrize("listed", LISTS, ids=["defaults", "with-project", "anchored"])
def test_a_name_is_judged_as_the_exact_rule_judges_it(listed: Denylist) -> None:
    """`name_denied` screens the spelling before it builds tails: every spelling, however
    written, gets the answer the exact rule gives its tails."""
    exact = denylist_module._exact_denial  # pyright: ignore[reportPrivateUsage]

    def reference(path: str, is_dir: bool) -> object:
        spellings = (path, path.split("\x00", 1)[0]) if "\x00" in path else (path,)
        for spelling in spellings:
            denial = exact(listed, _tails(spelling), is_dir=is_dir)
            if denial is not None:
                return denial
        return None

    for path in [*ODD, *_corpus()[:3000]]:
        for is_dir in (False, True):
            got = denylist_module.name_denied(path, listed, is_dir=is_dir)
            assert got == reference(path, is_dir), (path, is_dir)
