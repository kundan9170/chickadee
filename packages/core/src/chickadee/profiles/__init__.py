"""LanguageProfile: format, loader, resolution.

Adding a language should be a new file in `profiles/`, not a code change here.
"""

from __future__ import annotations

from chickadee.profiles.loader import (
    BUILTIN_DIR,
    CONVENTIONS_OVERRIDE,
    OVERRIDE_SUBDIR,
    ProfileSet,
    deep_merge,
    load_profile,
    load_runners,
    resolve_conventions,
    resolve_document,
    shipped,
    shipped_runners,
)
from chickadee.profiles.model import (
    BuildSpec,
    ConventionsSpec,
    DepsSpec,
    ExamplesSpec,
    LanguageProfile,
    LintSpec,
    LspSpec,
    ProfileError,
    SyntaxSpec,
    Template,
    TestRunner,
    TestSpec,
)
from chickadee.profiles.resolve import FileLanguage, Resolution, resolve

__all__ = [
    "BUILTIN_DIR",
    "CONVENTIONS_OVERRIDE",
    "OVERRIDE_SUBDIR",
    "BuildSpec",
    "ConventionsSpec",
    "DepsSpec",
    "ExamplesSpec",
    "FileLanguage",
    "LanguageProfile",
    "LintSpec",
    "LspSpec",
    "ProfileError",
    "ProfileSet",
    "Resolution",
    "SyntaxSpec",
    "Template",
    "TestRunner",
    "TestSpec",
    "deep_merge",
    "load_profile",
    "load_runners",
    "resolve",
    "resolve_conventions",
    "resolve_document",
    "shipped",
    "shipped_runners",
]
