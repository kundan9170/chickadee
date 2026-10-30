"""The frontier prompt set: Zeus's, Athena's tools' and the box tools' templates.

A role's `family` names only its prompt set since the adapters retired: the
directory a template is read from. The roles tied to no model family read `adapters/shared`.
"""

from pathlib import Path
from typing import Final

__all__ = ["TEMPLATES"]

TEMPLATES: Final = Path(__file__).resolve().parent / "templates"
