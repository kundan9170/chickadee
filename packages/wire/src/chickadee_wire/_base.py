"""The base of every generated wire type (`models.py`)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class WireModel(BaseModel):
    """Frozen, and ignoring a field it does not declare: a minor wire step may add one,
    and a client one minor behind must still read the answer."""

    model_config = ConfigDict(frozen=True, extra="ignore")
