"""State layout, configuration and trust.

One document owning what three findings each assumed someone else owned: where state lives,
how configuration layers merge, and what may run without being accepted.
"""

from chickadee.state.layers import CONFIG_NAME, ENV_PREFIX, Layer, LayerReadError, load_layers
from chickadee.state.merge import LayerProvenance, MergeError, UnknownKey, merge, merge_arrays
from chickadee.state.resolve import Resolved, resolve
from chickadee.state.roots import (
    LOCAL_DIR,
    PROJECT_DIR,
    USER_DIR,
    NotARepository,
    Roots,
    roots,
    xdg_config_home,
)
from chickadee.state.trust import TRUST_FILE, acceptance_record, executable_subset, is_active

__all__ = [
    "CONFIG_NAME",
    "ENV_PREFIX",
    "LOCAL_DIR",
    "PROJECT_DIR",
    "TRUST_FILE",
    "USER_DIR",
    "Layer",
    "LayerProvenance",
    "LayerReadError",
    "MergeError",
    "NotARepository",
    "Resolved",
    "Roots",
    "UnknownKey",
    "acceptance_record",
    "executable_subset",
    "is_active",
    "load_layers",
    "merge",
    "merge_arrays",
    "resolve",
    "roots",
    "xdg_config_home",
]
