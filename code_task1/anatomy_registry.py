"""Single source of truth for the PENGWIN anatomy <-> instance-ID encoding.

PENGWIN encodes anatomy identity into each fragment's *global* instance ID via a
fixed per-anatomy offset (capacity 50 IDs each)::

    Sacrum 1-50,  LeftHip 51-100,  RightHip 101-150,  Femur 151-200

Internally the model and loss only ever need ``(anatomy_group, local_id)``; the
global ID is purely an *I/O encoding* at the evaluation boundary (the PENGWIN
``.mha`` submission format requires global IDs). All instance-ID arithmetic in
``loss.py`` / ``eval.py`` / ``core.py`` / ``preprocessing.py`` / ``utils.py`` /
``visualize.py`` derives from this module so that:

  * adding an anatomy (or changing a range) is a one-row edit here, and
  * "did we cover every anatomy?" is answerable by inspection rather than by
    hunting scattered magic numbers (150, range(1,151), {1:1,2:51,3:101}, ...).

DECOY WARNING -- these are *instance* IDs. They are a DIFFERENT namespace from:
  * CASE IDs (``is_pelvic``/``is_femur`` in core.py: 1-120 / 151-200 / 251-420),
  * HU / CT-clip thresholds (e.g. 150.0, 2000),
  * probability thresholds (e.g. 0.50).
The same digits (150, 200, 50) appear in all of these. Do NOT route those through
this module; only route true instance-ID arithmetic here.

PELVIC-ONLY vs FULL -- some code paths are *deliberately* pelvic-scoped (the raw
Dataset537 pelvic source, the 3-anatomy decode proxy). For those, pass
``pelvic_only=True`` so the Femur exclusion is an explicit, reviewable choice
instead of a hidden ``<= 150``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Anatomy:
    """One anatomy's global instance-ID block."""

    name: str
    index: int  # 1-indexed semantic class (background = 0)
    id_lo: int
    id_hi: int

    @property
    def capacity(self) -> int:
        return self.id_hi - self.id_lo + 1


# =============================================================================
# THE single source of truth.  Add an anatomy = add one row.
# =============================================================================
ANATOMY_REGISTRY: list[Anatomy] = [
    Anatomy("Sacrum", 1, 1, 50),
    Anatomy("LeftHip", 2, 51, 100),
    Anatomy("RightHip", 3, 101, 150),
    Anatomy("Femur", 4, 151, 200),
]

# Pelvic-only subset (Sacrum / LeftHip / RightHip). Used by code that is
# *intentionally* pelvic-scoped; the Femur exclusion there is a semantic choice,
# made explicit here rather than hidden behind a magic 150.
PELVIC_ANATOMY_INDICES: tuple[int, ...] = (1, 2, 3)

# -----------------------------------------------------------------------------
# Derived scalars (never hardcode these downstream).
# -----------------------------------------------------------------------------
NUM_ANATOMIES: int = len(ANATOMY_REGISTRY)  # 4 (max semantic anatomy class index)
MIN_INSTANCE_ID: int = min(a.id_lo for a in ANATOMY_REGISTRY)  # 1
MAX_INSTANCE_ID: int = max(a.id_hi for a in ANATOMY_REGISTRY)  # 200
PELVIC_MAX_INSTANCE_ID: int = max(
    a.id_hi for a in ANATOMY_REGISTRY if a.index in PELVIC_ANATOMY_INDICES
)  # 150
INSTANCE_CAPACITY: int = max(a.capacity for a in ANATOMY_REGISTRY)  # 50 (uniform today)

# -----------------------------------------------------------------------------
# Legacy-compatible views so existing `from core import ANATOMY_RANGES` (and the
# retired utils.py dicts) keep resolving without behavior change.
# -----------------------------------------------------------------------------
ANATOMY_RANGES: list[tuple[str, int, int]] = [
    (a.name, a.id_lo, a.id_hi) for a in ANATOMY_REGISTRY
]
ANATOMY_NAMES: list[str] = [a.name for a in ANATOMY_REGISTRY]
ANATOMY_RANGE_DICT: dict[str, tuple[int, int]] = {
    a.name: (a.id_lo, a.id_hi) for a in ANATOMY_REGISTRY
}
ANATOMY_TO_INDEX: dict[str, int] = {a.name: a.index for a in ANATOMY_REGISTRY}

_BY_INDEX: dict[int, Anatomy] = {a.index: a for a in ANATOMY_REGISTRY}
_BY_NAME: dict[str, Anatomy] = {a.name: a for a in ANATOMY_REGISTRY}

# Vectorized id -> anatomy index lookup (slot 0 = background/invalid). Capacity-
# agnostic: works even if anatomies have non-uniform widths.
_ID_TO_ANATOMY = np.zeros(MAX_INSTANCE_ID + 1, dtype=np.int16)
for _a in ANATOMY_REGISTRY:
    _ID_TO_ANATOMY[_a.id_lo : _a.id_hi + 1] = _a.index
del _a


# =============================================================================
# Lookups
# =============================================================================
def anatomy_by_index(anatomy_index: int) -> Anatomy | None:
    return _BY_INDEX.get(int(anatomy_index))


def anatomy_by_name(name: str) -> Anatomy | None:
    return _BY_NAME.get(name)


def anatomy_of_id(gid: int) -> int:
    """Semantic anatomy index (1..N) for a global instance id; 0 if bg/invalid."""
    g = int(gid)
    a = _BY_INDEX.get(int(_ID_TO_ANATOMY[g])) if 0 <= g <= MAX_INSTANCE_ID else None
    return a.index if a is not None else 0


def id_range(anatomy_index: int) -> tuple[int, int]:
    """(lo, hi) global-ID block for an anatomy index. Replaces V5_ANATOMY_RANGES[name]."""
    a = _BY_INDEX[int(anatomy_index)]
    return (a.id_lo, a.id_hi)


def all_anatomy_ranges(pelvic_only: bool = False) -> dict[int, tuple[int, int]]:
    """{anatomy_index: (lo, hi)}. Replaces scattered {1:(1,50),...} dicts.

    pelvic_only=True omits Femur (the deliberate 3-anatomy proxy view).
    """
    return {
        a.index: (a.id_lo, a.id_hi)
        for a in ANATOMY_REGISTRY
        if (not pelvic_only or a.index in PELVIC_ANATOMY_INDICES)
    }


def anatomy_start_ids(pelvic_only: bool = False) -> dict[int, int]:
    """{anatomy_index: lo}. Replaces next_by_anatomy = {1:1, 2:51, 3:101}[, 4:151]."""
    return {
        a.index: a.id_lo
        for a in ANATOMY_REGISTRY
        if (not pelvic_only or a.index in PELVIC_ANATOMY_INDICES)
    }


def anatomy_ranges_by_name(pelvic_only: bool = False) -> dict[str, tuple[int, int]]:
    """{name: (lo, hi)}. Drop-in for V5_ANATOMY_RANGES / V5_ANATOMY_RANGES_WITH_FEMUR."""
    return {
        a.name: (a.id_lo, a.id_hi)
        for a in ANATOMY_REGISTRY
        if (not pelvic_only or a.index in PELVIC_ANATOMY_INDICES)
    }


# =============================================================================
# Global <-> local id conversion (capacity-aware; not hardcoded to 50)
# =============================================================================
def global_to_local(gid: int) -> int:
    """Global id -> anatomy-local id (1..capacity); 0 if bg/invalid.

    Replaces ``((id - 1) % 50) + 1``.
    """
    g = int(gid)
    a = _BY_INDEX.get(anatomy_of_id(g))
    return (g - a.id_lo + 1) if a is not None else 0


def local_to_global(anatomy_index: int, local_id: int) -> int:
    """(anatomy_index, local_id 1..capacity) -> global id.

    Replaces ``id_lo + counter`` reconstruction at the decode boundary.
    """
    a = _BY_INDEX[int(anatomy_index)]
    return a.id_lo + int(local_id) - 1


# =============================================================================
# Vectorized array helpers (replace the scattered magic-number masks)
# =============================================================================
def valid_instance_mask(arr: np.ndarray, pelvic_only: bool = False) -> np.ndarray:
    """Boolean mask of voxels holding a valid global instance id.

    Replaces ``(arr >= 1) & (arr <= 150)`` (and the half-migrated ``<= 200``).
    pelvic_only=True caps at RightHip (150) for intentionally pelvic-scoped code.
    """
    a = np.asarray(arr)
    hi = PELVIC_MAX_INSTANCE_ID if pelvic_only else MAX_INSTANCE_ID
    return (a >= MIN_INSTANCE_ID) & (a <= hi)


def clip_to_valid_instances(arr: np.ndarray, pelvic_only: bool = False) -> np.ndarray:
    """Zero out voxels that are not valid instance ids; keep valid ones.

    Replaces ``np.where((inst >= 1) & (inst <= 150), inst, 0)``.
    """
    a = np.asarray(arr)
    return np.where(valid_instance_mask(a, pelvic_only=pelvic_only), a, 0)


def anatomy_index_array(arr: np.ndarray) -> np.ndarray:
    """Per-voxel semantic anatomy index (0 where bg/invalid/out-of-range)."""
    a = np.asarray(arr)
    out = np.zeros(a.shape, dtype=np.int16)
    inb = (a >= 0) & (a <= MAX_INSTANCE_ID)
    out[inb] = _ID_TO_ANATOMY[a[inb].astype(np.int64)]
    return out


def same_anatomy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise: do (foreground) global ids a, b belong to the same anatomy?

    Replaces ``((a - 1) // 50) == ((b - 1) // 50)``. Note the ``// 50`` form
    already handles all four anatomies; this helper is capacity-aware and also
    excludes background-background matches.
    """
    ai = anatomy_index_array(a)
    bi = anatomy_index_array(b)
    return (ai == bi) & (ai > 0)
