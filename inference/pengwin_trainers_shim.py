"""nnUNet trainer-discovery shim for PENGWIN ABBC trainers.

nnUNet v2's `recursive_find_python_class` only walks files inside
`nnunetv2/training/nnUNetTrainer/`. Our custom trainers live in
`/opt/app/code_task1/core.py`, which is NOT on that walk. This shim is
copied into the nnUNet trainer dir at Docker build time so nnUNet's
discovery finds the trainer classes by name attribute on this module.

Mechanism:
    1. Ensure /opt/app/code_task1 is on sys.path (matches the Docker
       PYTHONPATH).
    2. `import core` to load our V288-V291 trainer definitions.
    3. Re-export every `PengwinTrainer*` class at module level so
       `getattr(module, trainer_class_name)` succeeds inside nnUNet.

This intentionally does NOT depend on `code_task1/__init__.py` (the
upstream package has no __init__).
"""
import os
import sys

_CODE_DIR = "/opt/app/code_task1"
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import core as _pengwin_core  # noqa: E402

# Re-export every PengwinTrainer* class at module level so nnUNet's
# recursive_find_python_class can pick it up by getattr().
for _name in dir(_pengwin_core):
    if _name.startswith("PengwinTrainer"):
        globals()[_name] = getattr(_pengwin_core, _name)

# Optional: expose the count for build-time / debug verification.
__pengwin_trainer_count__ = sum(
    1 for _n in dir(_pengwin_core) if _n.startswith("PengwinTrainer")
)
