"""Static deployment-contract checks for the v3.4 TotalSegmentator candidate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_selects_the_evaluated_candidate():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "chmod -R a+rX /opt/app/inference /opt/app/code_task1" in dockerfile
    assert (
        "PENGWIN_DS538_TRAINER="
        "PengwinTrainerSTUNetBaseAffinityV308DeployedVal"
    ) in dockerfile
    assert "PENGWIN_DS538_FOLD=0" in dockerfile
    assert "PENGWIN_DS538_OUT_CH=13" in dockerfile
    assert "PENGWIN_AFFINITY_DECODE=1" in dockerfile
    assert "PENGWIN_AGGLO_T=0.75" in dockerfile
    assert "PENGWIN_TARGET_ROUTER=1" in dockerfile
    assert "PENGWIN_RF_CONF_MARGIN=0.15" in dockerfile


def test_candidate_trainer_is_exportable_by_the_nnunet_shim():
    core_path = REPO_ROOT / "code_task1" / "core.py"
    spec = importlib.util.spec_from_file_location("pengwin_candidate_core", core_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    candidate = module.PengwinTrainerSTUNetBaseAffinityV308DeployedVal
    base = module.PengwinTrainerSTUNetBaseAffinityV308
    assert issubclass(candidate, base)
