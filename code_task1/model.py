"""PENGWIN 2026 Task 1 model utilities.

The active V5 fragment path intentionally uses the stock-compatible
`PengwinTrainer` Dice+CE contract; V3/V4 custom trainers are no longer active.

This module provides:
    - SWA (Stochastic Weight Averaging) over checkpoints
    - Trainer name to import-path mapping
    - Pretrained weight discovery
    - Optional plan patching hook
"""
from __future__ import annotations
import argparse
import copy
import json
import glob as _glob
from pathlib import Path
from typing import Dict, List

import torch

from core import (
    DATASETS, NN_PREP, NN_RES, RESULT_DATE, RESULT_REPORT, RESULT_WEIGHT,
    configure_nnunet_env, get_logger,
)
configure_nnunet_env()
log = get_logger(__name__)


TRAINER_INFO = {
    "PengwinTrainer": {
        "module": "nnunetv2.training.nnUNetTrainer.variants.pengwin.PengwinTrainer",
        "class_attrs": {
            "NUM_EPOCHS_DEFAULT": 1500,
            "ES_PATIENCE": 50,
            "ES_MIN_DELTA": 5e-3,
            "WARMUP_EPOCHS": 30,
            "DISABLE_X_MIRROR_DATASETS": {"Dataset532_PelvicAnatomyV2"},
        },
    },
}


PENGWIN_PLAN_CONFIGS = {
    # Split anatomy datasets intentionally use the nnU-Net ResEncL planner output without
    # a hand-authored patch. Keeping the hook as `None` makes this decision
    # explicit and lets `model.py patch-plans all` be a safe no-op.
    532: None,
    533: None,
    537: None,
}


SPLIT_ANATOMY_PLAN_VARIANTS = {
    # Femur shape failures after the split baseline should be tested with more
    # context before reintroducing mixed labels.
    "femur_zcontext_b1": {
        "source_plans": "nnUNetResEncUNetLPlans",
        "target_plans": "nnUNetResEncUNetLPlansFemurZContextB1",
        "dataset": 533,
        "description": "Dataset533 Femur ablation: deeper z context with batch size 1.",
        "configuration": {
            "3d_fullres": {
                "patch_size": [160, 288, 192],
                "batch_size": 1,
            }
        },
    },
}


# =============================================================================
# Checkpoint location helpers
# =============================================================================
def fold_dir(ds_id: int, fold: int = 0,
             plans: str = "nnUNetResEncUNetLPlans",
             config: str = "3d_fullres") -> Path:
    """Return path to nnUNet fold directory."""
    cfg = DATASETS[ds_id]
    return NN_RES / cfg["name"] / f"{cfg['trainer']}__{plans}__{config}" / f"fold_{fold}"


def best_ckpt(ds_id: int, fold: int = 0) -> Path:
    """Path to nnUNet's checkpoint_best.pth for given dataset / fold."""
    return fold_dir(ds_id, fold) / "checkpoint_best.pth"


def latest_ckpt(ds_id: int, fold: int = 0) -> Path:
    """Path to nnUNet's checkpoint_latest.pth for given dataset / fold."""
    return fold_dir(ds_id, fold) / "checkpoint_latest.pth"


# =============================================================================
# SWA — Stochastic Weight Averaging
# =============================================================================
def swa_average(ckpt_paths: List[Path], out_path: Path) -> None:
    """Equal-weight average of `network_weights` across multiple checkpoints.

    Algorithm:
      1. Load each checkpoint, sum tensors in float32 (avoid fp16 underflow).
      2. Divide by N at end, cast back to ORIGINAL dtype (taken from first
         checkpoint — assumes all checkpoints have identical dtypes per key,
         which is true for same-architecture models). If key dtypes differ
         across checkpoints (rare, unsupported), falls back to first ckpt's.
      3. Preserve all non-weights metadata (init_args, optimizer_state,
         _best_ema etc.) from the FIRST ckpt — only `network_weights` change.
      4. Append `_swa_n` (count) and `_swa_sources` (paths) for traceability.

    Args:
        ckpt_paths: list of .pth files. Order matters only for metadata source
            (first one is base for everything except weights).
        out_path: where to write the averaged checkpoint (.pth).

    Raises: ValueError if list is empty.
    """
    if not ckpt_paths:
        raise ValueError("no checkpoints to average")
    print(f"[SWA] averaging {len(ckpt_paths)} ckpts:")
    for p in ckpt_paths:
        print(f"   {p.name}")

    sums: Dict[str, torch.Tensor] = {}
    dtypes: Dict[str, torch.dtype] = {}   # captured from FIRST checkpoint, authoritative
    n = 0
    for p in ckpt_paths:
        # nnU-Net checkpoints include optimizer/logger metadata beyond tensors.
        # These files are local training artifacts, so loading with
        # weights_only=False is an explicit trust-boundary decision.
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        weights = ck["network_weights"]
        if not sums:
            sums = {k: v.float().clone() for k, v in weights.items()}
            dtypes = {k: v.dtype for k, v in weights.items()}   # original dtype
        else:
            for k, v in weights.items():
                sums[k] += v.float()
        n += 1

    # Cast back to original dtype (per-key from first ckpt) to keep file size +
    # downstream load behavior identical.
    avg = {k: (v / n).to(dtypes[k]) for k, v in sums.items()}
    base = torch.load(str(ckpt_paths[0]), map_location="cpu", weights_only=False)
    base["network_weights"] = avg
    base["_swa_n"] = n
    base["_swa_sources"] = [str(p) for p in ckpt_paths]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, str(out_path))
    print(f"\n✅ saved SWA ckpt → {out_path}  ({n} ckpts averaged)")


def swa_for_dataset(ds_id: int, last_n: int = 3, fold: int = 0,
                    out_path: Path | None = None) -> Path:
    """SWA last-N checkpoints in fold dir."""
    fdir = fold_dir(ds_id, fold)
    if not fdir.is_dir():
        raise FileNotFoundError(f"fold dir missing: {fdir}")
    candidates = sorted(_glob.glob(str(fdir / "checkpoint*.pth")))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints in {fdir}")
    # Prefer checkpoint_best + checkpoint_latest if exist, else last N by mtime
    selected = []
    for name in ["checkpoint_best.pth", "checkpoint_latest.pth"]:
        p = fdir / name
        if p.exists() and len(selected) < last_n:
            selected.append(p)
    # Fill remaining from any other checkpoint files
    for p in candidates:
        path = Path(p)
        if path not in selected and len(selected) < last_n:
            selected.append(path)

    # Keep derived weights under code_task1/result so checkpoint artifacts and
    # result dashboards stay in one auditable tree. Raw nnU-Net checkpoints
    # remain under nnunet/results and are referenced by visualize.py manifests.
    out_path = out_path or (RESULT_WEIGHT / f"swa_ds{ds_id}_fold{fold}.pth")
    swa_average(selected, out_path)
    return out_path


def patch_plans_json(ds_id: int, plans_name: str = "nnUNetResEncUNetLPlans") -> Path:
    """Apply optional PENGWIN plan overrides to nnU-Net's generated JSON."""
    if ds_id not in PENGWIN_PLAN_CONFIGS:
        raise ValueError(f"plans patch only defined for {sorted(PENGWIN_PLAN_CONFIGS)}")
    cfg = PENGWIN_PLAN_CONFIGS[ds_id]
    ds_name = DATASETS[ds_id]["name"]
    plans_path = NN_PREP / ds_name / f"{plans_name}.json"
    if not plans_path.exists():
        raise FileNotFoundError(f"plans json not found: {plans_path}")
    if cfg is None:
        print(f"[Ds{ds_id}] {ds_name}")
        print("  planner_default: no manual plan patch for this dataset")
        print(f"  checked:         {plans_path}")
        return plans_path

    plans = json.loads(plans_path.read_text())
    before = copy.deepcopy(plans["configurations"]["3d_fullres"])
    fullres = plans["configurations"]["3d_fullres"]
    arch = fullres["architecture"]["arch_kwargs"]
    fullres["patch_size"] = cfg["patch_size"]
    arch["n_stages"] = cfg["n_stages"]
    arch["features_per_stage"] = cfg["features_per_stage"]
    arch["n_blocks_per_stage"] = cfg["n_blocks_per_stage_encoder"]
    arch["n_conv_per_stage_decoder"] = cfg["n_conv_per_stage_decoder"]
    arch["strides"] = cfg["strides"]
    arch["kernel_sizes"] = cfg["kernel_sizes"]
    plans_path.write_text(json.dumps(plans, indent=4))

    print(f"[Ds{ds_id}] {ds_name}")
    print(f"  patch_size: {before['patch_size']} -> {fullres['patch_size']}")
    print(f"  n_stages:   {before['architecture']['arch_kwargs']['n_stages']} -> {arch['n_stages']}")
    print(f"  wrote:      {plans_path}")
    return plans_path


def _plans_path(ds_id: int, plans_name: str = "nnUNetResEncUNetLPlans") -> Path:
    return NN_PREP / DATASETS[ds_id]["name"] / f"{plans_name}.json"


def _warmstart_signature(plans: dict, config: str = "3d_fullres") -> dict:
    """Return architecture fields that must match for checkpoint warm-start."""
    conf = plans["configurations"][config]
    arch = conf["architecture"]["arch_kwargs"]
    return {
        "patch_size": conf.get("patch_size"),
        "batch_size": conf.get("batch_size"),
        "architecture_class_name": conf.get("architecture", {}).get("network_class_name"),
        "n_stages": arch.get("n_stages"),
        "features_per_stage": arch.get("features_per_stage"),
        "n_blocks_per_stage": arch.get("n_blocks_per_stage"),
        "n_conv_per_stage_decoder": arch.get("n_conv_per_stage_decoder"),
        "strides": arch.get("strides"),
        "kernel_sizes": arch.get("kernel_sizes"),
    }


def audit_plans_compatible(source: int = 532,
                           targets: list[int] | None = None,
                           plans_name: str = "nnUNetResEncUNetLPlans",
                           out_path: Path | None = None,
                           fail_on_mismatch: bool = True) -> dict:
    """Check architecture compatibility against a source plan.

    [AUDIT][Risk:Medium][Scope:warmstart]
    V5 does not require Dataset532 warm-start, but this command is kept as a
    read-only audit. It must not silently patch plans or imply that anatomy
    logits are model inputs for Dataset537.
    """
    targets = targets or [
        ds_id for ds_id, cfg in sorted(DATASETS.items())
        if cfg["kind"] == "bicm_v5"
    ]
    src_path = _plans_path(source, plans_name)
    if not src_path.exists():
        raise FileNotFoundError(f"source plans missing: {src_path}")
    src_plans = json.loads(src_path.read_text())
    src_sig = _warmstart_signature(src_plans)
    rows = {}
    ok = True
    for target in targets:
        tgt_path = _plans_path(target, plans_name)
        if not tgt_path.exists():
            rows[str(target)] = {
                "dataset_name": DATASETS[target]["name"],
                "plans": str(tgt_path),
                "ok": False,
                "error": "target plans missing",
            }
            ok = False
            continue
        tgt_plans = json.loads(tgt_path.read_text())
        tgt_sig = _warmstart_signature(tgt_plans)
        diff = {
            key: {"source": src_sig[key], "target": tgt_sig[key]}
            for key in src_sig
            if src_sig[key] != tgt_sig[key]
        }
        rows[str(target)] = {
            "dataset_name": DATASETS[target]["name"],
            "plans": str(tgt_path),
            "ok": not diff,
            "diff": diff,
            "warnings": {},
        }
        ok = ok and not diff
    report = {
        "source_dataset": source,
        "source_name": DATASETS[source]["name"],
        "plans_name": plans_name,
        "source_plans": str(src_path),
        "source_signature": src_sig,
        "targets": rows,
        "ok": ok,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
    if fail_on_mismatch and not ok:
        failed = [k for k, v in rows.items() if not v["ok"]]
        raise RuntimeError(f"plan compatibility failed for {failed}")
    return report


def patch_instance_plans_from_source(source: int = 532,
                                     targets: list[int] | None = None,
                                     plans_name: str = "nnUNetResEncUNetLPlans") -> list[Path]:
    """Copy Dataset532 3d_fullres topology fields into Dataset537 plans.

    [QA][Status:legacy_compat]
    This remains only for explicit manual compatibility checks. The V5 primary
    path should normally use planner output as-is and train `PengwinTrainer`
    with Dice+CE before any warm-start experiment is considered.
    """
    targets = targets or [
        ds_id for ds_id, cfg in sorted(DATASETS.items())
        if cfg["kind"] == "bicm_v5"
    ]
    src_path = _plans_path(source, plans_name)
    if not src_path.exists():
        raise FileNotFoundError(f"source plans missing: {src_path}")
    src_plans = json.loads(src_path.read_text())
    written = []
    for target in targets:
        tgt_path = _plans_path(target, plans_name)
        if not tgt_path.exists():
            raise FileNotFoundError(f"target plans missing: {tgt_path}")
        target_plans = json.loads(tgt_path.read_text())
        before = _warmstart_signature(target_plans)
        source_conf = copy.deepcopy(src_plans["configurations"]["3d_fullres"])
        target_plans["configurations"]["3d_fullres"] = source_conf
        target_plans["_pengwin_warmstart_source"] = {
            "source_dataset": source,
            "source_name": DATASETS[source]["name"],
            "source_plans": str(src_path),
            "purpose": "explicit manual V5 topology alignment audit only",
        }
        after = _warmstart_signature(target_plans)
        tgt_path.write_text(json.dumps(target_plans, indent=4))
        print(f"[plans] Ds{target} {DATASETS[target]['name']}:")
        print(f"  before: {before}")
        print(f"  after:  {after}")
        print(f"  wrote:  {tgt_path}")
        written.append(tgt_path)
    return written


patch_abbc_plans_from_source = patch_instance_plans_from_source


def create_plan_variant(ds_id: int, variant: str, overwrite: bool = False) -> Path:
    """Create a named plans JSON variant without modifying the B0 base plan.

    Upgrade experiments should use separate plans names so they are auditable
    and cannot accidentally change the active baseline checkpoints.
    """
    if ds_id not in DATASETS:
        raise ValueError(f"unknown active dataset {ds_id}. Choose from {sorted(DATASETS)}")
    if variant not in SPLIT_ANATOMY_PLAN_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(SPLIT_ANATOMY_PLAN_VARIANTS)}")
    expected = int(SPLIT_ANATOMY_PLAN_VARIANTS[variant].get("dataset", ds_id))
    if ds_id != expected:
        raise ValueError(f"variant {variant} is defined for Dataset{expected}, got Dataset{ds_id}")
    cfg = DATASETS[ds_id]
    spec = SPLIT_ANATOMY_PLAN_VARIANTS[variant]
    root = NN_PREP / cfg["name"]
    src = root / f"{spec['source_plans']}.json"
    dst = root / f"{spec['target_plans']}.json"
    if not src.exists():
        raise FileNotFoundError(f"source plans missing: {src}")
    if dst.exists() and not overwrite:
        print(f"[plan-variant] exists: {dst}")
        return dst

    plans = json.loads(src.read_text())
    plans["plans_name"] = spec["target_plans"]
    plans["_pengwin_variant"] = {
        "dataset": ds_id,
        "variant": variant,
        "source_plans": spec["source_plans"],
        "description": spec["description"],
    }
    for conf_name, updates in spec["configuration"].items():
        if conf_name not in plans["configurations"]:
            raise KeyError(f"configuration {conf_name!r} not in {src}")
        conf = plans["configurations"][conf_name]
        arch = conf["architecture"]["arch_kwargs"]
        for key, value in updates.items():
            if key == "n_conv_per_stage_decoder":
                arch[key] = value
            elif key in conf:
                conf[key] = value
            elif key in arch:
                arch[key] = value
            else:
                raise KeyError(f"unsupported plan variant key: {key}")
    dst.write_text(json.dumps(plans, indent=4))
    print(f"[plan-variant] {variant}: {src.name} -> {dst.name}")
    print(f"  description: {spec['description']}")
    return dst


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_swa = sub.add_parser("swa", help="SWA over fold checkpoints")
    p_swa.add_argument("--dataset", type=int, required=True)
    p_swa.add_argument("--fold", type=int, default=0)
    p_swa.add_argument("--last-n", type=int, default=3)
    p_swa.add_argument("--out", default=None)

    p_info = sub.add_parser("info", help="show trainer config")
    p_info.add_argument("--dataset", type=int, required=True)

    p_patch = sub.add_parser("patch-plans", help="patch/check active nnU-Net plans JSON")
    p_patch.add_argument("dataset", choices=[*(str(k) for k in sorted(DATASETS)), "all", "instance-from-532", "abbc-from-532"])
    p_patch.add_argument("--plans-name", default="nnUNetResEncUNetLPlans")

    p_audit = sub.add_parser("audit-plans-compatible", help="audit Dataset532 warm-start compatibility")
    p_audit.add_argument("--source", type=int, default=532)
    p_audit.add_argument("--targets", nargs="*", type=int, default=[537])
    p_audit.add_argument("--plans-name", default="nnUNetResEncUNetLPlans")
    p_audit.add_argument("--out", default=str(RESULT_REPORT / f"audit_plans_compatible_{RESULT_DATE}.json"))
    p_audit.add_argument("--allow-mismatch", action="store_true")

    p_variant = sub.add_parser("create-plan-variant", help="create split anatomy model ablation plans")
    p_variant.add_argument("variant", choices=sorted(SPLIT_ANATOMY_PLAN_VARIANTS))
    p_variant.add_argument("--dataset", type=int, default=533)
    p_variant.add_argument("--overwrite", action="store_true")

    args = p.parse_args()

    if args.cmd == "swa":
        out = Path(args.out) if args.out else None
        swa_for_dataset(args.dataset, last_n=args.last_n, fold=args.fold, out_path=out)
    elif args.cmd == "info":
        cfg = DATASETS[args.dataset]
        info = TRAINER_INFO[cfg["trainer"]]
        print(f"\nDataset {args.dataset}: {cfg['name']}")
        print(f"  kind:       {cfg['kind']}")
        print(f"  n_classes:  {cfg['n_classes']}")
        print(f"  trainer:    {cfg['trainer']}")
        print(f"  module:     {info['module']}")
        for k, v in info["class_attrs"].items():
            print(f"  {k:<24s}: {v}")
    elif args.cmd == "patch-plans":
        if args.dataset in {"instance-from-532", "abbc-from-532"}:
            patch_instance_plans_from_source(source=532, plans_name=args.plans_name)
        elif args.dataset == "all":
            targets = sorted(DATASETS)
            for ds_id in targets:
                patch_plans_json(ds_id, args.plans_name)
        else:
            targets = [int(args.dataset)]
            for ds_id in targets:
                patch_plans_json(ds_id, args.plans_name)
    elif args.cmd == "create-plan-variant":
        create_plan_variant(args.dataset, args.variant, overwrite=args.overwrite)
    elif args.cmd == "audit-plans-compatible":
        report = audit_plans_compatible(
            source=args.source,
            targets=args.targets,
            plans_name=args.plans_name,
            out_path=Path(args.out),
            fail_on_mismatch=not args.allow_mismatch,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
