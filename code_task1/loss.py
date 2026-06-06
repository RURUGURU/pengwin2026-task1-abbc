"""PENGWIN 2026 Task 1 — Custom loss components (v7).

Provides optional BoundaryDoU, Tversky, and class-weighted CE components on top
of nnU-Net's default Dice + CE. The production default is still plain DC+CE,
matching the PENGWIN 2024 top CT recipes; weak-class/boundary weighting are
explicit ablation profiles.

    - 4 anatomies (Sa/LH/RH/Femur), 1-200 instance IDs.
    - Pelvic cases (001-120, 151-200) carry only Sa/LH/RH.
    - Femur cases (251-420) carry only Femur.
    - Fragment counts are highly variable (2026 audit: Pelvic can exceed
      20 total fragments; RightHip is the densest hard anatomy).
    - Boundary voxels at fracture surfaces are the hardest signal —
      Dice ignores them; CE treats them uniformly.

BoundaryDoU is not enabled by default in the trainer; use loss profiles for
controlled ablation before promoting it.

Reference:
    BoundaryDoU — Sun et al., "Boundary Difference Over Union Loss for
    Medical Image Segmentation," MICCAI 2023.
    https://arxiv.org/abs/2308.00220 (2D in paper; we adapt to 3D below.)
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

# Single source of truth for the anatomy<->instance-ID encoding. Instance IDs in
# this module are torch tensors, so we use the registry's scalar bound rather than
# its numpy mask helpers: every "(x > 0) & (x <= 150)" support mask and "x > 150"
# guard becomes "<= MAX_INSTANCE_ID", which spans Femur (151-200). This is a safe
# superset — pelvic ROIs hold no 151-200 voxels, femur ROIs are no longer dropped.
# NOTE: "150" inside class names (BICMV150...) is a version tag, NOT an instance id.
from anatomy_registry import MAX_INSTANCE_ID


class BoundaryFragmentV3Loss(nn.Module):
    """BoundaryFragment V3 semantic-specialist objective.

    The network still emits a normal 5-class nnU-Net softmax tensor, but the
    checkpoint signal is not plain mean Dice. This wrapper keeps nnU-Net's base
    Dice+CE term and adds supervision for the pieces that determine IoU-F:

    - soft adjacent-label CE so class 2/3/4 errors are graded near boundaries;
    - binary fragment Dice over support {2,3,4};
    - precision-oriented Tversky on class 2 fracture barriers;
    - narrow-band CE emphasis on external context/barrier/shell labels.

    A true HD95 loss would require distance-transform sidecars at every deep
    supervision scale. For V3 we keep that as validation/checkpoint selection
    and use the narrow-band CE term as the differentiable surface proxy.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        soft_ce_lambda: float = 0.15,
        fragment_dice_lambda: float = 0.35,
        barrier_tversky_lambda: float = 0.18,
        narrow_band_lambda: float = 0.12,
        surface_proxy_lambda: float = 0.05,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.base = base_loss
        self.soft_ce_lambda = float(soft_ce_lambda)
        self.fragment_dice_lambda = float(fragment_dice_lambda)
        self.barrier_tversky_lambda = float(barrier_tversky_lambda)
        self.narrow_band_lambda = float(narrow_band_lambda)
        self.surface_proxy_lambda = float(surface_proxy_lambda)
        self.smooth = float(smooth)

    @staticmethod
    def _labels(target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 5 and target.shape[1] == 5:
            return target.argmax(dim=1)
        if target.ndim == 5 and target.shape[1] == 1:
            return target[:, 0].long()
        return target.long()

    def _soft_targets(self, labels: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        labels = labels.clamp(0, 4)
        one = F.one_hot(labels, num_classes=5).permute(0, 4, 1, 2, 3).to(dtype=dtype, device=device)
        soft = one.clone()
        external = labels == 1
        soft[:, 0][external], soft[:, 1][external], soft[:, 3][external] = 0.12, 0.80, 0.08
        barrier = labels == 2
        soft[:, 2][barrier], soft[:, 3][barrier] = 0.84, 0.16
        shell = labels == 3
        soft[:, 2][shell], soft[:, 3][shell], soft[:, 4][shell] = 0.08, 0.78, 0.14
        core = labels == 4
        soft[:, 3][core], soft[:, 4][core] = 0.08, 0.92
        return soft

    def _soft_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        soft = self._soft_targets(labels, logits.dtype, logits.device)
        return -(soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))
        inter = (pred * target).sum(dim=dims)
        denom = pred.sum(dim=dims) + target.sum(dim=dims)
        return (1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean()

    def _tversky(self, pred: torch.Tensor, target: torch.Tensor,
                 fp_weight: float, fn_weight: float) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))
        tp = (pred * target).sum(dim=dims)
        fp = (pred * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - pred) * target).sum(dim=dims)
        return (1.0 - (tp + self.smooth) / (tp + fp_weight * fp + fn_weight * fn + self.smooth)).mean()

    def _narrow_band_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        band = (labels == 1) | (labels == 2) | (labels == 3)
        if not band.any():
            return logits.new_zeros(())
        ce = F.cross_entropy(logits, labels, reduction="none")
        return ce[band].mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = self.base(pred_logits, target)
        labels = self._labels(target).to(device=pred_logits.device)
        prob = torch.softmax(pred_logits, dim=1)
        support_target = ((labels == 2) | (labels == 3) | (labels == 4)).float().unsqueeze(1)
        barrier_target = (labels == 2).float().unsqueeze(1)
        shell_target = ((labels == 2) | (labels == 3)).float().unsqueeze(1)
        support_prob = prob[:, 2:5].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        barrier_prob = prob[:, 2:3]
        shell_prob = prob[:, 2:4].sum(dim=1, keepdim=True).clamp(0.0, 1.0)

        total = total + self.soft_ce_lambda * self._soft_ce(pred_logits, labels)
        total = total + self.fragment_dice_lambda * self._dice_loss(support_prob, support_target)
        # Class 2 overprediction was the central failure mode, so FP carries
        # more weight than FN here. Recall is still enforced by the gate metric.
        total = total + self.barrier_tversky_lambda * self._tversky(
            barrier_prob, barrier_target, fp_weight=0.70, fn_weight=0.30
        )
        total = total + self.narrow_band_lambda * self._narrow_band_ce(pred_logits, labels)
        total = total + self.surface_proxy_lambda * self._tversky(
            shell_prob, shell_target, fp_weight=0.45, fn_weight=0.55
        )
        return total


class BoundaryFragmentV3RidgeFitLoss(BoundaryFragmentV3Loss):
    """Recall-oriented class-2 ridge objective for V3.1 root-cause tests.

    The input-ablation overfits showed that support/core can be learned while
    the thin class-2 ridge is absorbed into shell. This profile keeps the normal
    semantic head but adds terms that explicitly separate class 2 from class 3
    on full-resolution supervision. It is intentionally a training experiment,
    not a decoder/postprocess change.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        fullres_ridge_terms: bool = True,
        soft_ce_lambda: float = 0.10,
        fragment_dice_lambda: float = 0.35,
        barrier_tversky_lambda: float = 0.55,
        barrier_focal_lambda: float = 0.35,
        barrier_margin_lambda: float = 0.22,
        hard_negative_lambda: float = 0.18,
        narrow_band_lambda: float = 0.08,
        surface_proxy_lambda: float = 0.03,
        margin: float = 1.0,
        topk_fraction: float = 0.01,
        smooth: float = 1e-5,
    ):
        super().__init__(
            base_loss,
            soft_ce_lambda=soft_ce_lambda,
            fragment_dice_lambda=fragment_dice_lambda,
            barrier_tversky_lambda=barrier_tversky_lambda,
            narrow_band_lambda=narrow_band_lambda,
            surface_proxy_lambda=surface_proxy_lambda,
            smooth=smooth,
        )
        self.fullres_ridge_terms = bool(fullres_ridge_terms)
        self.barrier_focal_lambda = float(barrier_focal_lambda)
        self.barrier_margin_lambda = float(barrier_margin_lambda)
        self.hard_negative_lambda = float(hard_negative_lambda)
        self.margin = float(margin)
        self.topk_fraction = float(topk_fraction)

    def _soft_targets(self, labels: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # Keep adjacent-label smoothing for non-ridge bands, but make GT class 2
        # one-hot. The experiment specifically tests whether allowing class2->3
        # softness caused the thin ridge to be swallowed by shell.
        soft = super()._soft_targets(labels, dtype, device)
        barrier = labels == 2
        if barrier.any():
            soft[:, 0][barrier] = 0.0
            soft[:, 1][barrier] = 0.0
            soft[:, 2][barrier] = 1.0
            soft[:, 3][barrier] = 0.0
            soft[:, 4][barrier] = 0.0
        return soft

    def _barrier_focal(self, barrier_prob: torch.Tensor, barrier_target: torch.Tensor) -> torch.Tensor:
        # [QC][Invariant:amp_focal_dtype]
        # In fp16, `1.0 - 1e-5` rounds back to 1.0 and can make log(1 - p)
        # evaluate as log(0). Focal calibration is reduction-only, so compute it
        # in fp32 while keeping gradients connected to the original probabilities.
        prob = barrier_prob.float().clamp(self.smooth, 1.0 - self.smooth)
        target = barrier_target
        pos = target > 0.5
        neg = ~pos
        loss = prob.new_zeros(())
        if pos.any():
            pos_loss = -0.85 * ((1.0 - prob[pos]) ** 2.0) * torch.log(prob[pos])
            loss = loss + pos_loss.mean()
        if neg.any():
            neg_loss = -0.15 * (prob[neg] ** 2.0) * torch.log(1.0 - prob[neg])
            loss = loss + neg_loss.mean()
        return loss

    def _barrier_margin(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        barrier = labels == 2
        if not barrier.any():
            return logits.new_zeros(())
        class2 = logits[:, 2]
        other = torch.cat([logits[:, :2], logits[:, 3:]], dim=1).amax(dim=1)
        return F.relu(self.margin + other[barrier] - class2[barrier]).mean()

    def _hard_negative_topk(self, barrier_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Penalize the exact confusion that matters here: class2 probability on
        # external context and shell near-negatives. Top-k keeps this from being
        # dominated by easy background voxels.
        hard_neg = (labels == 1) | (labels == 3)
        if not hard_neg.any():
            return barrier_prob.new_zeros(())
        vals = barrier_prob[:, 0][hard_neg]
        if vals.numel() == 0:
            return barrier_prob.new_zeros(())
        k = max(1, int(round(float(vals.numel()) * self.topk_fraction)))
        return torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        labels = self._labels(target).to(device=pred_logits.device)
        prob = torch.softmax(pred_logits, dim=1)
        support_target = ((labels == 2) | (labels == 3) | (labels == 4)).float().unsqueeze(1)
        barrier_target = (labels == 2).float().unsqueeze(1)
        shell_target = ((labels == 2) | (labels == 3)).float().unsqueeze(1)
        support_prob = prob[:, 2:5].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        barrier_prob = prob[:, 2:3]
        shell_prob = prob[:, 2:4].sum(dim=1, keepdim=True).clamp(0.0, 1.0)

        total = self.base(pred_logits, target)
        total = total + self.fragment_dice_lambda * self._dice_loss(support_prob, support_target)
        if not self.fullres_ridge_terms:
            return total

        total = total + self.soft_ce_lambda * self._soft_ce(pred_logits, labels)
        total = total + self.barrier_tversky_lambda * self._tversky(
            barrier_prob, barrier_target, fp_weight=0.35, fn_weight=0.75
        )
        total = total + self.barrier_focal_lambda * self._barrier_focal(barrier_prob, barrier_target)
        total = total + self.barrier_margin_lambda * self._barrier_margin(pred_logits, labels)
        total = total + self.hard_negative_lambda * self._hard_negative_topk(barrier_prob, labels)
        total = total + self.narrow_band_lambda * self._narrow_band_ce(pred_logits, labels)
        total = total + self.surface_proxy_lambda * self._tversky(
            shell_prob, shell_target, fp_weight=0.45, fn_weight=0.55
        )
        return total


class BoundaryFragmentV3CoreRecallRidgeLoss(BoundaryFragmentV3RidgeFitLoss):
    """BFV3 recall-biased class-2 ridge objective with explicit core seed terms."""

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        fullres_ridge_terms: bool = True,
        core_tversky_lambda: float = 0.85,
        core_focal_lambda: float = 0.55,
        core_margin_lambda: float = 0.35,
        false_core_topk_lambda: float = 0.18,
        topk_fraction: float = 0.02,
        smooth: float = 1e-5,
    ):
        super().__init__(
            base_loss,
            fullres_ridge_terms=fullres_ridge_terms,
            smooth=smooth,
        )
        self.core_tversky_lambda = float(core_tversky_lambda)
        self.core_focal_lambda = float(core_focal_lambda)
        self.core_margin_lambda = float(core_margin_lambda)
        self.false_core_topk_lambda = float(false_core_topk_lambda)
        self.core_topk_fraction = float(topk_fraction)

    def _core_focal(self, core_prob: torch.Tensor, core_target: torch.Tensor) -> torch.Tensor:
        prob = core_prob.float().clamp(self.smooth, 1.0 - self.smooth)
        pos = core_target > 0.5
        neg = ~pos
        total = prob.new_zeros(())
        if pos.any():
            total = total + (-0.90 * ((1.0 - prob[pos]) ** 2.0) * torch.log(prob[pos])).mean()
        if neg.any():
            total = total + (-0.10 * (prob[neg] ** 2.0) * torch.log(1.0 - prob[neg])).mean()
        return total

    def _core_margin(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        core = labels == 4
        if not core.any():
            return logits.new_zeros(())
        class4 = logits[:, 4]
        other = logits[:, :4].amax(dim=1)
        return F.relu(1.0 + other[core] - class4[core]).mean()

    def _false_core_topk(self, core_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        non_core = labels != 4
        if not non_core.any():
            return core_prob.new_zeros(())
        vals = core_prob[:, 0][non_core]
        if vals.numel() == 0:
            return core_prob.new_zeros(())
        k = max(1, int(round(float(vals.numel()) * self.core_topk_fraction)))
        return torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = super().forward(pred_logits, target)
        labels = self._labels(target).to(device=pred_logits.device)
        prob = torch.softmax(pred_logits, dim=1)
        core_prob = prob[:, 4:5]
        core_target = (labels == 4).float().unsqueeze(1)
        # [AUDIT][Risk:High][Scope:barrier_core_tradeoff]
        # V167 recovered core but suppressed full-fold barrier after epoch 0.
        # This variant keeps the same core terms and changes only the class-2
        # ridge bias from precision-heavy to recall-oriented.
        total = total + self.core_tversky_lambda * self._tversky(
            core_prob, core_target, fp_weight=0.25, fn_weight=0.85
        )
        total = total + self.core_focal_lambda * self._core_focal(core_prob, core_target)
        total = total + self.core_margin_lambda * self._core_margin(pred_logits, labels)
        total = total + self.false_core_topk_lambda * self._false_core_topk(core_prob, labels)
        return total


class BoundaryFragmentV3CoreRecallRidgeMassCapLoss(BoundaryFragmentV3CoreRecallRidgeLoss):
    """V169 recall/core objective with a class-2 barrier false-mass budget.

    The target contract stays BFV3: 0 background, 1 exterior, 2 barrier,
    3 shell, 4 core. This variant changes only the class-2 calibration pressure
    to test whether V169's high recall can be retained while reducing the
    thresholded barrier overprediction observed on cached hard-case logits.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        barrier_mass_cap: float = 12.0,
        barrier_mass_lambda: float = 0.35,
        barrier_empty_mean_cap: float = 0.004,
        barrier_empty_lambda: float = 0.15,
        **kwargs,
    ):
        super().__init__(base_loss, **kwargs)
        self.barrier_mass_cap = float(barrier_mass_cap)
        self.barrier_mass_lambda = float(barrier_mass_lambda)
        self.barrier_empty_mean_cap = float(barrier_empty_mean_cap)
        self.barrier_empty_lambda = float(barrier_empty_lambda)

    def _barrier_mass_cap_loss(self, barrier_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = []
        barrier = labels == 2
        valid = labels != 0
        barrier_prob_f = barrier_prob.float()
        for idx in range(labels.shape[0]):
            sample_valid = valid[idx]
            if not sample_valid.any():
                continue
            # [QC][Invariant:amp_reduction_dtype]
            # Full 3D patches can contain millions of valid voxels. Under AMP,
            # summing float16 probabilities may overflow before the mass cap is
            # applied, which makes smoke tests report `inf` loss without a model
            # failure. Use fp32 for reduction-only calibration terms.
            pred_mass = barrier_prob_f[idx, 0][sample_valid].sum()
            target_mass = barrier[idx][sample_valid].float().sum()
            if bool((target_mass > 0).detach().cpu().item()):
                # [QC][Invariant:barrier_mass_ratio]
                # V169 restored barrier recall but produced broad class-2 masks.
                # Penalizing only mass above a cap preserves positive recall
                # pressure while stopping patches from becoming all-barrier.
                allowed_mass = self.barrier_mass_cap * target_mass
                violation = torch.log((pred_mass + 1.0) / (allowed_mass + 1.0))
                losses.append(F.relu(violation).pow(2))
            else:
                if self.barrier_empty_lambda <= 0.0:
                    continue
                # [QC][Invariant:empty_barrier_patch]
                # Some hard-case anatomies have no class-2 target. A pure
                # positive-patch ratio cap would not constrain those false
                # barriers, so empty patches use a normalized mean-probability
                # budget instead of silently falling back to base CE.
                mean_prob = pred_mass / sample_valid.float().sum().clamp_min(1.0)
                violation = F.relu((mean_prob / max(self.barrier_empty_mean_cap, self.smooth)) - 1.0)
                losses.append(self.barrier_empty_lambda * violation.pow(2))
        if not losses:
            return barrier_prob.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = super().forward(pred_logits, target)
        labels = self._labels(target).to(device=pred_logits.device)
        barrier_prob = torch.softmax(pred_logits, dim=1)[:, 2:3]
        total = total + self.barrier_mass_lambda * self._barrier_mass_cap_loss(barrier_prob, labels)
        return total


class BoundaryFragmentV3CoreRecallRidgeVerySoftPositiveMassCapLoss(
    BoundaryFragmentV3CoreRecallRidgeMassCapLoss
):
    """Very soft positive-patch cap for V173 recall recovery check."""

    def __init__(self, base_loss: nn.Module, **kwargs):
        super().__init__(
            base_loss,
            barrier_mass_cap=48.0,
            barrier_mass_lambda=0.20,
            barrier_empty_lambda=0.0,
            **kwargs,
        )


class BoundaryFragmentV3CoreRecallRidgeLogitCalibratedLoss(
    BoundaryFragmentV3CoreRecallRidgeVerySoftPositiveMassCapLoss
):
    """V178-style loss with direct class-2 barrier logit calibration.

    Args:
        base_loss: nnU-Net DC+CE base loss.
        barrier_logit_bce_lambda: Weight for one-vs-rest class-2 BCE.
        barrier_logit_focal_lambda: Weight for class-2 focal calibration.
        barrier_logit_margin_lambda: Weight for positive class-2 logit margin.
        false_barrier_topk_lambda: Weight for high-confidence false class-2 mass.

    Raises:
        ValueError: If logits are not the BFV3 5-class tensor expected here.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        barrier_logit_bce_lambda: float = 0.25,
        barrier_logit_focal_lambda: float = 0.25,
        barrier_logit_margin_lambda: float = 0.30,
        false_barrier_topk_lambda: float = 0.08,
        barrier_logit_pos_weight: float = 16.0,
        barrier_logit_focal_alpha: float = 0.80,
        barrier_logit_focal_gamma: float = 2.0,
        barrier_logit_margin: float = 0.60,
        topk_fraction: float = 0.01,
        **kwargs,
    ):
        super().__init__(base_loss, **kwargs)
        self.barrier_logit_bce_lambda = float(barrier_logit_bce_lambda)
        self.barrier_logit_focal_lambda = float(barrier_logit_focal_lambda)
        self.barrier_logit_margin_lambda = float(barrier_logit_margin_lambda)
        self.false_barrier_topk_lambda = float(false_barrier_topk_lambda)
        self.barrier_logit_pos_weight = float(barrier_logit_pos_weight)
        self.barrier_logit_focal_alpha = float(barrier_logit_focal_alpha)
        self.barrier_logit_focal_gamma = float(barrier_logit_focal_gamma)
        self.barrier_logit_margin = float(barrier_logit_margin)
        self.barrier_topk_fraction = float(topk_fraction)

    def _validate_logits(self, pred_logits: torch.Tensor) -> None:
        if pred_logits.ndim != 5 or pred_logits.shape[1] != 5:
            raise ValueError(
                "BoundaryFragmentV3 class-2 calibration expects logits [B, 5, D, H, W], "
                f"got {tuple(pred_logits.shape)}"
            )

    def _barrier_logit_bce(self, barrier_logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target = (labels == 2).to(dtype=barrier_logit.dtype)
        valid = labels != 0
        if not valid.any():
            return barrier_logit.new_zeros(())
        bce = F.binary_cross_entropy_with_logits(barrier_logit, target, reduction="none")
        weights = torch.ones_like(bce)
        weights = torch.where(target > 0.5, weights * self.barrier_logit_pos_weight, weights)
        return (bce[valid] * weights[valid]).mean()

    def _barrier_logit_focal(self, barrier_logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target = (labels == 2)
        valid = labels != 0
        prob = torch.sigmoid(barrier_logit).clamp(self.smooth, 1.0 - self.smooth)
        total = barrier_logit.new_zeros(())
        pos = target & valid
        neg = (~target) & valid
        gamma = self.barrier_logit_focal_gamma
        alpha = self.barrier_logit_focal_alpha
        if pos.any():
            total = total + (-alpha * ((1.0 - prob[pos]) ** gamma) * torch.log(prob[pos])).mean()
        if neg.any():
            total = total + (-(1.0 - alpha) * (prob[neg] ** gamma) * torch.log(1.0 - prob[neg])).mean()
        return total

    def _barrier_positive_margin(self, pred_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        barrier = labels == 2
        if not barrier.any():
            return pred_logits.new_zeros(())
        class2 = pred_logits[:, 2]
        other = torch.cat((pred_logits[:, :2], pred_logits[:, 3:]), dim=1).amax(dim=1)
        return F.relu(self.barrier_logit_margin + other[barrier] - class2[barrier]).mean()

    def _false_barrier_topk(self, barrier_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # [QC][Invariant:class2_threshold_calibration]
        # V178 showed broad low-probability class-2 mass below threshold 0.5.
        # Penalizing only the highest non-barrier probabilities targets that
        # calibration failure without forcing all non-barrier voxels to zero.
        non_barrier_valid = (labels != 2) & (labels != 0)
        if not non_barrier_valid.any():
            return barrier_prob.new_zeros(())
        vals = barrier_prob[:, 0][non_barrier_valid]
        if vals.numel() == 0:
            return barrier_prob.new_zeros(())
        k = max(1, int(round(float(vals.numel()) * self.barrier_topk_fraction)))
        return torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._validate_logits(pred_logits)
        total = super().forward(pred_logits, target)
        labels = self._labels(target).to(device=pred_logits.device)
        logits_f = pred_logits.float()
        barrier_logit = logits_f[:, 2]
        barrier_prob = torch.softmax(logits_f, dim=1)[:, 2:3]

        # [AUDIT][Risk:High][Scope:barrier_calibration]
        # V178 retained support/core signal but class-2 disappeared at threshold
        # 0.5 on hard-case prediction. These auxiliary terms change only the
        # class-2 logit calibration while keeping the BFV3 target contract fixed.
        total = total + self.barrier_logit_bce_lambda * self._barrier_logit_bce(barrier_logit, labels)
        total = total + self.barrier_logit_focal_lambda * self._barrier_logit_focal(barrier_logit, labels)
        total = total + self.barrier_logit_margin_lambda * self._barrier_positive_margin(logits_f, labels)
        total = total + self.false_barrier_topk_lambda * self._false_barrier_topk(barrier_prob, labels)
        return total


class BoundaryFragmentV3CoreRecallRidgePrecisionLogitCalibratedLoss(
    BoundaryFragmentV3CoreRecallRidgeLogitCalibratedLoss
):
    """Precision-biased class-2 calibration paired with V179."""

    def __init__(self, base_loss: nn.Module, **kwargs):
        super().__init__(
            base_loss,
            barrier_logit_bce_lambda=0.18,
            barrier_logit_focal_lambda=0.20,
            barrier_logit_margin_lambda=0.22,
            false_barrier_topk_lambda=0.22,
            barrier_logit_pos_weight=10.0,
            barrier_logit_focal_alpha=0.70,
            barrier_logit_margin=0.45,
            topk_fraction=0.02,
            **kwargs,
        )


class BoundaryFragmentV3CoreRecallRidgeCandidateBinaryBarrierHeadLoss(nn.Module):
    """BFV3 semantic scaffold with barrier head trained only in shell candidates."""

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        aux_bce_lambda: float = 0.46,
        aux_focal_lambda: float = 0.34,
        aux_tversky_lambda: float = 0.44,
        aux_positive_floor_lambda: float = 0.40,
        aux_shell_topk_lambda: float = 0.06,
        aux_pos_weight: float = 24.0,
        aux_positive_floor: float = 0.62,
        aux_focal_alpha: float = 0.88,
        aux_focal_gamma: float = 2.0,
        aux_topk_fraction: float = 0.012,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.semantic_loss = BoundaryFragmentV3CoreRecallRidgePrecisionLogitCalibratedLoss(base_loss)
        self.aux_bce_lambda = float(aux_bce_lambda)
        self.aux_focal_lambda = float(aux_focal_lambda)
        self.aux_tversky_lambda = float(aux_tversky_lambda)
        self.aux_positive_floor_lambda = float(aux_positive_floor_lambda)
        self.aux_shell_topk_lambda = float(aux_shell_topk_lambda)
        self.aux_pos_weight = float(aux_pos_weight)
        self.aux_positive_floor = float(aux_positive_floor)
        self.aux_focal_alpha = float(aux_focal_alpha)
        self.aux_focal_gamma = float(aux_focal_gamma)
        self.aux_topk_fraction = float(aux_topk_fraction)
        self.smooth = float(smooth)

    @staticmethod
    def _labels(target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 5 and target.shape[1] == 5:
            return target.argmax(dim=1)
        if target.ndim == 5 and target.shape[1] == 1:
            return target[:, 0].long()
        return target.long()

    def _validate_logits(self, pred_logits: torch.Tensor) -> None:
        if pred_logits.ndim != 5 or pred_logits.shape[1] != 6:
            raise ValueError(
                "BFV3 candidate barrier-head loss expects logits [B, 6, D, H, W], "
                f"got {tuple(pred_logits.shape)}"
            )

    def _candidate_bce(self, aux_logit: torch.Tensor, target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if not candidate.any():
            return aux_logit.new_zeros(())
        bce = F.binary_cross_entropy_with_logits(aux_logit, target, reduction="none")
        weights = torch.where(target > 0.5, aux_logit.new_full((), self.aux_pos_weight), aux_logit.new_ones(()))
        return (bce[candidate] * weights[candidate]).mean()

    def _candidate_focal(self, aux_prob: torch.Tensor, target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        prob = aux_prob.float().clamp(self.smooth, 1.0 - self.smooth)
        pos = candidate & (target > 0.5)
        neg = candidate & (target <= 0.5)
        total = aux_prob.new_zeros(())
        if pos.any():
            total = total + (
                -self.aux_focal_alpha
                * ((1.0 - prob[pos]) ** self.aux_focal_gamma)
                * torch.log(prob[pos])
            ).mean()
        if neg.any():
            total = total + (
                -(1.0 - self.aux_focal_alpha)
                * (prob[neg] ** self.aux_focal_gamma)
                * torch.log(1.0 - prob[neg])
            ).mean()
        return total

    def _candidate_tversky(self, aux_prob: torch.Tensor, target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if not candidate.any():
            return aux_prob.new_zeros(())
        pred = aux_prob[candidate].float()
        tgt = target[candidate].float()
        tp = (pred * tgt).sum()
        fp = (pred * (1.0 - tgt)).sum()
        fn = ((1.0 - pred) * tgt).sum()
        return 1.0 - (tp + self.smooth) / (tp + 0.45 * fp + 0.85 * fn + self.smooth)

    def _positive_floor_loss(self, aux_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        barrier = labels == 2
        if not barrier.any():
            return aux_prob.new_zeros(())
        return F.relu(self.aux_positive_floor - aux_prob[barrier].float()).mean()

    def _shell_topk_false(self, aux_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shell = labels == 3
        if not shell.any():
            return aux_prob.new_zeros(())
        vals = aux_prob[shell].float()
        if vals.numel() == 0:
            return aux_prob.new_zeros(())
        k = max(1, int(round(float(vals.numel()) * self.aux_topk_fraction)))
        return torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._validate_logits(pred_logits)
        semantic_logits = pred_logits[:, :5]
        aux_logit = pred_logits[:, 5].float()
        aux_prob = torch.sigmoid(aux_logit)
        labels = self._labels(target).to(device=pred_logits.device)
        aux_target = (labels == 2).to(dtype=aux_logit.dtype)
        candidate = (labels == 2) | (labels == 3)

        # [AUDIT][Risk:High][Scope:candidate_restricted_barrier]
        # V185 proved positive pressure can recover barrier recall only by
        # overpainting globally. This variant applies the binary objective only
        # inside the GT shell/barrier candidate band so the experiment tests local
        # barrier-vs-shell localization, not another all-volume binary classifier.
        total = self.semantic_loss(semantic_logits, target)
        total = total + self.aux_bce_lambda * self._candidate_bce(aux_logit, aux_target, candidate)
        total = total + self.aux_focal_lambda * self._candidate_focal(aux_prob, aux_target, candidate)
        total = total + self.aux_tversky_lambda * self._candidate_tversky(aux_prob, aux_target, candidate)
        total = total + self.aux_positive_floor_lambda * self._positive_floor_loss(aux_prob, labels)
        total = total + self.aux_shell_topk_lambda * self._shell_topk_false(aux_prob, labels)
        if not torch.isfinite(total):
            raise ValueError(f"BFV3 candidate binary barrier-head loss produced non-finite loss: {float(total.detach().cpu())}")
        return total


class BoundaryFragmentV3CoreRecallRidgeDenseCandidateCore025StrongPeakContactShellContrastBarrierHeadLoss(
    BoundaryFragmentV3CoreRecallRidgeCandidateBinaryBarrierHeadLoss
):
    """V220 strong core compactness with explicit contact-vs-shell aux contrast."""

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        fragment_peak_lambda: float = 0.12,
        fragment_peak_floor: float = 0.34,
        fragment_gap_lambda: float = 0.16,
        fragment_gap_margin: float = 0.12,
        fragment_false_topk_lambda: float = 0.14,
        fragment_false_topk_fraction: float = 0.03,
        contact_shell_contrast_lambda: float = 0.18,
        contact_shell_margin: float = 0.12,
        contact_low_fraction: float = 0.10,
        shell_high_fraction: float = 0.03,
        **kwargs,
    ):
        kwargs.setdefault("aux_bce_lambda", 0.32)
        kwargs.setdefault("aux_focal_lambda", 0.24)
        kwargs.setdefault("aux_tversky_lambda", 0.30)
        kwargs.setdefault("aux_positive_floor_lambda", 0.14)
        kwargs.setdefault("aux_shell_topk_lambda", 0.28)
        kwargs.setdefault("aux_pos_weight", 8.0)
        kwargs.setdefault("aux_positive_floor", 0.55)
        kwargs.setdefault("aux_focal_alpha", 0.68)
        kwargs.setdefault("aux_topk_fraction", 0.03)
        super().__init__(base_loss, **kwargs)
        self.fragment_peak_lambda = float(fragment_peak_lambda)
        self.fragment_peak_floor = float(fragment_peak_floor)
        self.fragment_gap_lambda = float(fragment_gap_lambda)
        self.fragment_gap_margin = float(fragment_gap_margin)
        self.fragment_false_topk_lambda = float(fragment_false_topk_lambda)
        self.fragment_false_topk_fraction = float(fragment_false_topk_fraction)
        self.contact_shell_contrast_lambda = float(contact_shell_contrast_lambda)
        self.contact_shell_margin = float(contact_shell_margin)
        self.contact_low_fraction = float(contact_low_fraction)
        self.shell_high_fraction = float(shell_high_fraction)
        for name, value in [
            ("fragment_false_topk_fraction", self.fragment_false_topk_fraction),
            ("contact_low_fraction", self.contact_low_fraction),
            ("shell_high_fraction", self.shell_high_fraction),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

    @staticmethod
    def _semantic_target(target):
        if isinstance(target, dict):
            if "semantic" not in target:
                raise KeyError("strong-peak contact-shell contrast loss requires target['semantic']")
            return target["semantic"]
        return target

    @staticmethod
    def _instance_target(target):
        if not isinstance(target, dict) or "instance" not in target:
            raise KeyError("strong-peak contact-shell contrast loss requires target['instance']")
        instance = target["instance"]
        if instance.ndim == 5 and instance.shape[1] == 1:
            instance = instance[:, 0]
        return instance.long()

    def _fragment_peak_losses(
        self,
        core_prob: torch.Tensor,
        labels: torch.Tensor,
        instance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peak_losses = []
        gap_losses = []
        false_losses = []
        for batch_idx in range(labels.shape[0]):
            inst = instance[batch_idx]
            lbl = labels[batch_idx]
            prob = core_prob[batch_idx].float()
            for frag_id in torch.unique(inst):
                fid = int(frag_id.detach().cpu())
                if fid <= 0 or fid > MAX_INSTANCE_ID:
                    continue
                fragment = inst == fid
                core_mask = fragment & (lbl == 4)
                if not core_mask.any():
                    continue
                support_non_core = fragment & ((lbl == 2) | (lbl == 3))
                core_top = prob[core_mask].max()
                peak_losses.append(F.relu(self.fragment_peak_floor - core_top))
                if support_non_core.any():
                    non_core_vals = prob[support_non_core]
                    gap_losses.append(F.relu(non_core_vals.max() + self.fragment_gap_margin - core_top))
                    k = max(1, int(round(float(non_core_vals.numel()) * self.fragment_false_topk_fraction)))
                    false_losses.append(torch.topk(non_core_vals, k=min(k, non_core_vals.numel()), largest=True).values.mean())
        zero = core_prob.new_zeros(())
        peak = torch.stack(peak_losses).mean() if peak_losses else zero
        gap = torch.stack(gap_losses).mean() if gap_losses else zero
        false = torch.stack(false_losses).mean() if false_losses else zero
        return peak, gap, false

    def _contact_shell_contrast_loss(self, aux_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        contact_vals = aux_prob[labels == 2].float()
        shell_vals = aux_prob[labels == 3].float()
        if contact_vals.numel() == 0 or shell_vals.numel() == 0:
            return aux_prob.new_zeros(())
        contact_k = max(1, int(round(float(contact_vals.numel()) * self.contact_low_fraction)))
        shell_k = max(1, int(round(float(shell_vals.numel()) * self.shell_high_fraction)))
        contact_low = torch.topk(contact_vals, k=min(contact_k, contact_vals.numel()), largest=False).values.mean()
        shell_high = torch.topk(shell_vals, k=min(shell_k, shell_vals.numel()), largest=True).values.mean()
        # [AUDIT][Risk:High][Scope:v245_v246_contact_separation]
        # Calibration probe showed V220/V241/V242 overpaint because GT label2
        # contact and GT label3 shell probabilities overlap, while V243/V244
        # suppress both classes. This term targets separation only: label2 must
        # sit above high shell responses by a margin without global mass/floor
        # pressure that would obscure whether separation improved.
        return F.relu(shell_high + self.contact_shell_margin - contact_low)

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        self._validate_logits(pred_logits)
        semantic_target = self._semantic_target(target)
        labels = self._labels(semantic_target).to(device=pred_logits.device)
        instance = self._instance_target(target).to(device=pred_logits.device)
        if tuple(instance.shape) != tuple(labels.shape):
            raise ValueError(
                "strong-peak contact-shell contrast target shape mismatch: "
                f"instance={tuple(instance.shape)} labels={tuple(labels.shape)}"
            )
        semantic_logits = pred_logits[:, :5]
        aux_logit = pred_logits[:, 5].float()
        aux_prob = torch.sigmoid(aux_logit)
        aux_target = (labels == 2).to(dtype=aux_logit.dtype)
        candidate = (labels == 2) | (labels == 3)

        total = self.semantic_loss(semantic_logits, semantic_target)
        total = total + self.aux_bce_lambda * self._candidate_bce(aux_logit, aux_target, candidate)
        total = total + self.aux_focal_lambda * self._candidate_focal(aux_prob, aux_target, candidate)
        total = total + self.aux_tversky_lambda * self._candidate_tversky(aux_prob, aux_target, candidate)
        total = total + self.aux_positive_floor_lambda * self._positive_floor_loss(aux_prob, labels)
        total = total + self.aux_shell_topk_lambda * self._shell_topk_false(aux_prob, labels)
        total = total + self.contact_shell_contrast_lambda * self._contact_shell_contrast_loss(aux_prob, labels)
        core_prob = torch.softmax(semantic_logits.float(), dim=1)[:, 4]
        peak_loss, gap_loss, false_loss = self._fragment_peak_losses(core_prob, labels, instance)
        total = total + self.fragment_peak_lambda * peak_loss
        total = total + self.fragment_gap_lambda * gap_loss
        total = total + self.fragment_false_topk_lambda * false_loss
        if not torch.isfinite(total):
            raise ValueError(f"BFV3 strong-peak contact-shell contrast loss produced non-finite loss: {float(total.detach().cpu())}")
        return total


class BoundaryFragmentV3Core025StrongPeakXYZAffinityHeadLoss(
    BoundaryFragmentV3CoreRecallRidgeDenseCandidateCore025StrongPeakContactShellContrastBarrierHeadLoss
):
    """V248 BFV3 semantic scaffold plus z/y/x same-fragment affinity loss.

    Args:
        base_loss: 5채널 BFV3 semantic softmax 경로에 적용할 nnU-Net base loss.

    전제조건:
        pred_logits는 [B, 8, Z, Y, X]이다. channel 0..4는 기존 BFV3 semantic
        logits이고, channel 5..7은 각각 z/y/x 방향의 same-fragment affinity
        logits이다. target dict는 `semantic`, `instance`를 포함해야 한다.

    실패 조건:
        shape/dtype/device 불일치, NaN/Inf, 잘못된 instance ID 범위는 즉시
        ValueError/TypeError로 중단한다.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        affinity_bce_lambda: float = 1.20,
        affinity_margin_lambda: float = 0.35,
        affinity_threshold_guard_lambda: float = 0.30,
        affinity_margin: float = 0.25,
        affinity_same_floor: float = 0.65,
        affinity_break_ceiling: float = 0.35,
        affinity_low_fraction: float = 0.05,
        affinity_high_fraction: float = 0.20,
        **kwargs,
    ):
        # [AUDIT][Risk:High][Scope:v248_representation_pivot]
        # V247의 binary contact head는 0.50 threshold에서 over-contact와 collapse를
        # 반복했다. V248은 scalar contact class를 더 세게 누르는 실험이 아니라,
        # decoder가 실제로 필요한 인접 voxel pair의 same-fragment 관계를 직접 학습한다.
        # 부모 초기화에서 semantic/core compactness scaffold만 재사용하고, 부모의
        # channel5 contact forward는 이 클래스에서 호출하지 않는다.
        super().__init__(base_loss, **kwargs)
        self.affinity_bce_lambda = float(affinity_bce_lambda)
        self.affinity_margin_lambda = float(affinity_margin_lambda)
        self.affinity_threshold_guard_lambda = float(affinity_threshold_guard_lambda)
        self.affinity_margin = float(affinity_margin)
        self.affinity_same_floor = float(affinity_same_floor)
        self.affinity_break_ceiling = float(affinity_break_ceiling)
        self.affinity_low_fraction = float(affinity_low_fraction)
        self.affinity_high_fraction = float(affinity_high_fraction)
        for name, value in [
            ("affinity_bce_lambda", self.affinity_bce_lambda),
            ("affinity_margin_lambda", self.affinity_margin_lambda),
            ("affinity_threshold_guard_lambda", self.affinity_threshold_guard_lambda),
            ("affinity_margin", self.affinity_margin),
            ("affinity_low_fraction", self.affinity_low_fraction),
            ("affinity_high_fraction", self.affinity_high_fraction),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        for name, value in [
            ("affinity_same_floor", self.affinity_same_floor),
            ("affinity_break_ceiling", self.affinity_break_ceiling),
        ]:
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if self.affinity_break_ceiling >= 0.5 or self.affinity_same_floor <= 0.5:
            raise ValueError(
                "V248 affinity guard must straddle the fixed 0.50 decoder split: "
                f"same_floor={self.affinity_same_floor}, break_ceiling={self.affinity_break_ceiling}"
            )

    @staticmethod
    def _validate_xyz_affinity_logits(pred_logits: torch.Tensor) -> None:
        # [QC][Invariant:channel_order]
        # V248 channel order is fixed: 0..4 BFV3 semantic softmax, 5..7 z/y/x
        # same-fragment affinity logits. Accidentally evaluating 6채널 V247이나
        # 10채널 V68 logits here would invert the decoder contract.
        if pred_logits.ndim != 5 or int(pred_logits.shape[1]) != 8:
            raise ValueError(
                "BFV3 V248 xyz-affinity loss expects logits [B, 8, Z, Y, X], "
                f"got {tuple(pred_logits.shape)}"
            )

    @staticmethod
    def _same_fragment_affinity_targets(instance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build z/y/x pairwise same-fragment affinity targets from instance IDs."""
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        if instance.ndim != 4:
            raise ValueError(
                "V248 affinity target requires instance shape [B, Z, Y, X] "
                f"or [B, 1, Z, Y, X], got {tuple(instance.shape)}"
            )
        inst = instance.long()
        target = torch.zeros((int(inst.shape[0]), 3, *inst.shape[1:]), dtype=torch.float32, device=inst.device)
        valid = torch.zeros_like(target, dtype=torch.bool)
        invalid = (inst < 0) | (inst > MAX_INSTANCE_ID)
        if invalid.any():
            bad_min = int(inst.min().detach().cpu())
            bad_max = int(inst.max().detach().cpu())
            raise ValueError(f"V248 instance IDs must be in [0, {MAX_INSTANCE_ID}], got min={bad_min} max={bad_max}")
        for axis in range(3):
            a_sl = [slice(None)] * 4
            b_sl = [slice(None)] * 4
            a_sl[axis + 1] = slice(0, -1)
            b_sl[axis + 1] = slice(1, None)
            a = inst[tuple(a_sl)]
            b = inst[tuple(b_sl)]
            a_fg = (a > 0) & (a <= MAX_INSTANCE_ID)
            b_fg = (b > 0) & (b <= MAX_INSTANCE_ID)
            same_anatomy = a_fg & b_fg & (((a - 1) // 50) == ((b - 1) // 50))
            same_fragment = same_anatomy & (a == b)
            target_axis = target[:, axis]
            valid_axis = valid[:, axis]
            target_axis[tuple(a_sl)] = same_fragment.to(dtype=target.dtype)
            valid_axis[tuple(a_sl)] = same_anatomy
        return target, valid

    def _validate_xyz_affinity_inputs(
        self,
        pred_logits: torch.Tensor,
        labels: torch.Tensor,
        instance: torch.Tensor,
        affinity_target: torch.Tensor,
        affinity_valid: torch.Tensor,
    ) -> None:
        # [QC][Invariant:shape_dtype_device]
        # Pairwise affinity는 voxel class target과 축이 하나 더 많은 target을 동시에
        # 사용한다. 여기서 shape/device를 강제하지 않으면 z/y/x edge가 semantic crop과
        # 어긋난 상태로도 loss가 계산될 수 있다.
        self._validate_xyz_affinity_logits(pred_logits)
        expected_voxel = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        expected_edge = tuple(int(v) for v in pred_logits.shape[0:1] + (3,) + pred_logits.shape[2:])
        if tuple(labels.shape) != expected_voxel:
            raise ValueError(f"V248 semantic labels shape mismatch: expected={expected_voxel}, got={tuple(labels.shape)}")
        if tuple(instance.shape) != expected_voxel:
            raise ValueError(f"V248 instance shape mismatch: expected={expected_voxel}, got={tuple(instance.shape)}")
        if tuple(affinity_target.shape) != expected_edge or tuple(affinity_valid.shape) != expected_edge:
            raise ValueError(
                "V248 affinity target shape mismatch: "
                f"expected={expected_edge}, target={tuple(affinity_target.shape)}, valid={tuple(affinity_valid.shape)}"
            )
        if (
            labels.device != pred_logits.device
            or instance.device != pred_logits.device
            or affinity_target.device != pred_logits.device
            or affinity_valid.device != pred_logits.device
        ):
            raise ValueError(
                "V248 affinity device mismatch: "
                f"logits={pred_logits.device} labels={labels.device} instance={instance.device} "
                f"affinity_target={affinity_target.device} affinity_valid={affinity_valid.device}"
            )
        if labels.dtype != torch.long:
            raise TypeError(f"V248 semantic labels must be torch.long, got {labels.dtype}")
        if instance.dtype != torch.long:
            raise TypeError(f"V248 instance target must be torch.long, got {instance.dtype}")

    @staticmethod
    def _probability_to_logit(probability: float) -> float:
        clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        return math.log(clipped / (1.0 - clipped))

    def _balanced_affinity_bce(
        self,
        affinity_logit: torch.Tensor,
        affinity_target: torch.Tensor,
        affinity_valid: torch.Tensor,
    ) -> torch.Tensor:
        same_edge = affinity_valid & (affinity_target > 0.5)
        break_edge = affinity_valid & (affinity_target <= 0.5)
        if not affinity_valid.any():
            # [DATA][Risk:High][Scope:no_pairwise_supervision]
            # 빈/단일-fragment crop에서는 same-anatomy pairwise edge가 없을 수 있다.
            # 이때 negative-only affinity loss를 만들면 decoder cost prior가 sampling
            # 비율에 따라 흔들리므로 semantic/core scaffold만 학습한다.
            return affinity_logit.new_zeros(())
        total = affinity_logit.new_zeros(())
        denom = 0.0
        if same_edge.any():
            pos_logits = affinity_logit[same_edge].float()
            total = total + F.binary_cross_entropy_with_logits(
                pos_logits,
                torch.ones_like(pos_logits),
                reduction="mean",
            )
            denom += 1.0
        if break_edge.any():
            neg_logits = affinity_logit[break_edge].float()
            total = total + F.binary_cross_entropy_with_logits(
                neg_logits,
                torch.zeros_like(neg_logits),
                reduction="mean",
            )
            denom += 1.0
        # [DATA][Risk:High][Scope:affinity_imbalance]
        # same-fragment edges are much denser than fragment-boundary break edges.
        # 전체 edge 평균 BCE를 쓰면 break edge가 묻혀서 watershed가 fragments를 merge한다.
        # 따라서 same-edge와 break-edge 항을 batch 안에서 1:1로 평균한다.
        return total / max(denom, 1.0)

    def _affinity_margin_loss(
        self,
        affinity_prob: torch.Tensor,
        affinity_target: torch.Tensor,
        affinity_valid: torch.Tensor,
    ) -> torch.Tensor:
        same_vals = affinity_prob[affinity_valid & (affinity_target > 0.5)].float()
        break_vals = affinity_prob[affinity_valid & (affinity_target <= 0.5)].float()
        if same_vals.numel() == 0 or break_vals.numel() == 0:
            return affinity_prob.new_zeros(())
        same_k = max(1, int(round(float(same_vals.numel()) * self.affinity_low_fraction)))
        break_k = max(1, int(round(float(break_vals.numel()) * self.affinity_high_fraction)))
        weak_same = torch.topk(same_vals, k=min(same_k, same_vals.numel()), largest=False).values.mean()
        hard_break = torch.topk(break_vals, k=min(break_k, break_vals.numel()), largest=True).values.mean()
        # [METRIC][Scope:affinity_threshold]
        # decoder에서는 same-fragment affinity 0.50 아래가 separator cost로 바뀐다.
        # 이 margin은 threshold search가 아니라, 같은 fragment edge의 하위 분위가
        # true break edge의 상위 tail보다 충분히 높아야 한다는 fixed contract guard다.
        return F.relu(hard_break + self.affinity_margin - weak_same).pow(2)

    def _affinity_threshold_guard_loss(
        self,
        affinity_logit: torch.Tensor,
        affinity_target: torch.Tensor,
        affinity_valid: torch.Tensor,
    ) -> torch.Tensor:
        same_edge = affinity_valid & (affinity_target > 0.5)
        break_edge = affinity_valid & (affinity_target <= 0.5)
        losses = []
        if same_edge.any():
            floor_logit = self._probability_to_logit(self.affinity_same_floor)
            losses.append(F.relu(float(floor_logit) - affinity_logit[same_edge].float()).mean())
        if break_edge.any():
            ceiling_logit = self._probability_to_logit(self.affinity_break_ceiling)
            losses.append(F.relu(affinity_logit[break_edge].float() - float(ceiling_logit)).mean())
        if not losses:
            return affinity_logit.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        semantic_target = self._semantic_target(target)
        if torch.is_tensor(semantic_target):
            semantic_target = semantic_target.to(device=pred_logits.device)
        labels = self._labels(semantic_target).to(device=pred_logits.device, dtype=torch.long)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        affinity_target, affinity_valid = self._same_fragment_affinity_targets(instance)
        self._validate_xyz_affinity_inputs(
            pred_logits,
            labels,
            instance,
            affinity_target,
            affinity_valid,
        )

        semantic_logits = pred_logits[:, :5]
        affinity_logit = pred_logits[:, 5:8].float()
        affinity_prob = torch.sigmoid(affinity_logit)

        total = self.semantic_loss(semantic_logits, semantic_target)
        total = total + self.affinity_bce_lambda * self._balanced_affinity_bce(
            affinity_logit,
            affinity_target,
            affinity_valid,
        )
        total = total + self.affinity_margin_lambda * self._affinity_margin_loss(
            affinity_prob,
            affinity_target,
            affinity_valid,
        )
        total = total + self.affinity_threshold_guard_lambda * self._affinity_threshold_guard_loss(
            affinity_logit,
            affinity_target,
            affinity_valid,
        )

        core_prob = torch.softmax(semantic_logits.float(), dim=1)[:, 4]
        peak_loss, gap_loss, false_loss = self._fragment_peak_losses(core_prob, labels, instance)
        total = total + self.fragment_peak_lambda * peak_loss
        total = total + self.fragment_gap_lambda * gap_loss
        total = total + self.fragment_false_topk_lambda * false_loss
        if not torch.isfinite(total):
            # [AUDIT][Risk:High][Scope:training_stability]
            # NaN/Inf loss 뒤 optimizer step을 허용하면 affinity logits 전체가 오염되고,
            # graph decoder의 fragment assignment까지 신뢰할 수 없어진다.
            raise ValueError(
                "BFV3 V248 xyz-affinity loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_affinity_ordering]
        # Expected: same-fragment edge logits high / break-edge logits low 설정의
        # loss가 반대 설정보다 낮고, backward 후 pred_logits.grad가 finite해야 한다.
        return total


class BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss(
    BoundaryFragmentV3Core025StrongPeakXYZAffinityHeadLoss
):
    """V253 train-ready loss for the oracle-passing affinity13 + seed contract.

    전제조건:
        pred_logits는 [B, 18, Z, Y, X]이다. channel 0..4는 각각
        support/exterior/fracture_surface/seed_center/seed_body binary logit이고,
        channel 5..17은 13개 26-neighborhood directed affinity logit이다.

    실패 조건:
        shape/dtype/device 불일치, 잘못된 instance ID, NaN/Inf는 즉시
        ValueError/TypeError로 중단한다.
    """

    AFFINITY13_OFFSETS_ZYX: tuple[tuple[int, int, int], ...] = (
        (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    )
    OUTPUT_CHANNELS = 5 + len(AFFINITY13_OFFSETS_ZYX)

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        support_bce_lambda: float = 0.45,
        exterior_bce_lambda: float = 0.20,
        fracture_bce_lambda: float = 0.35,
        seed_center_bce_lambda: float = 0.20,
        seed_body_bce_lambda: float = 0.30,
        affinity_bce_lambda: float = 1.35,
        affinity_margin_lambda: float = 0.35,
        affinity_threshold_guard_lambda: float = 0.30,
        **kwargs,
    ):
        # [AUDIT][Risk:High][Scope:v253_train_contract]
        # V251/V252 oracle 실패 원인은 loss scalar가 아니라 decoder가 보는 graph
        # contract였다. 이 loss는 oracle-pass V253 채널 의미를 그대로 학습 target으로
        # 만든다. 기존 V248의 3축 affinity나 contact-head contract로 fallback하지 않는다.
        super().__init__(
            base_loss,
            affinity_bce_lambda=affinity_bce_lambda,
            affinity_margin_lambda=affinity_margin_lambda,
            affinity_threshold_guard_lambda=affinity_threshold_guard_lambda,
            **kwargs,
        )
        self.support_bce_lambda = float(support_bce_lambda)
        self.exterior_bce_lambda = float(exterior_bce_lambda)
        self.fracture_bce_lambda = float(fracture_bce_lambda)
        self.seed_center_bce_lambda = float(seed_center_bce_lambda)
        self.seed_body_bce_lambda = float(seed_body_bce_lambda)
        for name in (
            "support_bce_lambda",
            "exterior_bce_lambda",
            "fracture_bce_lambda",
            "seed_center_bce_lambda",
            "seed_body_bce_lambda",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

    @staticmethod
    def _offset_slices(shape: tuple[int, int, int],
                       offset_zyx: tuple[int, int, int]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        src: list[slice] = []
        dst: list[slice] = []
        for dim, delta in zip(shape, offset_zyx):
            if abs(int(delta)) >= int(dim):
                raise ValueError(f"V253 affinity offset {offset_zyx} invalid for shape {shape}")
            if int(delta) > 0:
                src.append(slice(0, -int(delta)))
                dst.append(slice(int(delta), None))
            elif int(delta) < 0:
                src.append(slice(-int(delta), None))
                dst.append(slice(0, int(delta)))
            else:
                src.append(slice(None))
                dst.append(slice(None))
        return tuple(src), tuple(dst)

    @staticmethod
    def _seed_center_target(target) -> torch.Tensor:
        if isinstance(target, dict) and "seed_center" in target:
            seed_center = target["seed_center"]
        elif isinstance(target, dict) and "seed" in target:
            # [QA][Status:Fallback][Scope:legacy_dataloader]
            # Expected: V253 dataloader provides `seed_center`. This fallback keeps
            # synthetic/unit tests usable but full train readiness should see the
            # explicit key in batch targets.
            seed_center = target["seed"]
        else:
            raise KeyError("V253 loss requires target['seed_center'] or target['seed']")
        if seed_center.ndim == 5 and int(seed_center.shape[1]) == 1:
            seed_center = seed_center[:, 0]
        return seed_center.float()

    @staticmethod
    def _seed_body_target(target) -> torch.Tensor:
        if not isinstance(target, dict) or "seed" not in target:
            raise KeyError("V253 loss requires target['seed']")
        seed = target["seed"]
        if seed.ndim == 5 and int(seed.shape[1]) == 1:
            seed = seed[:, 0]
        return seed.float()

    @classmethod
    def _validate_affinity13_logits(cls, pred_logits: torch.Tensor) -> None:
        # [QC][Invariant:channel_order]
        # V253 channel order is fixed by the oracle-pass decoder:
        # 0 support, 1 exterior, 2 fracture surface, 3 seed center, 4 seed body,
        # 5..17 affinity13. A 6/8-channel BFV3 tensor here would silently train
        # the wrong decoder contract.
        if pred_logits.ndim != 5 or int(pred_logits.shape[1]) != int(cls.OUTPUT_CHANNELS):
            raise ValueError(
                f"BFV3 V253 affinity13-seed loss expects logits [B, {cls.OUTPUT_CHANNELS}, Z, Y, X], "
                f"got {tuple(pred_logits.shape)}"
            )

    @classmethod
    def _same_fragment_affinity13_targets(cls, instance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        if instance.ndim != 4:
            raise ValueError(
                "V253 affinity13 target requires instance shape [B, Z, Y, X] "
                f"or [B, 1, Z, Y, X], got {tuple(instance.shape)}"
            )
        inst = instance.long()
        invalid = (inst < 0) | (inst > MAX_INSTANCE_ID)
        if invalid.any():
            bad_min = int(inst.min().detach().cpu())
            bad_max = int(inst.max().detach().cpu())
            raise ValueError(f"V253 instance IDs must be in [0, {MAX_INSTANCE_ID}], got min={bad_min} max={bad_max}")
        target = torch.zeros(
            (int(inst.shape[0]), len(cls.AFFINITY13_OFFSETS_ZYX), *inst.shape[1:]),
            dtype=torch.float32,
            device=inst.device,
        )
        valid = torch.zeros_like(target, dtype=torch.bool)
        for i, offset in enumerate(cls.AFFINITY13_OFFSETS_ZYX):
            src_sl_zyx, dst_sl_zyx = cls._offset_slices(tuple(int(v) for v in inst.shape[1:]), offset)
            src_sl = (slice(None), *src_sl_zyx)
            dst_sl = (slice(None), *dst_sl_zyx)
            a = inst[src_sl]
            b = inst[dst_sl]
            a_fg = (a > 0) & (a <= MAX_INSTANCE_ID)
            b_fg = (b > 0) & (b <= MAX_INSTANCE_ID)
            same_anatomy = a_fg & b_fg & (((a - 1) // 50) == ((b - 1) // 50))
            same_fragment = same_anatomy & (a == b)
            target_i = target[:, i]
            valid_i = valid[:, i]
            # [QC][Invariant:shape_dtype_device]
            # target_i/valid_i는 [B, Z, Y, X]이고 src_sl_zyx는 공간축 3개만
            # 표현한다. batch 축을 명시적으로 보존하지 않으면 첫 공간 slice가
            # batch 축에 잘못 적용되어 affinity target 자체가 깨진다.
            target_i[(slice(None), *src_sl_zyx)] = same_fragment.to(dtype=target.dtype)
            valid_i[(slice(None), *src_sl_zyx)] = same_anatomy
        return target, valid

    @staticmethod
    def _balanced_binary_bce(
        logit: torch.Tensor,
        target: torch.Tensor,
        *,
        name: str,
        hard_negative_ratio: int = 16,
    ) -> torch.Tensor:
        if logit.shape != target.shape:
            raise ValueError(f"V253 {name} BCE shape mismatch: logit={tuple(logit.shape)} target={tuple(target.shape)}")
        pos = target > 0.5
        neg = ~pos
        if not pos.any() and not neg.any():
            return logit.new_zeros(())
        if pos.any():
            pos_logits = logit[pos].float()
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="mean")
        else:
            pos_loss = logit.new_zeros(())
        if neg.any():
            neg_logits = logit[neg].float()
            if pos.any():
                k = min(int(neg_logits.numel()), max(1, int(pos.sum().detach().cpu()) * int(hard_negative_ratio)))
                neg_logits = torch.topk(neg_logits, k=k, largest=True).values
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction="mean")
        else:
            neg_loss = logit.new_zeros(())
        if pos.any() and neg.any():
            return 0.5 * (pos_loss + neg_loss)
        return pos_loss + neg_loss

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        self._validate_affinity13_logits(pred_logits)
        semantic_target = self._semantic_target(target)
        if torch.is_tensor(semantic_target):
            semantic_target = semantic_target.to(device=pred_logits.device)
        labels = self._labels(semantic_target).to(device=pred_logits.device, dtype=torch.long)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_center = self._seed_center_target(target).to(device=pred_logits.device, dtype=torch.float32)
        seed_body = self._seed_body_target(target).to(device=pred_logits.device, dtype=torch.float32)
        if tuple(labels.shape) != tuple(pred_logits.shape[0:1] + pred_logits.shape[2:]):
            raise ValueError(f"V253 label shape mismatch: labels={tuple(labels.shape)} logits={tuple(pred_logits.shape)}")
        if tuple(instance.shape) != tuple(labels.shape):
            raise ValueError(f"V253 instance shape mismatch: instance={tuple(instance.shape)} labels={tuple(labels.shape)}")
        if tuple(seed_center.shape) != tuple(labels.shape) or tuple(seed_body.shape) != tuple(labels.shape):
            raise ValueError(
                f"V253 seed shape mismatch: seed_center={tuple(seed_center.shape)} "
                f"seed_body={tuple(seed_body.shape)} labels={tuple(labels.shape)}"
            )

        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        exterior_target = (labels == 1).float()
        fracture_target = (labels == 2).float()
        affinity_target, affinity_valid = self._same_fragment_affinity13_targets(instance)
        if affinity_target.device != pred_logits.device or affinity_valid.device != pred_logits.device:
            raise ValueError("V253 affinity target device mismatch")

        base_logits = pred_logits[:, :5].float()
        affinity_logit = pred_logits[:, 5:].float()
        affinity_prob = torch.sigmoid(affinity_logit)

        # [DATA][Risk:High][Scope:v253_binary_targets]
        # channel0 support와 channel1 exterior는 서로 보완적이지만 softmax가 아니다.
        # Oracle decoder는 support mask와 seed/affinity graph를 직접 쓰므로 각 channel은
        # 독립 binary logit으로 학습한다.
        total = pred_logits.new_zeros(())
        total = total + self.support_bce_lambda * self._balanced_binary_bce(base_logits[:, 0], support_target, name="support")
        total = total + self.exterior_bce_lambda * self._balanced_binary_bce(base_logits[:, 1], exterior_target, name="exterior")
        total = total + self.fracture_bce_lambda * self._balanced_binary_bce(base_logits[:, 2], fracture_target, name="fracture_surface")
        total = total + self.seed_center_bce_lambda * self._balanced_binary_bce(
            base_logits[:, 3],
            (seed_center > 0.5).float(),
            name="seed_center",
            hard_negative_ratio=32,
        )
        total = total + self.seed_body_bce_lambda * self._balanced_binary_bce(base_logits[:, 4], seed_body, name="seed_body")
        total = total + self.affinity_bce_lambda * self._balanced_affinity_bce(
            affinity_logit,
            affinity_target,
            affinity_valid,
        )
        total = total + self.affinity_margin_lambda * self._affinity_margin_loss(
            affinity_prob,
            affinity_target,
            affinity_valid,
        )
        total = total + self.affinity_threshold_guard_lambda * self._affinity_threshold_guard_loss(
            affinity_logit,
            affinity_target,
            affinity_valid,
        )
        if not torch.isfinite(total):
            # [AUDIT][Risk:High][Scope:training_stability]
            # NaN/Inf loss 뒤 optimizer step을 허용하면 seed/affinity decoder 전체가
            # 오염되어 full-volume instance/topology 평가를 해석할 수 없다.
            raise ValueError(
                "BFV3 V253 affinity13-seed loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_v253_loss_ordering]
        # Expected: support/seed/same-affinity logits가 target과 맞고 break-affinity가
        # 낮은 synthetic tensor의 loss가 반대 설정보다 낮으며 backward grad가 finite하다.
        return total


class BoundaryFragmentV3Core025StrongPeakAffinity13ContactHardNegativeSeedHealedHeadLoss(
    BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss
):
    """V254 affinity13 loss with explicit different-fragment/contact hard negatives.

    전제조건:
        output/decoder channel contract는 V253과 동일한 [B,18,Z,Y,X]이다.

    실패 조건:
        same-fragment positive와 different-fragment/contact hard negative의
        shape/device가 어긋나거나 loss가 NaN/Inf가 되면 즉시 중단한다.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        affinity_hard_negative_bce_lambda: float = 1.60,
        affinity_hard_negative_margin_lambda: float = 0.65,
        affinity_hard_negative_threshold_lambda: float = 0.75,
        hard_negative_top_fraction: float = 0.50,
        weak_positive_fraction: float = 0.10,
        hard_negative_margin_logit: float = 0.75,
        hard_negative_ceiling_probability: float = 0.25,
        same_fragment_floor_probability: float = 0.70,
        **kwargs,
    ):
        # [AUDIT][Risk:High][Scope:v254_affinity_supervision]
        # V253 forensic에서 different-fragment/contact edge의 96-97%가 0.50 threshold
        # 위로 join되었다. V254는 채널/decoder를 바꾸지 않고, loss가 보는 negative를
        # 명시적 hard negative edge로 재정의해 0.50 아래로 밀어내는 contract다.
        super().__init__(base_loss, **kwargs)
        self.affinity_hard_negative_bce_lambda = float(affinity_hard_negative_bce_lambda)
        self.affinity_hard_negative_margin_lambda = float(affinity_hard_negative_margin_lambda)
        self.affinity_hard_negative_threshold_lambda = float(affinity_hard_negative_threshold_lambda)
        self.hard_negative_top_fraction = float(hard_negative_top_fraction)
        self.weak_positive_fraction = float(weak_positive_fraction)
        self.hard_negative_margin_logit = float(hard_negative_margin_logit)
        self.hard_negative_ceiling_probability = float(hard_negative_ceiling_probability)
        self.same_fragment_floor_probability = float(same_fragment_floor_probability)
        for name in (
            "affinity_hard_negative_bce_lambda",
            "affinity_hard_negative_margin_lambda",
            "affinity_hard_negative_threshold_lambda",
            "hard_negative_margin_logit",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("hard_negative_top_fraction", "weak_positive_fraction"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {getattr(self, name)}")
        if not 0.0 < self.hard_negative_ceiling_probability < 0.5:
            raise ValueError(
                "hard_negative_ceiling_probability must be in (0, 0.5), "
                f"got {self.hard_negative_ceiling_probability}"
            )
        if not 0.5 < self.same_fragment_floor_probability < 1.0:
            raise ValueError(
                "same_fragment_floor_probability must be in (0.5, 1), "
                f"got {self.same_fragment_floor_probability}"
            )

    @classmethod
    def _same_and_contact_hard_negative_edges(cls, instance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return same-fragment positives and explicit contact/different-fragment negatives."""
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        if instance.ndim != 4:
            raise ValueError(
                "V254 affinity hard-negative target requires instance shape [B,Z,Y,X] "
                f"or [B,1,Z,Y,X], got {tuple(instance.shape)}"
            )
        inst = instance.long()
        invalid = (inst < 0) | (inst > MAX_INSTANCE_ID)
        if invalid.any():
            bad_min = int(inst.min().detach().cpu())
            bad_max = int(inst.max().detach().cpu())
            raise ValueError(f"V254 instance IDs must be in [0, {MAX_INSTANCE_ID}], got min={bad_min} max={bad_max}")
        shape = tuple(int(v) for v in inst.shape[1:])
        edge_shape = (int(inst.shape[0]), len(cls.AFFINITY13_OFFSETS_ZYX), *shape)
        same_edge = torch.zeros(edge_shape, dtype=torch.bool, device=inst.device)
        hard_negative_edge = torch.zeros_like(same_edge)
        valid_edge = torch.zeros_like(same_edge)
        for i, offset in enumerate(cls.AFFINITY13_OFFSETS_ZYX):
            src_sl_zyx, dst_sl_zyx = cls._offset_slices(shape, offset)
            src_sl = (slice(None), *src_sl_zyx)
            dst_sl = (slice(None), *dst_sl_zyx)
            a = inst[src_sl]
            b = inst[dst_sl]
            a_fg = (a > 0) & (a <= MAX_INSTANCE_ID)
            b_fg = (b > 0) & (b <= MAX_INSTANCE_ID)
            same_anatomy = a_fg & b_fg & (((a - 1) // 50) == ((b - 1) // 50))
            same_fragment = same_anatomy & (a == b)
            different_fragment = same_anatomy & (a != b)
            write_sl = (slice(None), i, *src_sl_zyx)
            same_edge[write_sl] = same_fragment
            hard_negative_edge[write_sl] = different_fragment
            valid_edge[write_sl] = same_anatomy
        # [DATA][Risk:High][Scope:v254_hard_negative_definition]
        # same_edge는 GT same-fragment target=1이다. hard_negative_edge는 같은 anatomy
        # 안에서 서로 다른 GT instance가 13-neighborhood offset으로 맞닿은 target=0 edge다.
        # background/support-boundary edge는 decoder contact 분리에 직접 대응하지 않아 제외한다.
        return same_edge, hard_negative_edge, valid_edge

    @staticmethod
    def _top_fraction(values: torch.Tensor, fraction: float, *, largest: bool) -> torch.Tensor:
        if values.numel() == 0:
            return values
        k = max(1, int(math.ceil(float(values.numel()) * float(fraction))))
        return torch.topk(values, k=min(k, int(values.numel())), largest=largest).values

    def _explicit_hard_negative_affinity_bce(
        self,
        affinity_logit: torch.Tensor,
        same_edge: torch.Tensor,
        hard_negative_edge: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        if same_edge.any():
            same_logits = affinity_logit[same_edge].float()
            weak_same = self._top_fraction(same_logits, self.weak_positive_fraction, largest=False)
            losses.append(F.binary_cross_entropy_with_logits(
                weak_same,
                torch.ones_like(weak_same),
                reduction="mean",
            ))
        if hard_negative_edge.any():
            hard_logits = affinity_logit[hard_negative_edge].float()
            hard_tail = self._top_fraction(hard_logits, self.hard_negative_top_fraction, largest=True)
            losses.append(F.binary_cross_entropy_with_logits(
                hard_tail,
                torch.zeros_like(hard_tail),
                reduction="mean",
            ))
        if not losses:
            return affinity_logit.new_zeros(())
        # [AUDIT][Risk:High][Scope:v254_hard_negative_tail]
        # V253은 전체 break-edge 평균이 낮아도 hard-negative tail이 threshold 위에 남았다.
        # V254 BCE는 weak positive tail과 high-logit hard-negative tail을 직접 본다.
        return torch.stack(losses).mean()

    def _explicit_hard_negative_margin_loss(
        self,
        affinity_logit: torch.Tensor,
        same_edge: torch.Tensor,
        hard_negative_edge: torch.Tensor,
    ) -> torch.Tensor:
        if not same_edge.any() or not hard_negative_edge.any():
            return affinity_logit.new_zeros(())
        same_logits = affinity_logit[same_edge].float()
        hard_logits = affinity_logit[hard_negative_edge].float()
        weak_same = self._top_fraction(same_logits, self.weak_positive_fraction, largest=False).mean()
        hard_tail = self._top_fraction(hard_logits, self.hard_negative_top_fraction, largest=True).mean()
        # [METRIC][Scope:affinity_threshold]
        # forensic gate는 q10(same)-q90(different/contact)가 양수가 되는지 본다.
        # 이 loss는 같은 방향으로 logit margin을 강제해 decoder threshold 0.50과 직접 연결된다.
        return F.relu(hard_tail + float(self.hard_negative_margin_logit) - weak_same).pow(2)

    def _explicit_hard_negative_threshold_loss(
        self,
        affinity_logit: torch.Tensor,
        same_edge: torch.Tensor,
        hard_negative_edge: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        if same_edge.any():
            floor_logit = self._probability_to_logit(self.same_fragment_floor_probability)
            same_logits = affinity_logit[same_edge].float()
            weak_same = self._top_fraction(same_logits, self.weak_positive_fraction, largest=False)
            losses.append(F.relu(float(floor_logit) - weak_same).mean())
        if hard_negative_edge.any():
            ceiling_logit = self._probability_to_logit(self.hard_negative_ceiling_probability)
            hard_logits = affinity_logit[hard_negative_edge].float()
            hard_tail = self._top_fraction(hard_logits, self.hard_negative_top_fraction, largest=True)
            losses.append(F.relu(hard_tail - float(ceiling_logit)).mean())
        if not losses:
            return affinity_logit.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        self._validate_affinity13_logits(pred_logits)
        semantic_target = self._semantic_target(target)
        if torch.is_tensor(semantic_target):
            semantic_target = semantic_target.to(device=pred_logits.device)
        labels = self._labels(semantic_target).to(device=pred_logits.device, dtype=torch.long)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_center = self._seed_center_target(target).to(device=pred_logits.device, dtype=torch.float32)
        seed_body = self._seed_body_target(target).to(device=pred_logits.device, dtype=torch.float32)
        if tuple(labels.shape) != tuple(pred_logits.shape[0:1] + pred_logits.shape[2:]):
            raise ValueError(f"V254 label shape mismatch: labels={tuple(labels.shape)} logits={tuple(pred_logits.shape)}")
        if tuple(instance.shape) != tuple(labels.shape):
            raise ValueError(f"V254 instance shape mismatch: instance={tuple(instance.shape)} labels={tuple(labels.shape)}")
        if tuple(seed_center.shape) != tuple(labels.shape) or tuple(seed_body.shape) != tuple(labels.shape):
            raise ValueError(
                f"V254 seed shape mismatch: seed_center={tuple(seed_center.shape)} "
                f"seed_body={tuple(seed_body.shape)} labels={tuple(labels.shape)}"
            )
        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        exterior_target = (labels == 1).float()
        fracture_target = (labels == 2).float()
        same_edge, hard_negative_edge, valid_edge = self._same_and_contact_hard_negative_edges(instance)
        expected_edge = tuple(int(v) for v in pred_logits.shape[0:1] + (13,) + pred_logits.shape[2:])
        if tuple(same_edge.shape) != expected_edge or tuple(hard_negative_edge.shape) != expected_edge:
            raise ValueError(
                f"V254 edge mask shape mismatch: expected={expected_edge}, "
                f"same={tuple(same_edge.shape)} hard_negative={tuple(hard_negative_edge.shape)}"
            )
        if same_edge.device != pred_logits.device or hard_negative_edge.device != pred_logits.device or valid_edge.device != pred_logits.device:
            raise ValueError("V254 affinity edge mask device mismatch")

        base_logits = pred_logits[:, :5].float()
        affinity_logit = pred_logits[:, 5:].float()
        # [QC][Invariant:decoder_contract]
        # channel 5..17은 V253/V254 decoder가 그대로 읽는 13 affinity logit이다.
        # 여기서는 architecture를 바꾸지 않고 hard-negative supervision만 바꾼다.
        total = pred_logits.new_zeros(())
        total = total + self.support_bce_lambda * self._balanced_binary_bce(base_logits[:, 0], support_target, name="support")
        total = total + self.exterior_bce_lambda * self._balanced_binary_bce(base_logits[:, 1], exterior_target, name="exterior")
        total = total + self.fracture_bce_lambda * self._balanced_binary_bce(base_logits[:, 2], fracture_target, name="fracture_surface")
        total = total + self.seed_center_bce_lambda * self._balanced_binary_bce(
            base_logits[:, 3],
            (seed_center > 0.5).float(),
            name="seed_center",
            hard_negative_ratio=32,
        )
        total = total + self.seed_body_bce_lambda * self._balanced_binary_bce(base_logits[:, 4], seed_body, name="seed_body")
        total = total + self.affinity_hard_negative_bce_lambda * self._explicit_hard_negative_affinity_bce(
            affinity_logit,
            same_edge,
            hard_negative_edge,
        )
        total = total + self.affinity_hard_negative_margin_lambda * self._explicit_hard_negative_margin_loss(
            affinity_logit,
            same_edge,
            hard_negative_edge,
        )
        total = total + self.affinity_hard_negative_threshold_lambda * self._explicit_hard_negative_threshold_loss(
            affinity_logit,
            same_edge,
            hard_negative_edge,
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V254 hard-negative affinity13 loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_v254_hard_negative_ordering]
        # Expected: same-fragment logits high / different-fragment-contact logits low 설정의
        # loss가 반대 설정보다 낮고 backward grad가 finite해야 한다.
        return total


class BoundaryFragmentV3Core025StrongPeakMutex13SeedHealedHeadLoss(
    BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss
):
    """V255 loss for attractive affinity + repulsive mutex decoder fields.

    전제조건:
        pred_logits는 [B, 31, Z, Y, X]이다. 0..4는 V253 base head,
        5..17은 same-fragment attractive affinity, 18..30은
        different-fragment/contact repulsive mutex logit이다.

    실패 조건:
        channel contract, shape/device, instance ID 범위, NaN/Inf가 깨지면
        즉시 ValueError를 발생시킨다.
    """

    OUTPUT_CHANNELS = 5 + 2 * len(BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss.AFFINITY13_OFFSETS_ZYX)
    ATTRACTIVE_START = 5
    REPULSIVE_START = 5 + len(BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss.AFFINITY13_OFFSETS_ZYX)

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        attractive_bce_lambda: float = 1.15,
        repulsive_bce_lambda: float = 1.75,
        mutex_threshold_guard_lambda: float = 0.85,
        mutex_consistency_lambda: float = 0.35,
        attractive_same_floor_probability: float = 0.70,
        attractive_contact_ceiling_probability: float = 0.25,
        repulsive_contact_floor_probability: float = 0.70,
        repulsive_same_ceiling_probability: float = 0.25,
        hard_negative_top_fraction: float = 0.50,
        weak_positive_fraction: float = 0.10,
        **kwargs,
    ):
        # [AUDIT][Risk:High][Scope:v255_representation_contract]
        # V254는 하나의 affinity logit에 same-fragment attraction과
        # different-fragment/contact repulsion을 동시에 넣었다. 10e20i forensic에서
        # contact hard negative가 여전히 0.50 위로 전부 join되어, 단일 logit 구조가
        # merge/topology gate에 충분하지 않다는 근거가 생겼다. V255는 decoder가
        # 직접 읽는 repulsive mutex head를 분리해 false merge를 별도 veto로 만든다.
        super().__init__(base_loss, **kwargs)
        self.attractive_bce_lambda = float(attractive_bce_lambda)
        self.repulsive_bce_lambda = float(repulsive_bce_lambda)
        self.mutex_threshold_guard_lambda = float(mutex_threshold_guard_lambda)
        self.mutex_consistency_lambda = float(mutex_consistency_lambda)
        self.attractive_same_floor_probability = float(attractive_same_floor_probability)
        self.attractive_contact_ceiling_probability = float(attractive_contact_ceiling_probability)
        self.repulsive_contact_floor_probability = float(repulsive_contact_floor_probability)
        self.repulsive_same_ceiling_probability = float(repulsive_same_ceiling_probability)
        self.hard_negative_top_fraction = float(hard_negative_top_fraction)
        self.weak_positive_fraction = float(weak_positive_fraction)
        for name in (
            "attractive_bce_lambda",
            "repulsive_bce_lambda",
            "mutex_threshold_guard_lambda",
            "mutex_consistency_lambda",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("hard_negative_top_fraction", "weak_positive_fraction"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {getattr(self, name)}")
        for name in (
            "attractive_same_floor_probability",
            "repulsive_contact_floor_probability",
        ):
            if not 0.5 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in (0.5, 1), got {getattr(self, name)}")
        for name in (
            "attractive_contact_ceiling_probability",
            "repulsive_same_ceiling_probability",
        ):
            if not 0.0 < getattr(self, name) < 0.5:
                raise ValueError(f"{name} must be in (0, 0.5), got {getattr(self, name)}")

    @classmethod
    def _validate_mutex13_logits(cls, pred_logits: torch.Tensor) -> None:
        # [QC][Invariant:channel_order]
        # V255 channel order is fixed by the decoder:
        # 0..4 base heads, 5..17 attractive same-fragment affinity,
        # 18..30 repulsive different-fragment/contact mutex veto.
        if pred_logits.ndim != 5 or int(pred_logits.shape[1]) != int(cls.OUTPUT_CHANNELS):
            raise ValueError(
                f"BFV3 V255 mutex13 loss expects logits [B, {cls.OUTPUT_CHANNELS}, Z, Y, X], "
                f"got {tuple(pred_logits.shape)}"
            )

    @classmethod
    def _same_and_contact_edges(cls, instance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return BoundaryFragmentV3Core025StrongPeakAffinity13ContactHardNegativeSeedHealedHeadLoss._same_and_contact_hard_negative_edges(
            instance
        )

    @staticmethod
    def _top_fraction(values: torch.Tensor, fraction: float, *, largest: bool) -> torch.Tensor:
        return BoundaryFragmentV3Core025StrongPeakAffinity13ContactHardNegativeSeedHealedHeadLoss._top_fraction(
            values,
            fraction,
            largest=largest,
        )

    def _edge_tail_bce(
        self,
        logit: torch.Tensor,
        positive_edge: torch.Tensor,
        negative_edge: torch.Tensor,
        *,
        positive_tail_fraction: float,
        negative_tail_fraction: float,
    ) -> torch.Tensor:
        losses = []
        if positive_edge.any():
            pos_logits = logit[positive_edge].float()
            pos_tail = self._top_fraction(pos_logits, positive_tail_fraction, largest=False)
            losses.append(F.binary_cross_entropy_with_logits(pos_tail, torch.ones_like(pos_tail), reduction="mean"))
        if negative_edge.any():
            neg_logits = logit[negative_edge].float()
            neg_tail = self._top_fraction(neg_logits, negative_tail_fraction, largest=True)
            losses.append(F.binary_cross_entropy_with_logits(neg_tail, torch.zeros_like(neg_tail), reduction="mean"))
        if not losses:
            return logit.new_zeros(())
        return torch.stack(losses).mean()

    def _mutex_threshold_guard(
        self,
        attractive_logit: torch.Tensor,
        repulsive_logit: torch.Tensor,
        same_edge: torch.Tensor,
        contact_edge: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        if same_edge.any():
            attr_floor = self._probability_to_logit(self.attractive_same_floor_probability)
            repel_ceiling = self._probability_to_logit(self.repulsive_same_ceiling_probability)
            weak_attr = self._top_fraction(attractive_logit[same_edge].float(), self.weak_positive_fraction, largest=False)
            hard_repel = self._top_fraction(repulsive_logit[same_edge].float(), self.hard_negative_top_fraction, largest=True)
            losses.append(F.relu(float(attr_floor) - weak_attr).mean())
            losses.append(F.relu(hard_repel - float(repel_ceiling)).mean())
        if contact_edge.any():
            attr_ceiling = self._probability_to_logit(self.attractive_contact_ceiling_probability)
            repel_floor = self._probability_to_logit(self.repulsive_contact_floor_probability)
            hard_attr = self._top_fraction(attractive_logit[contact_edge].float(), self.hard_negative_top_fraction, largest=True)
            weak_repel = self._top_fraction(repulsive_logit[contact_edge].float(), self.weak_positive_fraction, largest=False)
            losses.append(F.relu(hard_attr - float(attr_ceiling)).mean())
            losses.append(F.relu(float(repel_floor) - weak_repel).mean())
        if not losses:
            return attractive_logit.new_zeros(())
        # [METRIC][Scope:task1_merge_topology]
        # Fixed decoder는 attraction>=0.50 및 repulsion<0.50일 때만 join한다.
        # 이 guard는 threshold search가 아니라 Task1 Merge/Topology metric을 위한
        # decoder-facing contract 불변조건이다.
        return torch.stack(losses).mean()

    def _mutex_consistency_loss(
        self,
        attractive_logit: torch.Tensor,
        repulsive_logit: torch.Tensor,
        valid_edge: torch.Tensor,
    ) -> torch.Tensor:
        if not valid_edge.any():
            return attractive_logit.new_zeros(())
        attr = torch.sigmoid(attractive_logit[valid_edge].float())
        repel = torch.sigmoid(repulsive_logit[valid_edge].float())
        # [QC][Invariant:mutual_exclusion]
        # 같은 edge가 동시에 strong attraction/strong repulsion이 되면 decoder가
        # 불안정해진다. 합이 1을 넘는 tail만 벌점으로 처리해 두 head를 분리한다.
        return F.relu(attr + repel - 1.0).mean()

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        self._validate_mutex13_logits(pred_logits)
        semantic_target = self._semantic_target(target)
        if torch.is_tensor(semantic_target):
            semantic_target = semantic_target.to(device=pred_logits.device)
        labels = self._labels(semantic_target).to(device=pred_logits.device, dtype=torch.long)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_center = self._seed_center_target(target).to(device=pred_logits.device, dtype=torch.float32)
        seed_body = self._seed_body_target(target).to(device=pred_logits.device, dtype=torch.float32)
        expected_voxel = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        if tuple(labels.shape) != expected_voxel:
            raise ValueError(f"V255 label shape mismatch: labels={tuple(labels.shape)} logits={tuple(pred_logits.shape)}")
        if tuple(instance.shape) != expected_voxel:
            raise ValueError(f"V255 instance shape mismatch: instance={tuple(instance.shape)} labels={tuple(labels.shape)}")
        if tuple(seed_center.shape) != expected_voxel or tuple(seed_body.shape) != expected_voxel:
            raise ValueError(
                f"V255 seed shape mismatch: seed_center={tuple(seed_center.shape)} "
                f"seed_body={tuple(seed_body.shape)} labels={tuple(labels.shape)}"
            )

        same_edge, contact_edge, valid_edge = self._same_and_contact_edges(instance)
        expected_edge = tuple(int(v) for v in pred_logits.shape[0:1] + (13,) + pred_logits.shape[2:])
        if tuple(same_edge.shape) != expected_edge or tuple(contact_edge.shape) != expected_edge:
            raise ValueError(
                f"V255 edge mask shape mismatch: expected={expected_edge}, "
                f"same={tuple(same_edge.shape)} contact={tuple(contact_edge.shape)}"
            )
        if same_edge.device != pred_logits.device or contact_edge.device != pred_logits.device:
            raise ValueError("V255 mutex edge mask device mismatch")

        base_logits = pred_logits[:, :5].float()
        attractive_logit = pred_logits[:, self.ATTRACTIVE_START:self.REPULSIVE_START].float()
        repulsive_logit = pred_logits[:, self.REPULSIVE_START:].float()
        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        exterior_target = (labels == 1).float()
        fracture_target = (labels == 2).float()

        # [DATA][Risk:High][Scope:v255_edge_targets]
        # same_edge는 attraction=1/repulsion=0이고, contact_edge는
        # attraction=0/repulsion=1이다. background/anatomy-boundary edge는
        # Task1 same-anatomy fragment partition에 직접 대응하지 않아 edge loss에서 제외한다.
        total = pred_logits.new_zeros(())
        total = total + self.support_bce_lambda * self._balanced_binary_bce(base_logits[:, 0], support_target, name="support")
        total = total + self.exterior_bce_lambda * self._balanced_binary_bce(base_logits[:, 1], exterior_target, name="exterior")
        total = total + self.fracture_bce_lambda * self._balanced_binary_bce(base_logits[:, 2], fracture_target, name="fracture_surface")
        total = total + self.seed_center_bce_lambda * self._balanced_binary_bce(
            base_logits[:, 3],
            (seed_center > 0.5).float(),
            name="seed_center",
            hard_negative_ratio=32,
        )
        total = total + self.seed_body_bce_lambda * self._balanced_binary_bce(base_logits[:, 4], seed_body, name="seed_body")
        total = total + self.attractive_bce_lambda * self._edge_tail_bce(
            attractive_logit,
            same_edge,
            contact_edge,
            positive_tail_fraction=self.weak_positive_fraction,
            negative_tail_fraction=self.hard_negative_top_fraction,
        )
        total = total + self.repulsive_bce_lambda * self._edge_tail_bce(
            repulsive_logit,
            contact_edge,
            same_edge,
            positive_tail_fraction=self.weak_positive_fraction,
            negative_tail_fraction=self.hard_negative_top_fraction,
        )
        total = total + self.mutex_threshold_guard_lambda * self._mutex_threshold_guard(
            attractive_logit,
            repulsive_logit,
            same_edge,
            contact_edge,
        )
        total = total + self.mutex_consistency_lambda * self._mutex_consistency_loss(
            attractive_logit,
            repulsive_logit,
            valid_edge,
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V255 mutex13 loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_v255_mutex_ordering]
        # Expected: same edge는 attractive high/repulsive low, contact edge는
        # attractive low/repulsive high일 때 loss가 반대 설정보다 낮아야 한다.
        return total


class BoundaryFragmentV3Core025StrongPeakMutex13SupportLeakGuardSeedHealedHeadLoss(
    BoundaryFragmentV3Core025StrongPeakMutex13SeedHealedHeadLoss
):
    """V256 loss: V255 mutex contract plus decoder-visible support-leak edges.

    전제조건:
        출력 채널과 decoder는 V255와 동일한 [B,31,Z,Y,X] contract를 사용한다.

    실패 조건:
        support 밖 edge가 학습에서 빠지면 learned decoder는 false-support 영역을
        전부 join할 수 있다. 이 loss는 그 edge를 deterministic hard negative로
        포함해 Merge/Topology gate 실패를 조기에 드러낸다.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        support_bce_lambda: float = 1.05,
        attractive_bce_lambda: float = 1.25,
        repulsive_bce_lambda: float = 2.25,
        mutex_threshold_guard_lambda: float = 1.25,
        mutex_consistency_lambda: float = 0.35,
        leak_edge_bce_lambda: float = 1.10,
        leak_edge_threshold_lambda: float = 0.70,
        support_threshold_guard_lambda: float = 0.65,
        leak_edge_negative_ratio: int = 4,
        repulsive_contact_positive_fraction: float = 1.0,
        support_floor_probability: float = 0.70,
        support_ceiling_probability: float = 0.25,
        **kwargs,
    ):
        # [AUDIT][Risk:High][Scope:v256_decoder_train_gap]
        # V255 oracle는 통과했지만 learned 10e20i는 support가 GT보다 2.2x-3.4x 넓고
        # `mutex_veto_edges=0`이었다. 원인은 loss가 GT same/contact edge만 보고,
        # decoder가 실제로 join하는 false-support/background edge를 감독하지 않는 gap이다.
        # V256은 output/decoder는 유지하고 support 밖 decoder-visible edge만 hard negative로 추가한다.
        super().__init__(
            base_loss,
            support_bce_lambda=support_bce_lambda,
            attractive_bce_lambda=attractive_bce_lambda,
            repulsive_bce_lambda=repulsive_bce_lambda,
            mutex_threshold_guard_lambda=mutex_threshold_guard_lambda,
            mutex_consistency_lambda=mutex_consistency_lambda,
            **kwargs,
        )
        self.leak_edge_bce_lambda = float(leak_edge_bce_lambda)
        self.leak_edge_threshold_lambda = float(leak_edge_threshold_lambda)
        self.support_threshold_guard_lambda = float(support_threshold_guard_lambda)
        self.leak_edge_negative_ratio = int(leak_edge_negative_ratio)
        self.repulsive_contact_positive_fraction = float(repulsive_contact_positive_fraction)
        self.support_floor_probability = float(support_floor_probability)
        self.support_ceiling_probability = float(support_ceiling_probability)
        for name in ("leak_edge_bce_lambda", "leak_edge_threshold_lambda", "support_threshold_guard_lambda"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.leak_edge_negative_ratio < 1:
            raise ValueError(f"leak_edge_negative_ratio must be >= 1, got {self.leak_edge_negative_ratio}")
        if not 0.0 < self.repulsive_contact_positive_fraction <= 1.0:
            raise ValueError(
                "repulsive_contact_positive_fraction must be in (0, 1], "
                f"got {self.repulsive_contact_positive_fraction}"
            )
        if not 0.5 < self.support_floor_probability < 1.0:
            raise ValueError(f"support_floor_probability must be in (0.5, 1), got {self.support_floor_probability}")
        if not 0.0 < self.support_ceiling_probability < 0.5:
            raise ValueError(f"support_ceiling_probability must be in (0, 0.5), got {self.support_ceiling_probability}")

    @classmethod
    def _same_contact_leak_edges(
        cls,
        instance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return same/contact/leak edge masks under the V255 13-offset contract."""
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        if instance.ndim != 4:
            raise ValueError(
                "V256 mutex leak target requires instance shape [B,Z,Y,X] or [B,1,Z,Y,X], "
                f"got {tuple(instance.shape)}"
            )
        inst = instance.long()
        invalid = (inst < 0) | (inst > MAX_INSTANCE_ID)
        if invalid.any():
            bad_min = int(inst.min().detach().cpu())
            bad_max = int(inst.max().detach().cpu())
            raise ValueError(f"V256 instance IDs must be in [0, {MAX_INSTANCE_ID}], got min={bad_min} max={bad_max}")
        shape = tuple(int(v) for v in inst.shape[1:])
        edge_shape = (int(inst.shape[0]), len(cls.AFFINITY13_OFFSETS_ZYX), *shape)
        same_edge = torch.zeros(edge_shape, dtype=torch.bool, device=inst.device)
        contact_edge = torch.zeros_like(same_edge)
        leak_edge = torch.zeros_like(same_edge)
        edge_universe = torch.zeros_like(same_edge)
        for i, offset in enumerate(cls.AFFINITY13_OFFSETS_ZYX):
            src_sl_zyx, dst_sl_zyx = cls._offset_slices(shape, offset)
            src_sl = (slice(None), *src_sl_zyx)
            dst_sl = (slice(None), *dst_sl_zyx)
            a = inst[src_sl]
            b = inst[dst_sl]
            a_fg = (a > 0) & (a <= MAX_INSTANCE_ID)
            b_fg = (b > 0) & (b <= MAX_INSTANCE_ID)
            same_anatomy = a_fg & b_fg & (((a - 1) // 50) == ((b - 1) // 50))
            same_fragment = same_anatomy & (a == b)
            different_fragment = same_anatomy & (a != b)
            write_sl = (slice(None), i, *src_sl_zyx)
            same_edge[write_sl] = same_fragment
            contact_edge[write_sl] = different_fragment
            # [DATA][Risk:High][Scope:false_support_edges]
            # V255의 blind spot은 GT support 내부 contact가 아니라 GT 밖 false-support다.
            # decoder는 예측 support 위의 모든 edge를 valid로 보므로, same/contact가 아닌
            # in-volume edge는 attraction=0 및 repulsion=1 후보 hard negative가 된다.
            leak_edge[write_sl] = ~(same_fragment | different_fragment)
            edge_universe[write_sl] = True
        return same_edge, contact_edge, leak_edge, edge_universe

    def _select_leak_hard_negatives(
        self,
        attractive_logit: torch.Tensor,
        repulsive_logit: torch.Tensor,
        leak_edge: torch.Tensor,
        *,
        reference_count: int,
    ) -> torch.Tensor:
        if not leak_edge.any() or reference_count <= 0:
            return torch.zeros_like(leak_edge)
        risk = (attractive_logit.float() - repulsive_logit.float())[leak_edge]
        if risk.numel() == 0:
            return torch.zeros_like(leak_edge)
        k = min(int(risk.numel()), max(1, int(reference_count) * int(self.leak_edge_negative_ratio)))
        _, local_idx = torch.topk(risk, k=k, largest=True)
        selected_flat = torch.zeros(int(risk.numel()), dtype=torch.bool, device=risk.device)
        selected_flat[local_idx] = True
        selected = torch.zeros_like(leak_edge)
        selected[leak_edge] = selected_flat
        # [AUDIT][Risk:High][Scope:deterministic_hard_negative_mining]
        # background/background edge 전체를 평균하면 쉬운 negative가 압도한다. 여기서는
        # decoder join 위험도(attr_logit - repulsive_logit)가 높은 leak edge만 top-k로
        # 고정 선택해 random sampling 없이 재현 가능한 hard-negative mining을 한다.
        return selected

    def _support_threshold_guard_loss(
        self,
        support_logit: torch.Tensor,
        support_target: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        pos = support_target > 0.5
        neg = ~pos
        if pos.any():
            floor = self._probability_to_logit(self.support_floor_probability)
            weak_pos = self._top_fraction(support_logit[pos].float(), self.weak_positive_fraction, largest=False)
            losses.append(F.relu(float(floor) - weak_pos).mean())
        if neg.any():
            ceiling = self._probability_to_logit(self.support_ceiling_probability)
            hard_neg = self._top_fraction(support_logit[neg].float(), self.hard_negative_top_fraction, largest=True)
            losses.append(F.relu(hard_neg - float(ceiling)).mean())
        if not losses:
            return support_logit.new_zeros(())
        # [METRIC][Scope:support_ratio]
        # Task1 HD95/ASSD와 decoder runtime은 support over-prediction에 직접 민감하다.
        # hard3 gate에서 관찰된 2.2x-3.4x support ratio를 줄이기 위한 threshold guard다.
        return torch.stack(losses).mean()

    def _leak_edge_decoder_loss(
        self,
        attractive_logit: torch.Tensor,
        repulsive_logit: torch.Tensor,
        selected_leak_edge: torch.Tensor,
    ) -> torch.Tensor:
        if not selected_leak_edge.any():
            return attractive_logit.new_zeros(())
        attr = attractive_logit[selected_leak_edge].float()
        repel = repulsive_logit[selected_leak_edge].float()
        bce = 0.5 * (
            F.binary_cross_entropy_with_logits(attr, torch.zeros_like(attr), reduction="mean")
            + F.binary_cross_entropy_with_logits(repel, torch.ones_like(repel), reduction="mean")
        )
        attr_ceiling = self._probability_to_logit(self.attractive_contact_ceiling_probability)
        repel_floor = self._probability_to_logit(self.repulsive_contact_floor_probability)
        threshold = 0.5 * (
            F.relu(attr - float(attr_ceiling)).mean()
            + F.relu(float(repel_floor) - repel).mean()
        )
        # [METRIC][Scope:decoder_threshold_050]
        # selected leak edge는 predicted support가 잘못 넓어졌을 때 decoder가 실제로
        # join할 수 있는 edge다. attraction은 0.50 아래, repulsion은 0.50 위로
        # 보내야 `mutex_veto_edges`가 0으로 붕괴하지 않는다.
        return bce + self.leak_edge_threshold_lambda * threshold

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        self._validate_mutex13_logits(pred_logits)
        semantic_target = self._semantic_target(target)
        if torch.is_tensor(semantic_target):
            semantic_target = semantic_target.to(device=pred_logits.device)
        labels = self._labels(semantic_target).to(device=pred_logits.device, dtype=torch.long)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_center = self._seed_center_target(target).to(device=pred_logits.device, dtype=torch.float32)
        seed_body = self._seed_body_target(target).to(device=pred_logits.device, dtype=torch.float32)
        expected_voxel = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        # [QC][Invariant:shape_dtype_device]
        # logits/semantic/instance/seed는 모두 같은 batch/spatial shape와 device를 가져야
        # edge mining이 decoder와 같은 좌표계를 보장한다.
        if tuple(labels.shape) != expected_voxel:
            raise ValueError(f"V256 label shape mismatch: labels={tuple(labels.shape)} logits={tuple(pred_logits.shape)}")
        if tuple(instance.shape) != expected_voxel:
            raise ValueError(f"V256 instance shape mismatch: instance={tuple(instance.shape)} labels={tuple(labels.shape)}")
        if tuple(seed_center.shape) != expected_voxel or tuple(seed_body.shape) != expected_voxel:
            raise ValueError(
                f"V256 seed shape mismatch: seed_center={tuple(seed_center.shape)} "
                f"seed_body={tuple(seed_body.shape)} labels={tuple(labels.shape)}"
            )

        same_edge, contact_edge, leak_edge, edge_universe = self._same_contact_leak_edges(instance)
        expected_edge = tuple(int(v) for v in pred_logits.shape[0:1] + (13,) + pred_logits.shape[2:])
        if tuple(same_edge.shape) != expected_edge or tuple(contact_edge.shape) != expected_edge or tuple(leak_edge.shape) != expected_edge:
            raise ValueError(
                f"V256 edge mask shape mismatch: expected={expected_edge}, "
                f"same={tuple(same_edge.shape)} contact={tuple(contact_edge.shape)} leak={tuple(leak_edge.shape)}"
            )
        if same_edge.device != pred_logits.device or contact_edge.device != pred_logits.device or leak_edge.device != pred_logits.device:
            raise ValueError("V256 mutex edge mask device mismatch")

        base_logits = pred_logits[:, :5].float()
        attractive_logit = pred_logits[:, self.ATTRACTIVE_START:self.REPULSIVE_START].float()
        repulsive_logit = pred_logits[:, self.REPULSIVE_START:].float()
        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        exterior_target = (labels == 1).float()
        fracture_target = (labels == 2).float()
        reference_edges = int((same_edge | contact_edge).sum().detach().cpu())
        selected_leak_edge = self._select_leak_hard_negatives(
            attractive_logit,
            repulsive_logit,
            leak_edge & edge_universe,
            reference_count=reference_edges,
        )

        total = pred_logits.new_zeros(())
        total = total + self.support_bce_lambda * self._balanced_binary_bce(base_logits[:, 0], support_target, name="support")
        total = total + self.support_threshold_guard_lambda * self._support_threshold_guard_loss(base_logits[:, 0], support_target)
        total = total + self.exterior_bce_lambda * self._balanced_binary_bce(base_logits[:, 1], exterior_target, name="exterior")
        total = total + self.fracture_bce_lambda * self._balanced_binary_bce(base_logits[:, 2], fracture_target, name="fracture_surface")
        total = total + self.seed_center_bce_lambda * self._balanced_binary_bce(
            base_logits[:, 3],
            (seed_center > 0.5).float(),
            name="seed_center",
            hard_negative_ratio=32,
        )
        total = total + self.seed_body_bce_lambda * self._balanced_binary_bce(base_logits[:, 4], seed_body, name="seed_body")
        total = total + self.attractive_bce_lambda * self._edge_tail_bce(
            attractive_logit,
            same_edge,
            contact_edge,
            positive_tail_fraction=self.weak_positive_fraction,
            negative_tail_fraction=self.hard_negative_top_fraction,
        )
        total = total + self.repulsive_bce_lambda * self._edge_tail_bce(
            repulsive_logit,
            contact_edge,
            same_edge,
            positive_tail_fraction=self.repulsive_contact_positive_fraction,
            negative_tail_fraction=self.hard_negative_top_fraction,
        )
        total = total + self.leak_edge_bce_lambda * self._leak_edge_decoder_loss(
            attractive_logit,
            repulsive_logit,
            selected_leak_edge,
        )
        total = total + self.mutex_threshold_guard_lambda * self._mutex_threshold_guard(
            attractive_logit,
            repulsive_logit,
            same_edge,
            contact_edge,
        )
        total = total + self.mutex_consistency_lambda * self._mutex_consistency_loss(
            attractive_logit,
            repulsive_logit,
            same_edge | contact_edge | selected_leak_edge,
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V256 mutex13 support-leak loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_v256_support_leak_guard]
        # Expected: same high/contact+leak repulsive high/support calibrated logits의
        # loss가 inverted logits보다 낮고, selected leak edge 수는
        # leak_edge_negative_ratio * (same+contact) 이하로 deterministic해야 한다.
        return total


class BoundaryFragmentV3Core025StrongPeakMutex13SeedPeakSupportLeakGuardSeedHealedHeadLoss(
    BoundaryFragmentV3Core025StrongPeakMutex13SupportLeakGuardSeedHealedHeadLoss
):
    """V258 loss: train seed_center as a decoder peak field for NMS markers.

    전제조건:
        output/decoder channel contract는 V255/V256과 같은 [B,31,Z,Y,X]를 유지한다.
        V258 decoder는 channel3 seed_center의 local peak를 marker로 쓰고, channel4
        seed_body threshold mask에는 의존하지 않는다.

    실패 조건:
        seed_center peak가 fragment별로 하나씩 서지 않거나 false peak tail이 넓게
        남으면 Task1 Instance F1/Precision/Recall, Merge/Split, Topology gate가 실패한다.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        seed_center_bce_lambda: float = 1.20,
        seed_body_bce_lambda: float = 0.10,
        seed_peak_presence_lambda: float = 1.50,
        seed_peak_rank_lambda: float = 1.00,
        seed_peak_false_tail_lambda: float = 1.25,
        seed_peak_mass_cap_lambda: float = 0.70,
        seed_peak_floor_probability: float = 0.85,
        seed_peak_false_ceiling_probability: float = 0.15,
        seed_peak_rank_margin_logit: float = 1.25,
        seed_peak_false_negative_ratio: int = 128,
        seed_peak_mass_cap_ratio: float = 16.0,
        **kwargs,
    ):
        # [AUDIT][Risk:High][Scope:v258_seed_peak_contract]
        # V257에서 channel4 threshold seed는 fold0=0개, fold1=과다 seed로 갈라졌다.
        # V258은 같은 31채널 mutex graph를 유지하되 decoder marker를 channel3 local
        # peak로 재정의한다. loss도 absolute mask mass보다 peak 순위/false tail을
        # 직접 감독해 Task1 instance/topology 지표에 맞춘다.
        super().__init__(
            base_loss,
            seed_center_bce_lambda=seed_center_bce_lambda,
            seed_body_bce_lambda=seed_body_bce_lambda,
            **kwargs,
        )
        self.seed_peak_presence_lambda = float(seed_peak_presence_lambda)
        self.seed_peak_rank_lambda = float(seed_peak_rank_lambda)
        self.seed_peak_false_tail_lambda = float(seed_peak_false_tail_lambda)
        self.seed_peak_mass_cap_lambda = float(seed_peak_mass_cap_lambda)
        self.seed_peak_floor_probability = float(seed_peak_floor_probability)
        self.seed_peak_false_ceiling_probability = float(seed_peak_false_ceiling_probability)
        self.seed_peak_rank_margin_logit = float(seed_peak_rank_margin_logit)
        self.seed_peak_false_negative_ratio = int(seed_peak_false_negative_ratio)
        self.seed_peak_mass_cap_ratio = float(seed_peak_mass_cap_ratio)
        for name in (
            "seed_peak_presence_lambda",
            "seed_peak_rank_lambda",
            "seed_peak_false_tail_lambda",
            "seed_peak_mass_cap_lambda",
            "seed_peak_rank_margin_logit",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.5 < self.seed_peak_floor_probability < 1.0:
            raise ValueError(
                "seed_peak_floor_probability must be in (0.5, 1), "
                f"got {self.seed_peak_floor_probability}"
            )
        if not 0.0 < self.seed_peak_false_ceiling_probability < 0.5:
            raise ValueError(
                "seed_peak_false_ceiling_probability must be in (0, 0.5), "
                f"got {self.seed_peak_false_ceiling_probability}"
            )
        if self.seed_peak_false_negative_ratio < 1:
            raise ValueError(
                "seed_peak_false_negative_ratio must be >= 1, "
                f"got {self.seed_peak_false_negative_ratio}"
            )
        if self.seed_peak_mass_cap_ratio <= 1.0:
            raise ValueError(f"seed_peak_mass_cap_ratio must be > 1, got {self.seed_peak_mass_cap_ratio}")

    def _seed_peak_presence_loss(self, seed_logit: torch.Tensor, seed_center: torch.Tensor) -> torch.Tensor:
        center_mask = seed_center > 0.5
        if not center_mask.any():
            return seed_logit.new_zeros(())
        floor = self._probability_to_logit(self.seed_peak_floor_probability)
        center_logits = seed_logit[center_mask].float()
        # [DATA][Risk:High][Scope:seed_peak_positive]
        # seed_center target은 fragment당 1 voxel이다. decoder NMS가 이 peak를 marker로
        # 읽으므로 positive 평균이 아니라 가장 약한 center logit까지 0.50 위로 올라와야
        # Instance Recall/Topology가 닫히지 않는다.
        return F.relu(float(floor) - center_logits).mean()

    def _seed_peak_false_tail_loss(self, seed_logit: torch.Tensor, seed_center: torch.Tensor) -> torch.Tensor:
        center_mask = seed_center > 0.5
        center_count = int(center_mask.sum().detach().cpu())
        if center_count <= 0:
            return seed_logit.new_zeros(())
        false_logits = seed_logit[~center_mask].float()
        if false_logits.numel() == 0:
            return seed_logit.new_zeros(())
        k = min(
            int(false_logits.numel()),
            max(1, center_count * int(self.seed_peak_false_negative_ratio)),
        )
        hard_false = torch.topk(false_logits, k=k, largest=True).values
        ceiling = self._probability_to_logit(self.seed_peak_false_ceiling_probability)
        bce = F.binary_cross_entropy_with_logits(hard_false, torch.zeros_like(hard_false), reduction="mean")
        threshold = F.relu(hard_false - float(ceiling)).mean()
        # [AUDIT][Risk:High][Scope:false_seed_peak_tail]
        # V257 fold1은 channel3이 0.50 위로 넓게 떠서 seed_union 294k voxels를 만들었다.
        # V258 decoder는 NMS로 marker 수를 제한하지만, false local peak가 많으면
        # Split/Precision/Topology가 여전히 실패하므로 high-logit false tail을 직접 억제한다.
        return 0.5 * bce + 0.5 * threshold

    def _seed_peak_rank_loss(
        self,
        seed_logit: torch.Tensor,
        seed_center: torch.Tensor,
        instance: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for batch_idx in range(int(seed_logit.shape[0])):
            ids = torch.unique(instance[batch_idx])
            ids = ids[(ids > 0) & (ids <= MAX_INSTANCE_ID)]
            for frag_id in ids.tolist():
                frag = instance[batch_idx] == int(frag_id)
                center = frag & (seed_center[batch_idx] > 0.5)
                non_center = frag & ~center
                if not center.any() or not non_center.any():
                    continue
                pos = seed_logit[batch_idx][center].float().max()
                hard_false = self._top_fraction(
                    seed_logit[batch_idx][non_center].float(),
                    self.weak_positive_fraction,
                    largest=True,
                ).mean()
                losses.append(F.relu(float(self.seed_peak_rank_margin_logit) - (pos - hard_false)))
        if not losses:
            return seed_logit.new_zeros(())
        # [METRIC][Scope:task1_split_merge_topology]
        # local peak decoder는 한 fragment 내부에서 center peak가 주변 false peak보다
        # 충분히 높다는 순위 계약을 필요로 한다. 이 margin이 없으면 같은 fragment 안의
        # 작은 noise peak가 추가 marker가 되어 Split/Topology 지표를 망친다.
        return torch.stack(losses).mean()

    def _seed_peak_mass_cap_loss(
        self,
        seed_logit: torch.Tensor,
        seed_center: torch.Tensor,
        support_target: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        seed_prob = torch.sigmoid(seed_logit.float())
        for batch_idx in range(int(seed_prob.shape[0])):
            support = support_target[batch_idx] > 0.5
            center_count = seed_center[batch_idx][support].float().sum()
            if float(center_count.detach().cpu()) <= 0.0:
                continue
            pred_mass = seed_prob[batch_idx][support].sum()
            cap = float(self.seed_peak_mass_cap_ratio) * center_count
            losses.append(F.relu(pred_mass - cap) / torch.clamp(center_count, min=1.0))
        if not losses:
            return seed_logit.new_zeros(())
        # [METRIC][Scope:seed_peak_mass]
        # seed_center는 dense mask가 아니라 peak field다. support 내부 확률 질량이
        # center 수 대비 과하면 NMS 전 후보가 폭증하고 Task1 proxy eval runtime과
        # Split/Precision이 동시에 악화된다.
        return torch.stack(losses).mean()

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        total = super().forward(pred_logits, target)
        semantic_target = self._semantic_target(target)
        if torch.is_tensor(semantic_target):
            semantic_target = semantic_target.to(device=pred_logits.device)
        labels = self._labels(semantic_target).to(device=pred_logits.device, dtype=torch.long)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_center = self._seed_center_target(target).to(device=pred_logits.device, dtype=torch.float32)
        expected_voxel = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        # [QC][Invariant:shape_dtype_device]
        # V258 추가 loss는 channel3 seed_center와 instance ID가 같은 voxel grid에
        # 놓여 있다는 전제에서 fragment-wise rank를 계산한다. shape mismatch를
        # 허용하면 잘못된 fragment peak가 학습되어 fulltrain gate가 무의미해진다.
        if tuple(labels.shape) != expected_voxel or tuple(instance.shape) != expected_voxel:
            raise ValueError(
                f"V258 target shape mismatch: labels={tuple(labels.shape)} "
                f"instance={tuple(instance.shape)} logits={tuple(pred_logits.shape)}"
            )
        if tuple(seed_center.shape) != expected_voxel:
            raise ValueError(
                f"V258 seed_center shape mismatch: seed_center={tuple(seed_center.shape)} "
                f"logits={tuple(pred_logits.shape)}"
            )

        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        seed_logit = pred_logits[:, 3].float()
        total = total + self.seed_peak_presence_lambda * self._seed_peak_presence_loss(seed_logit, seed_center)
        total = total + self.seed_peak_false_tail_lambda * self._seed_peak_false_tail_loss(seed_logit, seed_center)
        total = total + self.seed_peak_rank_lambda * self._seed_peak_rank_loss(seed_logit, seed_center, instance)
        total = total + self.seed_peak_mass_cap_lambda * self._seed_peak_mass_cap_loss(
            seed_logit,
            seed_center,
            support_target,
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V258 mutex13 seed-peak support-leak loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_v258_seed_peak]
        # Expected: center peak high/false peak low logits should score lower than
        # missing-center or broad false-peak logits, with finite backward gradients.
        return total


class BoundaryFragmentV3Core025StrongPeakMutex13SeedPeakBodySupportLeakGuardSeedHealedHeadLoss(
    BoundaryFragmentV3Core025StrongPeakMutex13SeedPeakSupportLeakGuardSeedHealedHeadLoss
):
    """V259 loss: train channel3 peak decoder from compact seed-body positives.

    [AUDIT][Risk:High][Scope:v259_seed_target_density]
    V258 made the decoder robust to component explosion, but learned channel3 was
    suppressed to near zero and every ROI used top1 fallback. The one-voxel
    seed_center target is too sparse for the current short gate. V259 keeps the
    same V258 peak-NMS decoder and trains channel3 from compact seed_body
    positives, while still requiring the exact center voxel to be high.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        seed_center_bce_lambda: float = 0.0,
        seed_body_bce_lambda: float = 0.10,
        seed_peak_body_bce_lambda: float = 1.40,
        seed_peak_center_floor_lambda: float = 0.80,
        seed_peak_body_false_tail_lambda: float = 1.10,
        seed_peak_body_mass_cap_lambda: float = 0.45,
        seed_peak_body_floor_probability: float = 0.75,
        seed_peak_body_false_ceiling_probability: float = 0.20,
        seed_peak_body_negative_ratio: int = 64,
        seed_peak_body_mass_cap_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__(
            base_loss,
            seed_center_bce_lambda=seed_center_bce_lambda,
            seed_body_bce_lambda=seed_body_bce_lambda,
            **kwargs,
        )
        self.seed_peak_body_bce_lambda = float(seed_peak_body_bce_lambda)
        self.seed_peak_center_floor_lambda = float(seed_peak_center_floor_lambda)
        self.seed_peak_body_false_tail_lambda = float(seed_peak_body_false_tail_lambda)
        self.seed_peak_body_mass_cap_lambda = float(seed_peak_body_mass_cap_lambda)
        self.seed_peak_body_floor_probability = float(seed_peak_body_floor_probability)
        self.seed_peak_body_false_ceiling_probability = float(seed_peak_body_false_ceiling_probability)
        self.seed_peak_body_negative_ratio = int(seed_peak_body_negative_ratio)
        self.seed_peak_body_mass_cap_ratio = float(seed_peak_body_mass_cap_ratio)
        for name in (
            "seed_peak_body_bce_lambda",
            "seed_peak_center_floor_lambda",
            "seed_peak_body_false_tail_lambda",
            "seed_peak_body_mass_cap_lambda",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.5 < self.seed_peak_body_floor_probability < 1.0:
            raise ValueError(
                "seed_peak_body_floor_probability must be in (0.5, 1), "
                f"got {self.seed_peak_body_floor_probability}"
            )
        if not 0.0 < self.seed_peak_body_false_ceiling_probability < 0.5:
            raise ValueError(
                "seed_peak_body_false_ceiling_probability must be in (0, 0.5), "
                f"got {self.seed_peak_body_false_ceiling_probability}"
            )
        if self.seed_peak_body_negative_ratio < 1:
            raise ValueError(f"seed_peak_body_negative_ratio must be >= 1, got {self.seed_peak_body_negative_ratio}")
        if self.seed_peak_body_mass_cap_ratio <= 1.0:
            raise ValueError(f"seed_peak_body_mass_cap_ratio must be > 1, got {self.seed_peak_body_mass_cap_ratio}")

    def _seed_peak_body_bce_loss(self, seed_logit: torch.Tensor, seed_body: torch.Tensor) -> torch.Tensor:
        body_target = (seed_body > 0.5).float()
        # [DATA][Risk:High][Scope:v259_peak_body_positive]
        # channel3 decoder peak는 one-hot center가 아니라 compact seed_body 안에서
        # 배울 수 있는 ridge/peak field여야 한다. 이 BCE는 positive exposure를 늘리되
        # decoder marker는 eval에서 NMS로 하나씩 제한한다.
        return self._balanced_binary_bce(
            seed_logit,
            body_target,
            name="seed_peak_body",
            hard_negative_ratio=int(self.seed_peak_body_negative_ratio),
        )

    def _seed_peak_body_floor_loss(
        self,
        seed_logit: torch.Tensor,
        seed_center: torch.Tensor,
        seed_body: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        body = seed_body > 0.5
        if body.any():
            floor = self._probability_to_logit(self.seed_peak_body_floor_probability)
            weak_body = self._top_fraction(seed_logit[body].float(), self.weak_positive_fraction, largest=False)
            losses.append(F.relu(float(floor) - weak_body).mean())
        center = seed_center > 0.5
        if center.any():
            center_floor = self._probability_to_logit(max(self.seed_peak_body_floor_probability, 0.85))
            losses.append(F.relu(float(center_floor) - seed_logit[center].float()).mean())
        if not losses:
            return seed_logit.new_zeros(())
        # [METRIC][Scope:seed_peak_threshold_050]
        # V258 fold0은 모든 ROI가 top1 fallback을 사용했다. V259는 compact body와
        # exact center가 0.50 위로 올라오도록 해 fallback 없이 local peak marker가
        # 생기는지를 fulltrain gate 전에 확인한다.
        return torch.stack(losses).mean()

    def _seed_peak_body_false_tail_loss(self, seed_logit: torch.Tensor, seed_body: torch.Tensor) -> torch.Tensor:
        body = seed_body > 0.5
        positive_count = int(body.sum().detach().cpu())
        if positive_count <= 0:
            return seed_logit.new_zeros(())
        false_logits = seed_logit[~body].float()
        if false_logits.numel() == 0:
            return seed_logit.new_zeros(())
        k = min(
            int(false_logits.numel()),
            max(1, positive_count * int(self.seed_peak_body_negative_ratio)),
        )
        hard_false = torch.topk(false_logits, k=k, largest=True).values
        ceiling = self._probability_to_logit(self.seed_peak_body_false_ceiling_probability)
        return F.relu(hard_false - float(ceiling)).mean()

    def _seed_peak_body_mass_cap_loss(
        self,
        seed_logit: torch.Tensor,
        seed_body: torch.Tensor,
        support_target: torch.Tensor,
    ) -> torch.Tensor:
        seed_prob = torch.sigmoid(seed_logit.float())
        losses = []
        for batch_idx in range(int(seed_prob.shape[0])):
            support = support_target[batch_idx] > 0.5
            target_mass = seed_body[batch_idx][support].float().sum()
            if float(target_mass.detach().cpu()) <= 0.0:
                continue
            pred_mass = seed_prob[batch_idx][support].sum()
            cap = float(self.seed_peak_body_mass_cap_ratio) * target_mass
            losses.append(F.relu(pred_mass - cap) / torch.clamp(target_mass, min=1.0))
        if not losses:
            return seed_logit.new_zeros(())
        # [QC][Risk:Major][Scope:seed_peak_mass_cap]
        # compact body target는 one-hot보다 넓지만 여전히 marker field다. 질량 cap이
        # 없으면 V257 fold1처럼 broad seed가 생기고 NMS 후보가 많아져 Split/Topology가 흔들린다.
        return torch.stack(losses).mean()

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        total = BoundaryFragmentV3Core025StrongPeakMutex13SupportLeakGuardSeedHealedHeadLoss.forward(
            self,
            pred_logits,
            target,
        )
        semantic_target = self._semantic_target(target)
        if torch.is_tensor(semantic_target):
            semantic_target = semantic_target.to(device=pred_logits.device)
        labels = self._labels(semantic_target).to(device=pred_logits.device, dtype=torch.long)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_center = self._seed_center_target(target).to(device=pred_logits.device, dtype=torch.float32)
        seed_body = self._seed_body_target(target).to(device=pred_logits.device, dtype=torch.float32)
        expected_voxel = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        # [QC][Invariant:shape_dtype_device]
        # V259는 channel3을 seed_body target으로 학습하지만, exact center guard도 같이
        # 계산한다. 두 target이 같은 voxel grid가 아니면 local peak contract가 깨진다.
        if tuple(labels.shape) != expected_voxel or tuple(instance.shape) != expected_voxel:
            raise ValueError(
                f"V259 target shape mismatch: labels={tuple(labels.shape)} "
                f"instance={tuple(instance.shape)} logits={tuple(pred_logits.shape)}"
            )
        if tuple(seed_center.shape) != expected_voxel or tuple(seed_body.shape) != expected_voxel:
            raise ValueError(
                f"V259 seed shape mismatch: seed_center={tuple(seed_center.shape)} "
                f"seed_body={tuple(seed_body.shape)} logits={tuple(pred_logits.shape)}"
            )
        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        seed_logit = pred_logits[:, 3].float()
        total = total + self.seed_peak_body_bce_lambda * self._seed_peak_body_bce_loss(seed_logit, seed_body)
        total = total + self.seed_peak_center_floor_lambda * self._seed_peak_body_floor_loss(
            seed_logit,
            seed_center,
            seed_body,
        )
        total = total + self.seed_peak_body_false_tail_lambda * self._seed_peak_body_false_tail_loss(seed_logit, seed_body)
        total = total + self.seed_peak_body_mass_cap_lambda * self._seed_peak_body_mass_cap_loss(
            seed_logit,
            seed_body,
            support_target,
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V259 mutex13 seed-peak-body support-leak loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_v259_seed_body_peak]
        # Expected: compact seed body high / outside low should beat missing body
        # or broad outside logits and produce finite gradients.
        return total


class BoundaryFragmentV3Core025StrongPeakMutex13FractureSeedCalibratedSupportLeakGuardSeedHealedHeadLoss(
    BoundaryFragmentV3Core025StrongPeakMutex13SeedPeakBodySupportLeakGuardSeedHealedHeadLoss
):
    """V260 loss: decoder-visible fracture barrier and per-fragment seed peaks.

    [AUDIT][Risk:High][Scope:v260_fulltrain_gate]
    V259 reached Dice/Local Dice 0.90+ but failed Task1 Instance Recall, Merge,
    Fracture Dice, and Topology because channel2 fracture and channel3 seed never
    crossed the fixed 0.50 decoder threshold. V260 keeps the 31-channel mutex
    output contract and changes only the decoder-facing calibration terms.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        seed_peak_body_bce_lambda: float = 1.80,
        seed_peak_center_floor_lambda: float = 1.20,
        seed_peak_body_false_tail_lambda: float = 1.40,
        seed_peak_body_mass_cap_lambda: float = 0.55,
        seed_fragment_presence_lambda: float = 2.25,
        seed_fragment_rank_lambda: float = 1.25,
        fracture_barrier_bce_lambda: float = 1.40,
        fracture_barrier_floor_lambda: float = 1.10,
        fracture_barrier_false_tail_lambda: float = 1.20,
        fracture_barrier_margin_lambda: float = 0.80,
        fracture_floor_probability: float = 0.70,
        fracture_false_ceiling_probability: float = 0.20,
        fracture_negative_ratio: int = 96,
        seed_fragment_floor_probability: float = 0.80,
        seed_fragment_rank_margin_logit: float = 1.50,
        seed_fragment_negative_fraction: float = 0.02,
        **kwargs,
    ):
        super().__init__(
            base_loss,
            seed_peak_body_bce_lambda=seed_peak_body_bce_lambda,
            seed_peak_center_floor_lambda=seed_peak_center_floor_lambda,
            seed_peak_body_false_tail_lambda=seed_peak_body_false_tail_lambda,
            seed_peak_body_mass_cap_lambda=seed_peak_body_mass_cap_lambda,
            **kwargs,
        )
        self.seed_fragment_presence_lambda = float(seed_fragment_presence_lambda)
        self.seed_fragment_rank_lambda = float(seed_fragment_rank_lambda)
        self.fracture_barrier_bce_lambda = float(fracture_barrier_bce_lambda)
        self.fracture_barrier_floor_lambda = float(fracture_barrier_floor_lambda)
        self.fracture_barrier_false_tail_lambda = float(fracture_barrier_false_tail_lambda)
        self.fracture_barrier_margin_lambda = float(fracture_barrier_margin_lambda)
        self.fracture_floor_probability = float(fracture_floor_probability)
        self.fracture_false_ceiling_probability = float(fracture_false_ceiling_probability)
        self.fracture_negative_ratio = int(fracture_negative_ratio)
        self.seed_fragment_floor_probability = float(seed_fragment_floor_probability)
        self.seed_fragment_rank_margin_logit = float(seed_fragment_rank_margin_logit)
        self.seed_fragment_negative_fraction = float(seed_fragment_negative_fraction)
        for name in (
            "seed_fragment_presence_lambda",
            "seed_fragment_rank_lambda",
            "fracture_barrier_bce_lambda",
            "fracture_barrier_floor_lambda",
            "fracture_barrier_false_tail_lambda",
            "fracture_barrier_margin_lambda",
            "seed_fragment_rank_margin_logit",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.5 < self.fracture_floor_probability < 1.0:
            raise ValueError(f"fracture_floor_probability must be in (0.5,1), got {self.fracture_floor_probability}")
        if not 0.0 < self.fracture_false_ceiling_probability < 0.5:
            raise ValueError(
                "fracture_false_ceiling_probability must be in (0,0.5), "
                f"got {self.fracture_false_ceiling_probability}"
            )
        if self.fracture_negative_ratio < 1:
            raise ValueError(f"fracture_negative_ratio must be >= 1, got {self.fracture_negative_ratio}")
        if not 0.5 < self.seed_fragment_floor_probability < 1.0:
            raise ValueError(
                "seed_fragment_floor_probability must be in (0.5,1), "
                f"got {self.seed_fragment_floor_probability}"
            )
        if not 0.0 < self.seed_fragment_negative_fraction <= 1.0:
            raise ValueError(
                "seed_fragment_negative_fraction must be in (0,1], "
                f"got {self.seed_fragment_negative_fraction}"
            )

    def _fracture_barrier_bce_loss(
        self,
        fracture_logit: torch.Tensor,
        fracture_target: torch.Tensor,
        support_target: torch.Tensor,
    ) -> torch.Tensor:
        # [DATA][Risk:High][Scope:v260_fracture_positive_negative]
        # positive는 GT BFV3 label2 fracture surface다. hard negative는 같은 support
        # 안의 non-fracture voxel만 쓴다. background까지 negative로 섞으면 decoder가
        # 실제 split해야 하는 anatomy 내부 ridge를 배우지 못한다.
        if fracture_logit.shape != fracture_target.shape or fracture_logit.shape != support_target.shape:
            raise ValueError(
                "V260 fracture BCE shape mismatch: "
                f"logit={tuple(fracture_logit.shape)} target={tuple(fracture_target.shape)} "
                f"support={tuple(support_target.shape)}"
            )
        pos = fracture_target > 0.5
        neg = (support_target > 0.5) & ~pos
        losses: list[torch.Tensor] = []
        if pos.any():
            pos_logits = fracture_logit[pos].float()
            losses.append(F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="mean"))
        if neg.any():
            neg_logits = fracture_logit[neg].float()
            if pos.any():
                k = min(
                    int(neg_logits.numel()),
                    max(1, int(pos.sum().detach().cpu()) * int(self.fracture_negative_ratio)),
                )
            else:
                # [AUDIT][Risk:High][Scope:non_fractured_anatomy]
                # Sacrum/RightHip처럼 fracture가 없는 ROI도 split false positive를 만들면
                # Instance Precision/Split/Topology가 무너진다. positive가 없어도 support
                # 내부 상위 false tail은 deterministic top-k로 억제한다.
                k = min(int(neg_logits.numel()), 4096)
            hard_neg = torch.topk(neg_logits, k=k, largest=True).values
            losses.append(F.binary_cross_entropy_with_logits(hard_neg, torch.zeros_like(hard_neg), reduction="mean"))
        if not losses:
            return fracture_logit.new_zeros(())
        return torch.stack(losses).mean()

    def _fracture_barrier_floor_loss(
        self,
        fracture_logit: torch.Tensor,
        fracture_target: torch.Tensor,
    ) -> torch.Tensor:
        pos = fracture_target > 0.5
        if not pos.any():
            return fracture_logit.new_zeros(())
        floor = self._probability_to_logit(self.fracture_floor_probability)
        weak_pos = self._top_fraction(fracture_logit[pos].float(), self.weak_positive_fraction, largest=False)
        # [METRIC][Scope:task1_fracture_threshold_050]
        # fixed decoder는 channel2 sigmoid 0.50 이상을 watershed barrier energy로 본다.
        # V259 forensic에서 label2와 non-fracture가 모두 0.38~0.40에 묶였으므로, low
        # quantile positive가 0.50 위로 올라오는지 직접 guard한다.
        return F.relu(float(floor) - weak_pos).mean()

    def _fracture_barrier_false_tail_loss(
        self,
        fracture_logit: torch.Tensor,
        fracture_target: torch.Tensor,
        support_target: torch.Tensor,
    ) -> torch.Tensor:
        neg = (support_target > 0.5) & ~(fracture_target > 0.5)
        if not neg.any():
            return fracture_logit.new_zeros(())
        neg_logits = fracture_logit[neg].float()
        k = min(int(neg_logits.numel()), max(1, int(fracture_target.sum().detach().cpu()) * int(self.fracture_negative_ratio), 4096))
        hard_neg = torch.topk(neg_logits, k=k, largest=True).values
        ceiling = self._probability_to_logit(self.fracture_false_ceiling_probability)
        return F.relu(hard_neg - float(ceiling)).mean()

    def _fracture_barrier_margin_loss(
        self,
        fracture_logit: torch.Tensor,
        fracture_target: torch.Tensor,
        support_target: torch.Tensor,
    ) -> torch.Tensor:
        pos = fracture_target > 0.5
        neg = (support_target > 0.5) & ~pos
        if not pos.any() or not neg.any():
            return fracture_logit.new_zeros(())
        weak_pos = self._top_fraction(fracture_logit[pos].float(), self.weak_positive_fraction, largest=False).mean()
        hard_neg = self._top_fraction(fracture_logit[neg].float(), self.weak_positive_fraction, largest=True).mean()
        margin = self._probability_to_logit(self.fracture_floor_probability) - self._probability_to_logit(
            self.fracture_false_ceiling_probability
        )
        # [AUDIT][Risk:High][Scope:v260_fracture_rank]
        # fracture Dice proxy와 watershed split은 absolute 0.50뿐 아니라 label2가
        # support non-fracture보다 높다는 순위 계약이 필요하다.
        return F.relu(float(margin) - (weak_pos - hard_neg))

    def _seed_fragment_presence_loss(
        self,
        seed_logit: torch.Tensor,
        seed_body: torch.Tensor,
        instance: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        floor = self._probability_to_logit(self.seed_fragment_floor_probability)
        for batch_idx in range(int(seed_logit.shape[0])):
            ids = torch.unique(instance[batch_idx])
            ids = ids[(ids > 0) & (ids <= MAX_INSTANCE_ID)]
            for frag_id in ids.tolist():
                frag = instance[batch_idx] == int(frag_id)
                body = frag & (seed_body[batch_idx] > 0.5)
                candidate = body if body.any() else frag
                if not candidate.any():
                    continue
                peak = seed_logit[batch_idx][candidate].float().max()
                losses.append(F.relu(float(floor) - peak))
        if not losses:
            return seed_logit.new_zeros(())
        # [METRIC][Scope:task1_instance_recall]
        # Task1 Instance Recall은 fragment마다 decoder-visible marker가 하나 이상
        # 있어야 열린다. 평균 BCE는 작은 fragment peak를 묻을 수 있으므로 fragment별
        # max peak가 0.50 위로 올라오는지를 직접 감독한다.
        return torch.stack(losses).mean()

    def _seed_fragment_rank_loss(
        self,
        seed_logit: torch.Tensor,
        seed_body: torch.Tensor,
        instance: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for batch_idx in range(int(seed_logit.shape[0])):
            ids = torch.unique(instance[batch_idx])
            ids = ids[(ids > 0) & (ids <= MAX_INSTANCE_ID)]
            for frag_id in ids.tolist():
                frag = instance[batch_idx] == int(frag_id)
                body = frag & (seed_body[batch_idx] > 0.5)
                non_body = frag & ~body
                if not body.any() or not non_body.any():
                    continue
                peak = seed_logit[batch_idx][body].float().max()
                false = self._top_fraction(
                    seed_logit[batch_idx][non_body].float(),
                    self.seed_fragment_negative_fraction,
                    largest=True,
                ).mean()
                losses.append(F.relu(float(self.seed_fragment_rank_margin_logit) - (peak - false)))
        if not losses:
            return seed_logit.new_zeros(())
        # [AUDIT][Risk:High][Scope:false_seed_split]
        # support-distance decoder-only test는 Sacrum/RightHip을 과분할했다. V260은
        # fragment 내부 seed_body peak가 같은 fragment의 나머지 voxel보다 높다는
        # 순위 계약을 추가해 false local peak로 인한 Split/Precision 실패를 줄인다.
        return torch.stack(losses).mean()

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        total = super().forward(pred_logits, target)
        semantic_target = self._semantic_target(target)
        if torch.is_tensor(semantic_target):
            semantic_target = semantic_target.to(device=pred_logits.device)
        labels = self._labels(semantic_target).to(device=pred_logits.device, dtype=torch.long)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_body = self._seed_body_target(target).to(device=pred_logits.device, dtype=torch.float32)
        expected_voxel = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        # [QC][Invariant:shape_dtype_device]
        # V260 fracture/seed calibration is decoder-facing. logits, semantic labels,
        # instance IDs, and seed_body must share the same voxel grid or the marker
        # and barrier channels would be trained against different anatomy.
        if tuple(labels.shape) != expected_voxel or tuple(instance.shape) != expected_voxel or tuple(seed_body.shape) != expected_voxel:
            raise ValueError(
                f"V260 target shape mismatch: labels={tuple(labels.shape)} "
                f"instance={tuple(instance.shape)} seed_body={tuple(seed_body.shape)} "
                f"logits={tuple(pred_logits.shape)}"
            )
        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        fracture_target = (labels == 2).float()
        fracture_logit = pred_logits[:, 2].float()
        seed_logit = pred_logits[:, 3].float()
        total = total + self.fracture_barrier_bce_lambda * self._fracture_barrier_bce_loss(
            fracture_logit,
            fracture_target,
            support_target,
        )
        total = total + self.fracture_barrier_floor_lambda * self._fracture_barrier_floor_loss(
            fracture_logit,
            fracture_target,
        )
        total = total + self.fracture_barrier_false_tail_lambda * self._fracture_barrier_false_tail_loss(
            fracture_logit,
            fracture_target,
            support_target,
        )
        total = total + self.fracture_barrier_margin_lambda * self._fracture_barrier_margin_loss(
            fracture_logit,
            fracture_target,
            support_target,
        )
        total = total + self.seed_fragment_presence_lambda * self._seed_fragment_presence_loss(
            seed_logit,
            seed_body,
            instance,
        )
        total = total + self.seed_fragment_rank_lambda * self._seed_fragment_rank_loss(
            seed_logit,
            seed_body,
            instance,
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V260 fracture/seed calibrated mutex13 loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_v260_fracture_seed_calibration]
        # Expected: fracture/seed logits high on GT label2/seed_body and low on
        # support non-fracture/non-seed must beat the inverted setup with finite gradients.
        return total


class BoundaryFragmentV3Core025StrongPeakPairwiseSoftmaxMutex13SeedHealedHeadLoss(
    BoundaryFragmentV3Core025StrongPeakMutex13FractureSeedCalibratedSupportLeakGuardSeedHealedHeadLoss
):
    """V266 loss: pairwise-softmax graph partition contract.

    전제조건:
        pred_logits는 [B, 31, Z, Y, X]이다. 0..4는 support/exterior/fracture/
        seed_center/seed_body이고, 5..17은 join logits, 18..30은 cut logits이다.

    실패 조건:
        edge별 join/cut 상대 logit contract가 깨지거나, Task1 proxy fracture
        surface와 다른 target을 쓰면 Instance/Merge/Split/Topology 및 Fracture
        Dice gate를 동시에 만족할 수 없다.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        support_bce_lambda: float = 1.10,
        support_threshold_guard_lambda: float = 0.85,
        exterior_bce_lambda: float = 0.10,
        seed_body_bce_lambda: float = 0.20,
        fracture_barrier_bce_lambda: float = 1.80,
        fracture_barrier_floor_lambda: float = 1.40,
        fracture_barrier_false_tail_lambda: float = 1.60,
        fracture_barrier_margin_lambda: float = 1.10,
        pairwise_edge_bce_lambda: float = 3.00,
        pairwise_edge_margin_lambda: float = 1.50,
        pairwise_edge_threshold_lambda: float = 1.00,
        pairwise_margin_logit: float = 2.00,
        hard_negative_top_fraction: float = 1.00,
        weak_positive_fraction: float = 0.20,
        **kwargs,
    ):
        # [AUDIT][Risk:Blocker][Scope:v266_representation_contract]
        # V255-V260은 attraction/repulsion을 독립 sigmoid로 학습했고, learned gate에서
        # 둘 다 threshold 주변에 묶이거나 전부 join되는 실패가 반복됐다. V266은 같은
        # 31채널을 쓰되 decoder decision을 join_logit - cut_logit 하나로 고정한다.
        # 이는 새 scalar bracket이 아니라 graph edge의 상호배타적 class contract다.
        super().__init__(
            base_loss,
            support_bce_lambda=support_bce_lambda,
            support_threshold_guard_lambda=support_threshold_guard_lambda,
            exterior_bce_lambda=exterior_bce_lambda,
            seed_center_bce_lambda=0.0,
            seed_body_bce_lambda=seed_body_bce_lambda,
            attractive_bce_lambda=0.0,
            repulsive_bce_lambda=0.0,
            mutex_threshold_guard_lambda=0.0,
            mutex_consistency_lambda=0.0,
            leak_edge_bce_lambda=0.0,
            leak_edge_threshold_lambda=0.0,
            fracture_barrier_bce_lambda=fracture_barrier_bce_lambda,
            fracture_barrier_floor_lambda=fracture_barrier_floor_lambda,
            fracture_barrier_false_tail_lambda=fracture_barrier_false_tail_lambda,
            fracture_barrier_margin_lambda=fracture_barrier_margin_lambda,
            hard_negative_top_fraction=hard_negative_top_fraction,
            weak_positive_fraction=weak_positive_fraction,
            **kwargs,
        )
        self.pairwise_edge_bce_lambda = float(pairwise_edge_bce_lambda)
        self.pairwise_edge_margin_lambda = float(pairwise_edge_margin_lambda)
        self.pairwise_edge_threshold_lambda = float(pairwise_edge_threshold_lambda)
        self.pairwise_margin_logit = float(pairwise_margin_logit)
        for name in (
            "pairwise_edge_bce_lambda",
            "pairwise_edge_margin_lambda",
            "pairwise_edge_threshold_lambda",
            "pairwise_margin_logit",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

    @classmethod
    def _edge_mask_to_voxel_mask(cls, edge_mask: torch.Tensor) -> torch.Tensor:
        if edge_mask.ndim != 5 or int(edge_mask.shape[1]) != len(cls.AFFINITY13_OFFSETS_ZYX):
            raise ValueError(f"V266 edge mask expects [B,13,Z,Y,X], got {tuple(edge_mask.shape)}")
        out = torch.zeros(
            (int(edge_mask.shape[0]), *edge_mask.shape[2:]),
            dtype=torch.bool,
            device=edge_mask.device,
        )
        shape = tuple(int(v) for v in edge_mask.shape[2:])
        for idx, offset in enumerate(cls.AFFINITY13_OFFSETS_ZYX):
            src_sl_zyx, dst_sl_zyx = cls._offset_slices(shape, offset)
            edge_src = edge_mask[:, idx][(slice(None), *src_sl_zyx)]
            out[(slice(None), *src_sl_zyx)] |= edge_src
            out[(slice(None), *dst_sl_zyx)] |= edge_src
        # [DATA][Risk:High][Scope:task1_fracture_proxy]
        # Task1 Fracture Dice proxy는 same-anatomy instance contact surface를 본다.
        # semantic BFV3 label2만 쓰면 V265처럼 label2/non2가 분리되지 않아 fracture
        # axis가 0.90 근처로 갈 수 없다. contact edge 양끝을 voxel mask로 투영해
        # decoder와 metric이 같은 fracture/contact evidence를 보게 한다.
        return out

    def _pairwise_edge_bce_loss(
        self,
        edge_diff_logit: torch.Tensor,
        same_edge: torch.Tensor,
        contact_edge: torch.Tensor,
        selected_leak_edge: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        if same_edge.any():
            weak_same = self._top_fraction(edge_diff_logit[same_edge].float(), self.weak_positive_fraction, largest=False)
            losses.append(F.binary_cross_entropy_with_logits(weak_same, torch.ones_like(weak_same), reduction="mean"))
        negative_edge = contact_edge | selected_leak_edge
        if negative_edge.any():
            hard_neg = self._top_fraction(edge_diff_logit[negative_edge].float(), self.hard_negative_top_fraction, largest=True)
            losses.append(F.binary_cross_entropy_with_logits(hard_neg, torch.zeros_like(hard_neg), reduction="mean"))
        if not losses:
            return edge_diff_logit.new_zeros(())
        # [AUDIT][Risk:High][Scope:pairwise_class_balance]
        # same edge 수가 압도적으로 많아도 negative contact/leak edge가 평균에서 묻히면
        # decoder는 다시 all-join으로 붕괴한다. positive tail과 hard negative tail을
        # 동일 단위로 평균해 Merge/Split/Topology 실패축을 직접 본다.
        return torch.stack(losses).mean()

    def _pairwise_edge_margin_loss(
        self,
        edge_diff_logit: torch.Tensor,
        same_edge: torch.Tensor,
        contact_edge: torch.Tensor,
        selected_leak_edge: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        margin = float(self.pairwise_margin_logit)
        if same_edge.any():
            weak_same = self._top_fraction(edge_diff_logit[same_edge].float(), self.weak_positive_fraction, largest=False)
            losses.append(F.relu(margin - weak_same).mean())
        negative_edge = contact_edge | selected_leak_edge
        if negative_edge.any():
            hard_neg = self._top_fraction(edge_diff_logit[negative_edge].float(), self.hard_negative_top_fraction, largest=True)
            losses.append(F.relu(hard_neg + margin).mean())
        if not losses:
            return edge_diff_logit.new_zeros(())
        return torch.stack(losses).mean()

    def _pairwise_threshold_guard_loss(
        self,
        edge_diff_logit: torch.Tensor,
        same_edge: torch.Tensor,
        contact_edge: torch.Tensor,
        selected_leak_edge: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        if same_edge.any():
            weak_same = self._top_fraction(edge_diff_logit[same_edge].float(), self.weak_positive_fraction, largest=False)
            losses.append(F.relu(-weak_same).mean())
        negative_edge = contact_edge | selected_leak_edge
        if negative_edge.any():
            hard_neg = self._top_fraction(edge_diff_logit[negative_edge].float(), self.hard_negative_top_fraction, largest=True)
            losses.append(F.relu(hard_neg).mean())
        if not losses:
            return edge_diff_logit.new_zeros(())
        # [METRIC][Scope:task1_threshold_050]
        # V266 decoder의 edge threshold 0.50은 sigmoid(join-cut)>=0.50, 즉
        # join_logit-cut_logit>=0과 동치다. 이 guard는 threshold sweep이 아니라
        # train target과 fixed decoder contract를 직접 묶는 안전장치다.
        return torch.stack(losses).mean()

    @classmethod
    def _support_adjacent_edge_mask(cls, instance: torch.Tensor) -> torch.Tensor:
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        if instance.ndim != 4:
            raise ValueError(f"V266 support-adjacent edge mask expects [B,Z,Y,X], got {tuple(instance.shape)}")
        inst = instance.long()
        support = (inst > 0) & (inst <= MAX_INSTANCE_ID)
        shape = tuple(int(v) for v in inst.shape[1:])
        edge_shape = (int(inst.shape[0]), len(cls.AFFINITY13_OFFSETS_ZYX), *shape)
        edge_mask = torch.zeros(edge_shape, dtype=torch.bool, device=inst.device)
        for i, offset in enumerate(cls.AFFINITY13_OFFSETS_ZYX):
            src_sl_zyx, dst_sl_zyx = cls._offset_slices(shape, offset)
            a = support[(slice(None), *src_sl_zyx)]
            b = support[(slice(None), *dst_sl_zyx)]
            edge_mask[(slice(None), i, *src_sl_zyx)] = a | b
        # [AUDIT][Risk:High][Scope:v266_runtime_and_signal]
        # full-volume background/background leak edge 전체를 top-k로 보면 patch 하나에서
        # 수천만 edge가 생겨 short gate가 CPU/GPU top-k에 묶인다. Task1 decoder failure와
        # 직접 연결되는 것은 support 내부/경계 주변 false edge이므로 leak mining 후보를
        # support-adjacent edge로 제한한다. 원거리 background는 support BCE/guard가 담당한다.
        return edge_mask

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        self._validate_mutex13_logits(pred_logits)
        semantic_target = self._semantic_target(target)
        if torch.is_tensor(semantic_target):
            semantic_target = semantic_target.to(device=pred_logits.device)
        labels = self._labels(semantic_target).to(device=pred_logits.device, dtype=torch.long)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_center = self._seed_center_target(target).to(device=pred_logits.device, dtype=torch.float32)
        seed_body = self._seed_body_target(target).to(device=pred_logits.device, dtype=torch.float32)
        expected_voxel = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        # [QC][Invariant:shape_dtype_device]
        # pairwise edge diff, contact-surface projection, seed marker supervision은 모두
        # 같은 voxel grid를 전제로 한다. 여기서 통과시키면 later Task1 proxy 수치가
        # 좋거나 나빠도 원인을 신뢰할 수 없다.
        if tuple(labels.shape) != expected_voxel:
            raise ValueError(f"V266 label shape mismatch: labels={tuple(labels.shape)} logits={tuple(pred_logits.shape)}")
        if tuple(instance.shape) != expected_voxel:
            raise ValueError(f"V266 instance shape mismatch: instance={tuple(instance.shape)} labels={tuple(labels.shape)}")
        if tuple(seed_center.shape) != expected_voxel or tuple(seed_body.shape) != expected_voxel:
            raise ValueError(
                f"V266 seed shape mismatch: seed_center={tuple(seed_center.shape)} "
                f"seed_body={tuple(seed_body.shape)} logits={tuple(pred_logits.shape)}"
            )

        same_edge, contact_edge, leak_edge, edge_universe = self._same_contact_leak_edges(instance)
        join_logit = pred_logits[:, self.ATTRACTIVE_START:self.REPULSIVE_START].float()
        cut_logit = pred_logits[:, self.REPULSIVE_START:].float()
        edge_diff_logit = join_logit - cut_logit
        reference_edges = int((same_edge | contact_edge).sum().detach().cpu())
        support_adjacent_edge = self._support_adjacent_edge_mask(instance)
        selected_leak_edge = self._select_leak_hard_negatives(
            join_logit,
            cut_logit,
            leak_edge & edge_universe & support_adjacent_edge,
            reference_count=reference_edges,
        )
        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        exterior_target = (labels == 1).float()
        fracture_target = self._edge_mask_to_voxel_mask(contact_edge).float()

        base_logits = pred_logits[:, :5].float()
        total = pred_logits.new_zeros(())
        total = total + self.support_bce_lambda * self._balanced_binary_bce(base_logits[:, 0], support_target, name="support")
        total = total + self.support_threshold_guard_lambda * self._support_threshold_guard_loss(base_logits[:, 0], support_target)
        total = total + self.exterior_bce_lambda * self._balanced_binary_bce(base_logits[:, 1], exterior_target, name="exterior")
        total = total + self.fracture_barrier_bce_lambda * self._fracture_barrier_bce_loss(
            base_logits[:, 2],
            fracture_target,
            support_target,
        )
        total = total + self.fracture_barrier_floor_lambda * self._fracture_barrier_floor_loss(
            base_logits[:, 2],
            fracture_target,
        )
        total = total + self.fracture_barrier_false_tail_lambda * self._fracture_barrier_false_tail_loss(
            base_logits[:, 2],
            fracture_target,
            support_target,
        )
        total = total + self.fracture_barrier_margin_lambda * self._fracture_barrier_margin_loss(
            base_logits[:, 2],
            fracture_target,
            support_target,
        )
        total = total + self.seed_body_bce_lambda * self._balanced_binary_bce(base_logits[:, 4], seed_body, name="seed_body")
        total = total + self.seed_fragment_presence_lambda * self._seed_fragment_presence_loss(
            base_logits[:, 3],
            seed_body,
            instance,
        )
        total = total + self.seed_fragment_rank_lambda * self._seed_fragment_rank_loss(
            base_logits[:, 3],
            seed_body,
            instance,
        )
        total = total + self.pairwise_edge_bce_lambda * self._pairwise_edge_bce_loss(
            edge_diff_logit,
            same_edge,
            contact_edge,
            selected_leak_edge,
        )
        total = total + self.pairwise_edge_margin_lambda * self._pairwise_edge_margin_loss(
            edge_diff_logit,
            same_edge,
            contact_edge,
            selected_leak_edge,
        )
        total = total + self.pairwise_edge_threshold_lambda * self._pairwise_threshold_guard_loss(
            edge_diff_logit,
            same_edge,
            contact_edge,
            selected_leak_edge,
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V266 pairwise-softmax mutex13 loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Pass][Test:synthetic_v266_pairwise_softmax]
        # Evidence: 2026-05-28 synthetic [1,31,8,8,8]에서 target-aligned logits
        # loss 0.0938 < inverted logits loss 75.3957, finite grad True.
        # fulltrain 전 hard3 Task1 proxy와 visual audit는 별도 필수다.
        return total


class BoundaryFragmentV3Core025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedHeadLoss(
    BoundaryFragmentV3Core025StrongPeakPairwiseSoftmaxMutex13SeedHealedHeadLoss
):
    """V268 loss: contact head 없이 instance graph partition을 직접 학습한다.

    전제조건:
        pred_logits는 [B, 31, Z, Y, X]이다. channel 0은 support, 3/4는
        seed score/body, 5..17은 join logits, 18..30은 cut logits이다.
        channel 1/2는 decoder에서 사용하지 않는 inert compatibility row다.

    실패 조건:
        shape/device가 맞지 않거나, edge target이 NaN/Inf를 만들거나, support/seed/
        pairwise graph loss가 finite scalar를 만들지 못하면 즉시 중단한다.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        support_bce_lambda: float = 1.10,
        support_threshold_guard_lambda: float = 0.85,
        seed_body_bce_lambda: float = 0.45,
        pairwise_edge_bce_lambda: float = 3.50,
        pairwise_edge_margin_lambda: float = 1.75,
        pairwise_edge_threshold_lambda: float = 1.20,
        pairwise_margin_logit: float = 2.00,
        hard_negative_top_fraction: float = 1.00,
        weak_positive_fraction: float = 0.20,
        **kwargs,
    ):
        # [AUDIT][Risk:Blocker][Scope:v268_contact_rejection]
        # V267 forensic에서 support Dice는 0.92 수준이지만 GT center recall은 0/26,
        # offset cosine은 음수였다. 따라서 다음 변경 변수는 contact/fracture head가 아니라
        # decoder가 직접 읽는 same-instance graph edge다. channel2 fracture/contact BCE는
        # 제거하고, Task1 Fracture Dice proxy는 최종 instance boundary에서만 평가한다.
        super().__init__(
            base_loss,
            support_bce_lambda=support_bce_lambda,
            support_threshold_guard_lambda=support_threshold_guard_lambda,
            exterior_bce_lambda=0.0,
            seed_body_bce_lambda=seed_body_bce_lambda,
            fracture_barrier_bce_lambda=0.0,
            fracture_barrier_floor_lambda=0.0,
            fracture_barrier_false_tail_lambda=0.0,
            fracture_barrier_margin_lambda=0.0,
            pairwise_edge_bce_lambda=pairwise_edge_bce_lambda,
            pairwise_edge_margin_lambda=pairwise_edge_margin_lambda,
            pairwise_edge_threshold_lambda=pairwise_edge_threshold_lambda,
            pairwise_margin_logit=pairwise_margin_logit,
            hard_negative_top_fraction=hard_negative_top_fraction,
            weak_positive_fraction=weak_positive_fraction,
            **kwargs,
        )

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        self._validate_mutex13_logits(pred_logits)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_body = self._seed_body_target(target).to(device=pred_logits.device, dtype=torch.float32)
        expected_voxel = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        # [QC][Invariant:shape_dtype_device]
        # V268은 semantic label2를 쓰지 않는다. 대신 instance ID와 seed_body가 같은 grid에
        # 있어야 support, seed, join/cut edge가 동일한 Task1 anatomy ROI를 가리킨다.
        if tuple(instance.shape) != expected_voxel:
            raise ValueError(
                f"V268 instance shape mismatch: instance={tuple(instance.shape)} "
                f"logits={tuple(pred_logits.shape)}"
            )
        if tuple(seed_body.shape) != expected_voxel:
            raise ValueError(
                f"V268 seed_body shape mismatch: seed_body={tuple(seed_body.shape)} "
                f"logits={tuple(pred_logits.shape)}"
            )
        if instance.device != pred_logits.device or seed_body.device != pred_logits.device:
            raise ValueError(
                f"V268 target device mismatch: logits={pred_logits.device} "
                f"instance={instance.device} seed_body={seed_body.device}"
            )

        same_edge, different_fragment_edge, leak_edge, edge_universe = self._same_contact_leak_edges(instance)
        expected_edge = tuple(int(v) for v in pred_logits.shape[0:1] + (13,) + pred_logits.shape[2:])
        if tuple(same_edge.shape) != expected_edge or tuple(different_fragment_edge.shape) != expected_edge:
            raise ValueError(
                f"V268 edge mask shape mismatch: expected={expected_edge}, "
                f"same={tuple(same_edge.shape)} different={tuple(different_fragment_edge.shape)}"
            )

        join_logit = pred_logits[:, self.ATTRACTIVE_START:self.REPULSIVE_START].float()
        cut_logit = pred_logits[:, self.REPULSIVE_START:].float()
        edge_diff_logit = join_logit - cut_logit
        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        support_adjacent_edge = self._support_adjacent_edge_mask(instance)
        reference_edges = int((same_edge | different_fragment_edge).sum().detach().cpu())
        selected_leak_edge = self._select_leak_hard_negatives(
            join_logit,
            cut_logit,
            leak_edge & edge_universe & support_adjacent_edge,
            reference_count=reference_edges,
        )

        # [DATA][Risk:High][Scope:v268_edge_targets]
        # positive는 같은 anatomy/같은 fragment edge, hard negative는 같은 anatomy/다른
        # fragment edge와 support-adjacent leak edge다. 여기에는 BFV3 label2 contact
        # probability를 쓰지 않으므로 contact-head threshold 반복으로 되돌아가지 않는다.
        total = pred_logits.new_zeros(())
        total = total + self.support_bce_lambda * self._balanced_binary_bce(
            pred_logits[:, 0].float(),
            support_target,
            name="v268_support",
        )
        total = total + self.support_threshold_guard_lambda * self._support_threshold_guard_loss(
            pred_logits[:, 0].float(),
            support_target,
        )
        total = total + self.seed_body_bce_lambda * self._balanced_binary_bce(
            pred_logits[:, 4].float(),
            seed_body,
            name="v268_seed_body",
        )
        total = total + self.seed_fragment_presence_lambda * self._seed_fragment_presence_loss(
            pred_logits[:, 3].float(),
            seed_body,
            instance,
        )
        total = total + self.seed_fragment_rank_lambda * self._seed_fragment_rank_loss(
            pred_logits[:, 3].float(),
            seed_body,
            instance,
        )
        total = total + self.pairwise_edge_bce_lambda * self._pairwise_edge_bce_loss(
            edge_diff_logit,
            same_edge,
            different_fragment_edge,
            selected_leak_edge,
        )
        total = total + self.pairwise_edge_margin_lambda * self._pairwise_edge_margin_loss(
            edge_diff_logit,
            same_edge,
            different_fragment_edge,
            selected_leak_edge,
        )
        total = total + self.pairwise_edge_threshold_lambda * self._pairwise_threshold_guard_loss(
            edge_diff_logit,
            same_edge,
            different_fragment_edge,
            selected_leak_edge,
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V268 no-contact pairwise-softmax mutex13 loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_v268_no_contact_pairwise]
        # Expected: support/seed/same-edge join 및 different-fragment cut이 target과 맞는
        # synthetic logits는 inverted join/cut logits보다 loss가 낮고 backward grad가 finite해야 한다.
        return total


class BoundaryFragmentV3Core025StrongPeakNoContactPairwiseSoftmaxSeedHealedV273HeadLoss(
    BoundaryFragmentV3Core025StrongPeakNoContactPairwiseSoftmaxMutex13SeedHealedHeadLoss
):
    """V273 loss: contact/fracture/exterior rows removed, boundary-exact graph retained.

    전제조건:
        pred_logits는 `[B, 28, Z, Y, X]`이다. channel 0은 support, channel 1은
        seed_body, channel 2..14는 join13, channel 15..27은 cut13이다.

    실패 조건:
        instance/seed target shape/device가 logits와 다르거나 loss가 NaN/Inf가 되면
        즉시 중단한다.
    """

    OUTPUT_CHANNELS = 28
    ATTRACTIVE_START = 2
    REPULSIVE_START = 15

    @classmethod
    def _validate_v273_logits(cls, pred_logits: torch.Tensor) -> None:
        # [QC][Invariant:channel_order]
        # V273은 contact/fracture/exterior compatibility row를 삭제한다. 31채널 V268
        # checkpoint를 여기서 허용하면 "contact 제거" 실험이 아니라 다른 contract 평가가 된다.
        if pred_logits.ndim != 5 or int(pred_logits.shape[1]) != int(cls.OUTPUT_CHANNELS):
            raise ValueError(
                f"BFV3 V273 no-contact pairwise loss expects logits [B, {cls.OUTPUT_CHANNELS}, Z, Y, X], "
                f"got {tuple(pred_logits.shape)}"
            )

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        self._validate_v273_logits(pred_logits)
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        seed_body = self._seed_body_target(target).to(device=pred_logits.device, dtype=torch.float32)
        expected_voxel = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        # [QC][Invariant:shape_dtype_device]
        # support/seed/join/cut은 같은 ROI grid에서만 의미가 있다. 이 검증이 없으면
        # pairwise edge target이 다른 crop의 boundary를 학습해도 loss scalar는 계산될 수 있다.
        if tuple(instance.shape) != expected_voxel:
            raise ValueError(
                f"V273 instance shape mismatch: instance={tuple(instance.shape)} "
                f"logits={tuple(pred_logits.shape)}"
            )
        if tuple(seed_body.shape) != expected_voxel:
            raise ValueError(
                f"V273 seed_body shape mismatch: seed_body={tuple(seed_body.shape)} "
                f"logits={tuple(pred_logits.shape)}"
            )
        if instance.device != pred_logits.device or seed_body.device != pred_logits.device:
            raise ValueError(
                f"V273 target device mismatch: logits={pred_logits.device} "
                f"instance={instance.device} seed_body={seed_body.device}"
            )

        same_edge, different_fragment_edge, leak_edge, edge_universe = self._same_contact_leak_edges(instance)
        expected_edge = tuple(int(v) for v in pred_logits.shape[0:1] + (len(self.AFFINITY13_OFFSETS_ZYX),) + pred_logits.shape[2:])
        if tuple(same_edge.shape) != expected_edge or tuple(different_fragment_edge.shape) != expected_edge:
            raise ValueError(
                f"V273 edge mask shape mismatch: expected={expected_edge}, "
                f"same={tuple(same_edge.shape)} different={tuple(different_fragment_edge.shape)}"
            )

        join_logit = pred_logits[:, self.ATTRACTIVE_START:self.REPULSIVE_START].float()
        cut_logit = pred_logits[:, self.REPULSIVE_START:].float()
        edge_diff_logit = join_logit - cut_logit
        support_target = ((instance > 0) & (instance <= MAX_INSTANCE_ID)).float()
        support_adjacent_edge = self._support_adjacent_edge_mask(instance)
        reference_edges = int((same_edge | different_fragment_edge).sum().detach().cpu())
        selected_leak_edge = self._select_leak_hard_negatives(
            join_logit,
            cut_logit,
            leak_edge & edge_universe & support_adjacent_edge,
            reference_count=reference_edges,
        )

        # [DATA][Risk:High][Scope:v273_no_contact_boundary_exact]
        # positive edge는 같은 GT fragment, hard negative edge는 같은 anatomy 안의 다른
        # fragment 및 support-adjacent leak이다. BFV3 label2/contact surface는 target에
        # 들어오지 않는다. Task1 Fracture Dice는 최종 instance boundary에서만 평가한다.
        total = pred_logits.new_zeros(())
        total = total + self.support_bce_lambda * self._balanced_binary_bce(
            pred_logits[:, 0].float(),
            support_target,
            name="v273_support",
        )
        total = total + self.support_threshold_guard_lambda * self._support_threshold_guard_loss(
            pred_logits[:, 0].float(),
            support_target,
        )
        total = total + self.seed_body_bce_lambda * self._balanced_binary_bce(
            pred_logits[:, 1].float(),
            seed_body,
            name="v273_seed_body",
        )
        total = total + self.seed_fragment_presence_lambda * self._seed_fragment_presence_loss(
            pred_logits[:, 1].float(),
            seed_body,
            instance,
        )
        total = total + self.seed_fragment_rank_lambda * self._seed_fragment_rank_loss(
            pred_logits[:, 1].float(),
            seed_body,
            instance,
        )
        total = total + self.pairwise_edge_bce_lambda * self._pairwise_edge_bce_loss(
            edge_diff_logit,
            same_edge,
            different_fragment_edge,
            selected_leak_edge,
        )
        total = total + self.pairwise_edge_margin_lambda * self._pairwise_edge_margin_loss(
            edge_diff_logit,
            same_edge,
            different_fragment_edge,
            selected_leak_edge,
        )
        total = total + self.pairwise_edge_threshold_lambda * self._pairwise_threshold_guard_loss(
            edge_diff_logit,
            same_edge,
            different_fragment_edge,
            selected_leak_edge,
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V273 no-contact pairwise-softmax loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [METRIC][Scope:task1_primary_gate]
        # V273의 fixed decoder threshold는 sigmoid(join-cut)>=0.50이다. fulltrain
        # 판단은 Dice/Fracture Dice/Local Dice/HD95/ASSD/Instance/Merge/Split/Topology와
        # visual audit만 사용한다.
        # [QA][Status:Not run][Test:synthetic_v273_no_contact_pairwise]
        # Expected: target-aligned support/seed/join/cut logits가 inverted logits보다 낮은
        # finite loss와 finite gradient를 만들어야 한다.
        return total


class BoundaryFragmentV3Core025StrongPeakNoContactSeparatorGapV277HeadLoss(nn.Module):
    """V277 loss: old contact head 없이 support와 separator-gap만 학습한다.

    전제조건:
        pred_logits는 `[B, 2, Z, Y, X]`이다. channel 0은 support foreground,
        channel 1은 connected component를 끊는 separator gap이다.

    실패 조건:
        target에 instance가 없거나 shape/device/finite 조건이 깨지면 즉시 중단한다.
    """

    OUTPUT_CHANNELS = 2
    INSTANCE_ID_MAX = MAX_INSTANCE_ID

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        *,
        support_bce_lambda: float = 0.65,
        support_dice_lambda: float = 0.35,
        separator_bce_lambda: float = 1.50,
        separator_dice_lambda: float = 0.75,
        support_negative_ratio: int = 2,
        separator_negative_ratio: int = 32,
        empty_separator_negative_topk: int = 2048,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.base = base_loss
        self.support_bce_lambda = float(support_bce_lambda)
        self.support_dice_lambda = float(support_dice_lambda)
        self.separator_bce_lambda = float(separator_bce_lambda)
        self.separator_dice_lambda = float(separator_dice_lambda)
        self.support_negative_ratio = int(support_negative_ratio)
        self.separator_negative_ratio = int(separator_negative_ratio)
        self.empty_separator_negative_topk = int(empty_separator_negative_topk)
        self.smooth = float(smooth)
        for name in ("support_bce_lambda", "support_dice_lambda", "separator_bce_lambda", "separator_dice_lambda", "smooth"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("support_negative_ratio", "separator_negative_ratio", "empty_separator_negative_topk"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")

    @staticmethod
    def _instance_target(target) -> torch.Tensor:
        if not isinstance(target, dict) or "instance" not in target:
            raise KeyError("V277 separator-gap loss requires target['instance']")
        instance = target["instance"]
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        return instance.long()

    @classmethod
    def _validate_logits_and_instance(cls, pred_logits: torch.Tensor, instance: torch.Tensor) -> None:
        # [QC][Invariant:shape_dtype_device]
        # V277은 `[B,2,Z,Y,X]` logits와 `[B,Z,Y,X]` instance target이 같은 device에
        # 있을 때만 decoder-facing support/separator supervision으로 해석된다.
        if pred_logits.ndim != 5 or int(pred_logits.shape[1]) != int(cls.OUTPUT_CHANNELS):
            raise ValueError(
                f"BFV3 V277 separator-gap loss expects logits [B, {cls.OUTPUT_CHANNELS}, Z, Y, X], "
                f"got {tuple(pred_logits.shape)}"
            )
        expected = tuple(int(v) for v in pred_logits.shape[0:1] + pred_logits.shape[2:])
        if tuple(instance.shape) != expected:
            raise ValueError(
                f"V277 instance shape mismatch: instance={tuple(instance.shape)} logits={tuple(pred_logits.shape)}"
            )
        if instance.device != pred_logits.device:
            raise ValueError(f"V277 instance device mismatch: logits={pred_logits.device} instance={instance.device}")
        if not torch.isfinite(pred_logits).all():
            raise ValueError("V277 logits contain NaN/Inf")
        invalid = (instance < 0) | (instance > int(cls.INSTANCE_ID_MAX))
        if invalid.any():
            bad_min = int(instance.min().detach().cpu())
            bad_max = int(instance.max().detach().cpu())
            raise ValueError(f"V277 instance IDs must be in [0, {cls.INSTANCE_ID_MAX}], got min={bad_min} max={bad_max}")

    @classmethod
    def _separator_gap_targets(cls, instance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return support and separator-gap voxel targets from instance IDs."""
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        if instance.ndim != 4:
            raise ValueError(f"V277 separator target requires instance [B,Z,Y,X], got {tuple(instance.shape)}")
        inst = instance.long()
        invalid = (inst < 0) | (inst > int(cls.INSTANCE_ID_MAX))
        if invalid.any():
            bad_min = int(inst.min().detach().cpu())
            bad_max = int(inst.max().detach().cpu())
            raise ValueError(f"V277 instance IDs must be in [0, {cls.INSTANCE_ID_MAX}], got min={bad_min} max={bad_max}")
        support = (inst > 0) & (inst <= int(cls.INSTANCE_ID_MAX))
        separator = torch.zeros_like(support, dtype=torch.bool)
        shape = tuple(int(v) for v in inst.shape[1:])
        for offset in BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss.AFFINITY13_OFFSETS_ZYX:
            src_sl_zyx, dst_sl_zyx = BoundaryFragmentV3Core025StrongPeakAffinity13SeedHealedHeadLoss._offset_slices(shape, offset)
            src_sl = (slice(None), *src_sl_zyx)
            dst_sl = (slice(None), *dst_sl_zyx)
            a = inst[src_sl]
            b = inst[dst_sl]
            a_fg = (a > 0) & (a <= int(cls.INSTANCE_ID_MAX))
            b_fg = (b > 0) & (b <= int(cls.INSTANCE_ID_MAX))
            same_anatomy = a_fg & b_fg & (((a - 1) // 50) == ((b - 1) // 50))
            different_fragment = same_anatomy & (a != b)
            if different_fragment.any():
                src_view = separator[src_sl]
                dst_view = separator[dst_sl]
                separator[src_sl] = src_view | different_fragment
                separator[dst_sl] = dst_view | different_fragment
        # [DATA][Risk:High][Scope:v277_separator_target]
        # separator는 old contact probability가 아니라 같은 anatomy 안에서 instance ID가
        # 바뀌는 13-neighbor voxel이다. 이 target은 CC decoder의 cut surface로만 쓰이며
        # contact precision/recall을 승격 지표로 사용하지 않는다.
        return support, separator & support

    def _balanced_topk_bce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        valid: torch.Tensor,
        negative_ratio: int,
        empty_negative_topk: int | None = None,
    ) -> torch.Tensor:
        pos = valid & target
        neg = valid & ~target
        terms: list[torch.Tensor] = []
        if pos.any():
            terms.append(F.binary_cross_entropy_with_logits(logits[pos].float(), torch.ones_like(logits[pos].float())))
        if neg.any():
            if pos.any():
                k = min(int(neg.sum().detach().cpu()), int(negative_ratio) * int(pos.sum().detach().cpu()))
            else:
                k = min(int(neg.sum().detach().cpu()), int(empty_negative_topk or negative_ratio))
            neg_logits = logits[neg].float()
            if k > 0 and int(neg_logits.numel()) > k:
                # [AUDIT][Risk:High][Scope:class_imbalance]
                # separator positive는 support 대비 매우 희소하다. 전체 negative 평균을 쓰면
                # false separator high-logit tail이 묻히므로 decoder가 보는 0.50 이상 영역만
                # deterministic top-k hard negative로 직접 누른다.
                hard_idx = torch.topk(torch.sigmoid(neg_logits), k=int(k), largest=True).indices
                neg_logits = neg_logits[hard_idx]
            terms.append(F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits)))
        if not terms:
            return logits.new_zeros(())
        return torch.stack(terms).mean()

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor, *, valid: torch.Tensor | None = None) -> torch.Tensor:
        prob = torch.sigmoid(logits.float())
        target_f = target.float()
        if valid is not None:
            valid_f = valid.float()
            prob = prob * valid_f
            target_f = target_f * valid_f
        dims = tuple(range(1, prob.ndim))
        inter = (prob * target_f).sum(dim=dims)
        denom = prob.sum(dim=dims) + target_f.sum(dim=dims)
        return (1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean()

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        self._validate_logits_and_instance(pred_logits, instance)
        support_target, separator_target = self._separator_gap_targets(instance)
        support_logits = pred_logits[:, 0].float()
        separator_logits = pred_logits[:, 1].float()
        valid_all = torch.ones_like(support_target, dtype=torch.bool, device=pred_logits.device)

        # [AUDIT][Risk:Blocker][Scope:v277_decoder_contract]
        # active path에서 contact/channel5/min(candidate,aux) contract를 폐기한다.
        # 학습 target은 decoder가 직접 threshold 0.50으로 읽는 support와 separator gap뿐이다.
        support_bce = self._balanced_topk_bce(
            support_logits,
            support_target,
            valid=valid_all,
            negative_ratio=self.support_negative_ratio,
            empty_negative_topk=self.empty_separator_negative_topk,
        )
        support_dice = self._dice_loss(support_logits, support_target)
        separator_bce = self._balanced_topk_bce(
            separator_logits,
            separator_target,
            valid=support_target,
            negative_ratio=self.separator_negative_ratio,
            empty_negative_topk=self.empty_separator_negative_topk,
        )
        separator_dice = self._dice_loss(separator_logits, separator_target, valid=support_target)
        total = (
            self.support_bce_lambda * support_bce
            + self.support_dice_lambda * support_dice
            + self.separator_bce_lambda * separator_bce
            + self.separator_dice_lambda * separator_dice
        )
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V277 no-contact separator-gap loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [METRIC][Scope:task1_primary_gate]
        # decoder threshold는 support/separator 모두 0.50이다. 승격 판단은
        # Dice/Fracture Dice/Local Dice/Instance/Merge/Split/Topology proxy와 visual audit이다.
        # [QA][Status:Not run][Test:synthetic_v277_separator_gap]
        # Expected: support high + separator target high 설정의 loss가 separator를 모두
        # 끈 bad logits보다 낮고, backward 후 gradient가 finite해야 한다.
        return total


class BoundaryFragmentV3Core025StrongPeakNoContactABBCV288HeadLoss(
    BoundaryFragmentV3Core025StrongPeakNoContactSeparatorGapV277HeadLoss
):
    """V288 loss: ABBC 4-class softmax [background, border, boundary, core].

    [AUDIT][Risk:Blocker][Scope:v288_abbc_representation]
    PENGWIN 2024 CT 1st place (MIC-DKFZ, IoU-F 0.9296) Adaptive Border Boundary
    Core representation. V287의 3-class [background, body, separator]는 body가
    seed 역할을 못해 같은 anatomy 안에서 fragment가 머지됐다. V288은 별도 core
    seed class와 fragment-size-adaptive boundary thickness로 watershed-based
    instance separation을 가능하게 한다.

    전제조건:
        pred_logits는 `[B, 4, Z, Y, X]`이다. channel 0 background, 1 border,
        2 boundary, 3 core.

    실패 조건:
        target에 instance가 없거나 shape/device/finite 조건이 깨지면 즉시 중단한다.
    """

    OUTPUT_CHANNELS = 4

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        *,
        class_ce_lambda: float = 1.0,
        class_dice_lambda: float = 1.0,
        boundary_dilate_vox: int = 2,
        core_erode_vox: int = 2,
        smooth: float = 1e-5,
    ):
        super().__init__(base_loss=None, smooth=smooth)
        self.base = base_loss
        self.class_ce_lambda = float(class_ce_lambda)
        self.class_dice_lambda = float(class_dice_lambda)
        self.boundary_dilate_vox = int(boundary_dilate_vox)
        self.core_erode_vox = int(core_erode_vox)
        for name in ("class_ce_lambda", "class_dice_lambda"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("boundary_dilate_vox", "core_erode_vox"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

    @classmethod
    def _abbc_class_target(
        cls,
        instance: torch.Tensor,
        *,
        boundary_dilate_vox: int,
        core_erode_vox: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ABBC class target `[B, Z, Y, X]` long and support `[B, Z, Y, X]` bool.

        [DATA][Risk:High][Scope:v288_abbc_target]
        torch-only path. boundary는 V277 13-neighbor between-fragment band를
        `boundary_dilate_vox` 회 3x3x3 max_pool로 dilate한 뒤 support로 clip한다.
        core는 support를 `core_erode_vox` 회 erode한 뒤 boundary와의 XOR로 정의한다.
        scipy 기반 fragment-size-adaptive thickness/forced-seed fallback은 oracle
        eval 단계에서만 사용한다. 학습 시에는 동일 의도의 단순 morphology만 사용해
        매 step GPU에서 빠르게 target을 만든다.
        """
        if instance.ndim == 5 and int(instance.shape[1]) == 1:
            instance = instance[:, 0]
        if instance.ndim != 4:
            raise ValueError(f"V288 ABBC target requires instance [B,Z,Y,X], got {tuple(instance.shape)}")
        support, raw_between = (
            BoundaryFragmentV3Core025StrongPeakNoContactSeparatorGapV277HeadLoss
            ._separator_gap_targets(instance)
        )
        non_support = ~support
        if int(boundary_dilate_vox) > 0 and bool(raw_between.any()):
            kernel = 2 * int(boundary_dilate_vox) + 1
            dilated = F.max_pool3d(
                raw_between.float().unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=int(boundary_dilate_vox),
            )[:, 0]
            boundary = (dilated > 0.5) & support
        else:
            boundary = raw_between & support
        if int(core_erode_vox) > 0:
            kernel = 2 * int(core_erode_vox) + 1
            non_support_dilated = F.max_pool3d(
                non_support.float().unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=int(core_erode_vox),
            )[:, 0]
            support_eroded = support & (non_support_dilated <= 0.5)
        else:
            support_eroded = support.clone()
        core = support_eroded & (~boundary)
        border = support & (~core) & (~boundary)
        class_target = torch.zeros_like(instance, dtype=torch.long)
        class_target[border] = 1
        class_target[boundary] = 2
        class_target[core] = 3
        return class_target, support

    def _present_class_ce(
        self,
        logits: torch.Tensor,
        class_target: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits.float(), class_target.long(), reduction="none")
        terms: list[torch.Tensor] = []
        for class_id in range(int(self.OUTPUT_CHANNELS)):
            mask = class_target == int(class_id)
            if mask.any():
                terms.append(ce[mask].mean())
        if not terms:
            return logits.sum() * 0.0
        # [DATA][Risk:High][Scope:v288_rare_class_balance]
        # core, boundary, border는 background에 비해 희소하다. voxel 평균 CE를 쓰면
        # background에 묻혀 seed/separator 학습이 V287/V277처럼 약해진다.
        return torch.stack(terms).mean()

    def _present_class_dice(
        self,
        logits: torch.Tensor,
        class_target: torch.Tensor,
    ) -> torch.Tensor:
        prob = torch.softmax(logits.float(), dim=1)
        terms: list[torch.Tensor] = []
        dims = tuple(range(1, prob[:, 0].ndim))
        for class_id in range(int(self.OUTPUT_CHANNELS)):
            target = (class_target == int(class_id)).float()
            if not bool(target.any()):
                continue
            pred = prob[:, int(class_id)]
            inter = (pred * target).sum(dim=dims)
            denom = pred.sum(dim=dims) + target.sum(dim=dims)
            terms.append((1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean())
        if not terms:
            return logits.sum() * 0.0
        return torch.stack(terms).mean()

    def forward(self, pred_logits: torch.Tensor, target) -> torch.Tensor:
        instance = self._instance_target(target).to(device=pred_logits.device, dtype=torch.long)
        # [QC][Invariant:shape_dtype_device]
        # V288은 4-class softmax contract다. V287의 3-channel logits를 받으면
        # core class 학습이 누락되어 decoder가 watershed seed를 못 가진다.
        self._validate_logits_and_instance(pred_logits, instance)
        class_target, _ = self._abbc_class_target(
            instance,
            boundary_dilate_vox=self.boundary_dilate_vox,
            core_erode_vox=self.core_erode_vox,
        )
        total = pred_logits.new_zeros(())
        total = total + self.class_ce_lambda * self._present_class_ce(pred_logits, class_target)
        total = total + self.class_dice_lambda * self._present_class_dice(pred_logits, class_target)
        if not torch.isfinite(total):
            raise ValueError(
                "BFV3 V288 no-contact ABBC loss produced non-finite loss: "
                f"{float(total.detach().cpu())}"
            )
        # [QA][Status:Not run][Test:synthetic_v288_abbc]
        # Expected: ABBC 4-class가 맞는 logits가 background-only collapse 또는
        # core-missing logits보다 낮은 finite loss와 finite gradient를 만든다.
        return total


class BoundaryFragmentV3Core025StrongPeakNoContactABBCV291HeadLoss(
    BoundaryFragmentV3Core025StrongPeakNoContactABBCV288HeadLoss
):
    """V291 loss: V288 ABBC with boundary class weight on CE + Dice.

    [AUDIT][Risk:Blocker][Scope:v291_boundary_class_weight]
    V288 50e patch boundary_precision/recall = 0.066/0.21 because the boundary
    class is ~5% of support voxels and equally weighted with background/border/
    core in the per-class mean. V291 multiplies the boundary class CE and Dice
    terms by `boundary_class_weight` (default 5x). No SDF, no EDT, no CPU
    bottleneck -- same throughput as V288 (~38 sec/epoch).

    전제조건:
        pred_logits는 `[B, 4, Z, Y, X]`이다. channel 0 background, 1 border,
        2 boundary, 3 core (V288와 동일).

    실패 조건:
        V288 shape/device/finite guard를 통과하지 못하면 즉시 중단한다.
    """

    OUTPUT_CHANNELS = 4

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        *,
        class_ce_lambda: float = 1.0,
        class_dice_lambda: float = 1.0,
        boundary_class_weight: float = 5.0,
        boundary_dilate_vox: int = 2,
        core_erode_vox: int = 2,
        smooth: float = 1e-5,
    ):
        super().__init__(
            base_loss=base_loss,
            class_ce_lambda=class_ce_lambda,
            class_dice_lambda=class_dice_lambda,
            boundary_dilate_vox=boundary_dilate_vox,
            core_erode_vox=core_erode_vox,
            smooth=smooth,
        )
        # [DATA][Risk:High][Scope:v291_boundary_weight_sweep]
        # Allow hyperparameter sweep without adding a new loss class: read
        # PENGWIN_BFV3_V291_BOUNDARY_CLASS_WEIGHT and override the default. Useful
        # for boundary weight ablation across multiple GPUs/training jobs.
        env_override = os.environ.get("PENGWIN_BFV3_V291_BOUNDARY_CLASS_WEIGHT", "").strip()
        if env_override:
            try:
                boundary_class_weight = float(env_override)
            except ValueError:
                pass
        self.boundary_class_weight = float(boundary_class_weight)
        if self.boundary_class_weight <= 0.0:
            raise ValueError(f"boundary_class_weight must be > 0, got {boundary_class_weight}")

    def _present_class_ce(
        self,
        logits: torch.Tensor,
        class_target: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits.float(), class_target.long(), reduction="none")
        terms: list[torch.Tensor] = []
        for class_id in range(int(self.OUTPUT_CHANNELS)):
            mask = class_target == int(class_id)
            if mask.any():
                term = ce[mask].mean()
                if int(class_id) == 2:  # boundary class
                    term = term * float(self.boundary_class_weight)
                terms.append(term)
        if not terms:
            return logits.sum() * 0.0
        return torch.stack(terms).mean()

    def _present_class_dice(
        self,
        logits: torch.Tensor,
        class_target: torch.Tensor,
    ) -> torch.Tensor:
        prob = torch.softmax(logits.float(), dim=1)
        terms: list[torch.Tensor] = []
        dims = tuple(range(1, prob[:, 0].ndim))
        for class_id in range(int(self.OUTPUT_CHANNELS)):
            target = (class_target == int(class_id)).float()
            if not bool(target.any()):
                continue
            pred = prob[:, int(class_id)]
            inter = (pred * target).sum(dim=dims)
            denom = pred.sum(dim=dims) + target.sum(dim=dims)
            term = (1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean()
            if int(class_id) == 2:
                term = term * float(self.boundary_class_weight)
            terms.append(term)
        if not terms:
            return logits.sum() * 0.0
        return torch.stack(terms).mean()


class BoundaryFragmentV3RidgeFitDeepSupervisionWrapper(nn.Module):
    """Use full ridgefit terms only on full-resolution deep supervision output."""

    def __init__(self, fullres_loss: nn.Module, aux_loss: nn.Module, weight_factors):
        super().__init__()
        self.fullres_loss = fullres_loss
        self.aux_loss = aux_loss
        self.weight_factors = tuple(float(x) for x in weight_factors)
        if not any(x != 0 for x in self.weight_factors):
            raise ValueError("At least one deep-supervision weight must be nonzero")

    def forward(self, *args):
        if not all(isinstance(i, (tuple, list)) for i in args):
            raise AssertionError(f"all args must be tuple/list, got {[type(i) for i in args]}")
        total = None
        for idx, inputs in enumerate(zip(*args)):
            weight = self.weight_factors[idx]
            if weight == 0:
                continue
            loss = self.fullres_loss(*inputs) if idx == 0 else self.aux_loss(*inputs)
            total = weight * loss if total is None else total + weight * loss
        return total


class BICMV5SparseLoss(nn.Module):
    """Sparse core/contact objective for Dataset537 BICM V5.

    V5 deliberately keeps the output as a normal 5-class nnU-Net semantic head:
    `0 background, 1 exterior_context, 2 shell, 3 core, 4 contact_surface`.
    The first case003 overfit proved that plain Dice+CE learns shell/support but
    collapses the tiny core/contact labels to zero. This wrapper keeps the base
    DC+CE term and adds only the terms needed to test that root cause.

    Raises:
        ValueError: if logits/labels do not match the 5-class V5 contract.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        fullres_sparse_terms: bool = True,
        support_dice_lambda: float = 0.25,
        core_tversky_lambda: float = 0.65,
        core_focal_lambda: float = 0.45,
        contact_tversky_lambda: float = 0.55,
        contact_focal_lambda: float = 0.35,
        sparse_margin_lambda: float = 0.20,
        false_sparse_topk_lambda: float = 0.10,
        topk_fraction: float = 0.02,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.base = base_loss
        self.fullres_sparse_terms = bool(fullres_sparse_terms)
        self.support_dice_lambda = float(support_dice_lambda)
        self.core_tversky_lambda = float(core_tversky_lambda)
        self.core_focal_lambda = float(core_focal_lambda)
        self.contact_tversky_lambda = float(contact_tversky_lambda)
        self.contact_focal_lambda = float(contact_focal_lambda)
        self.sparse_margin_lambda = float(sparse_margin_lambda)
        self.false_sparse_topk_lambda = float(false_sparse_topk_lambda)
        self.topk_fraction = float(topk_fraction)
        self.smooth = float(smooth)

    @staticmethod
    def _labels(target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 5 and target.shape[1] == 1:
            return target[:, 0].long()
        if target.ndim == 5 and target.shape[1] == 5:
            return target.argmax(dim=1)
        return target.long()

    def _check_contract(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        # [QC][Invariant:shape_label_contract]
        # V5 loss terms index classes 2/3/4 directly. A silent class-count
        # mismatch would train the wrong anatomical meaning and make IoU-F
        # evidence unusable, so fail before computing the loss.
        if logits.ndim != 5 or logits.shape[1] != 5:
            raise ValueError(f"BICMV5SparseLoss expects logits [B,5,D,H,W], got {tuple(logits.shape)}")
        if labels.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                "BICMV5SparseLoss target shape mismatch: "
                f"labels={tuple(labels.shape)} logits={tuple(logits.shape)}"
            )
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) > 4):
            raise ValueError(
                f"BICMV5SparseLoss labels must be in 0..4, got min={int(labels.min())} "
                f"max={int(labels.max())}"
            )

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))
        inter = (pred * target).sum(dim=dims)
        denom = pred.sum(dim=dims) + target.sum(dim=dims)
        return (1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean()

    def _tversky(self, pred: torch.Tensor, target: torch.Tensor,
                 fp_weight: float, fn_weight: float) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))
        tp = (pred * target).sum(dim=dims)
        fp = (pred * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - pred) * target).sum(dim=dims)
        return (1.0 - (tp + self.smooth) / (tp + fp_weight * fp + fn_weight * fn + self.smooth)).mean()

    def _focal(self, prob: torch.Tensor, target: torch.Tensor,
               pos_alpha: float, neg_alpha: float) -> torch.Tensor:
        prob = prob.clamp(self.smooth, 1.0 - self.smooth)
        pos = target > 0.5
        neg = ~pos
        total = prob.new_zeros(())
        if pos.any():
            total = total + (-pos_alpha * ((1.0 - prob[pos]) ** 2.0) * torch.log(prob[pos])).mean()
        if neg.any():
            total = total + (-neg_alpha * (prob[neg] ** 2.0) * torch.log(1.0 - prob[neg])).mean()
        return total

    def _sparse_margin(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # [METRIC][Risk:Major][Scope:core_contact_collapse]
        # IoU-F cannot be decoded without core markers. This margin applies
        # only on GT core/contact voxels and asks the correct sparse class to
        # beat the best competing logit; it does not alter the decoder.
        sparse = (labels == 3) | (labels == 4)
        if not sparse.any():
            return logits.new_zeros(())
        correct = logits.gather(1, labels.clamp(0, 4).unsqueeze(1)).squeeze(1)
        other_logits = logits.clone()
        other_logits.scatter_(1, labels.clamp(0, 4).unsqueeze(1), float("-inf"))
        best_other = other_logits.amax(dim=1)
        return torch.relu(1.0 + best_other[sparse] - correct[sparse]).mean()

    def _false_sparse_topk(self, sparse_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        non_sparse = (labels != 3) & (labels != 4)
        if not non_sparse.any():
            return sparse_prob.new_zeros(())
        vals = sparse_prob[:, 0][non_sparse]
        if vals.numel() == 0:
            return sparse_prob.new_zeros(())
        k = max(1, int(round(float(vals.numel()) * self.topk_fraction)))
        return torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        labels = self._labels(target).to(device=pred_logits.device)
        self._check_contract(pred_logits, labels)
        total = self.base(pred_logits, target)
        prob = torch.softmax(pred_logits, dim=1)
        support_target = ((labels == 2) | (labels == 3) | (labels == 4)).float().unsqueeze(1)
        support_prob = prob[:, 2:5].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        total = total + self.support_dice_lambda * self._dice_loss(support_prob, support_target)
        if not self.fullres_sparse_terms:
            return total

        core_target = (labels == 3).float().unsqueeze(1)
        contact_target = (labels == 4).float().unsqueeze(1)
        core_prob = prob[:, 3:4]
        contact_prob = prob[:, 4:5]
        sparse_prob = (core_prob + contact_prob).clamp(0.0, 1.0)

        total = total + self.core_tversky_lambda * self._tversky(
            core_prob, core_target, fp_weight=0.30, fn_weight=0.85
        )
        total = total + self.core_focal_lambda * self._focal(
            core_prob, core_target, pos_alpha=0.90, neg_alpha=0.10
        )
        if contact_target.any():
            total = total + self.contact_tversky_lambda * self._tversky(
                contact_prob, contact_target, fp_weight=0.45, fn_weight=0.75
            )
            total = total + self.contact_focal_lambda * self._focal(
                contact_prob, contact_target, pos_alpha=0.85, neg_alpha=0.15
            )
        total = total + self.sparse_margin_lambda * self._sparse_margin(pred_logits, labels)
        total = total + self.false_sparse_topk_lambda * self._false_sparse_topk(sparse_prob, labels)
        return total


class BICMV5SupportGeometryLoss(nn.Module):
    """Support-geometry objective for Dataset537 BICM V5.

    The V5.4 oracle-hybrid diagnostic showed that deterministic seed healing
    can remove core component explosion, but IoU-F remains poor when predicted
    support misses oracle core/contact geometry. This loss intentionally changes
    only the binary fragment-support objective while keeping the 5-class target,
    model head, input channel, sampling policy, and fixed decoder unchanged.

    Args:
        base_loss: Native nnU-Net DC+CE loss for the 5-class semantic target.
        fullres_geometry_terms: If false, only the coarse support Dice term is
            applied. This is used for low-resolution deep supervision where
            top-k false-positive and sparse coverage terms are not stable.

    Raises:
        ValueError: If logits/labels do not match the V5 5-class contract.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        fullres_geometry_terms: bool = True,
        support_dice_lambda: float = 0.85,
        support_tversky_lambda: float = 0.55,
        core_coverage_lambda: float = 0.30,
        contact_coverage_lambda: float = 0.20,
        support_topk_false_lambda: float = 0.35,
        topk_fraction: float = 0.02,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.base = base_loss
        self.fullres_geometry_terms = bool(fullres_geometry_terms)
        self.support_dice_lambda = float(support_dice_lambda)
        self.support_tversky_lambda = float(support_tversky_lambda)
        self.core_coverage_lambda = float(core_coverage_lambda)
        self.contact_coverage_lambda = float(contact_coverage_lambda)
        self.support_topk_false_lambda = float(support_topk_false_lambda)
        self.topk_fraction = float(topk_fraction)
        self.smooth = float(smooth)

    @staticmethod
    def _labels(target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 5 and target.shape[1] == 1:
            return target[:, 0].long()
        if target.ndim == 5 and target.shape[1] == 5:
            return target.argmax(dim=1)
        return target.long()

    def _check_contract(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        # [QC][Invariant:shape_label_contract]
        # This loss treats labels {2,3,4} as binary fragment support. If a
        # different dataset/head reaches this branch, full-volume IoU-F evidence
        # would be invalid, so fail before training silently optimizes nonsense.
        if logits.ndim != 5 or logits.shape[1] != 5:
            raise ValueError(f"BICMV5SupportGeometryLoss expects logits [B,5,D,H,W], got {tuple(logits.shape)}")
        if labels.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                "BICMV5SupportGeometryLoss target shape mismatch: "
                f"labels={tuple(labels.shape)} logits={tuple(logits.shape)}"
            )
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) > 4):
            raise ValueError(
                f"BICMV5SupportGeometryLoss labels must be in 0..4, got min={int(labels.min())} "
                f"max={int(labels.max())}"
            )

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))
        inter = (pred * target).sum(dim=dims)
        denom = pred.sum(dim=dims) + target.sum(dim=dims)
        return (1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean()

    def _tversky(self, pred: torch.Tensor, target: torch.Tensor,
                 fp_weight: float, fn_weight: float) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))
        tp = (pred * target).sum(dim=dims)
        fp = (pred * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - pred) * target).sum(dim=dims)
        return (1.0 - (tp + self.smooth) / (tp + fp_weight * fp + fn_weight * fn + self.smooth)).mean()

    def _positive_coverage(self, support_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [METRIC][Risk:Major][Scope:oracle_hybrid]
        # V5.4 measured how much target core/contact lies inside predicted
        # support. This term optimizes that exact invariant without requiring
        # the network to predict contact/core correctly in this experiment.
        vals = support_prob[:, 0][mask]
        if vals.numel() == 0:
            return support_prob.new_zeros(())
        return -torch.log(vals.clamp(self.smooth, 1.0 - self.smooth)).mean()

    def _support_topk_false(self, support_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        non_support = (labels != 2) & (labels != 3) & (labels != 4)
        if not non_support.any():
            return support_prob.new_zeros(())
        vals = support_prob[:, 0][non_support]
        if vals.numel() == 0:
            return support_prob.new_zeros(())
        k = max(1, int(round(float(vals.numel()) * self.topk_fraction)))
        return torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        labels = self._labels(target).to(device=pred_logits.device)
        self._check_contract(pred_logits, labels)
        total = self.base(pred_logits, target)
        prob = torch.softmax(pred_logits, dim=1)
        support_target = ((labels == 2) | (labels == 3) | (labels == 4)).float().unsqueeze(1)
        support_prob = prob[:, 2:5].sum(dim=1, keepdim=True).clamp(0.0, 1.0)

        # [QA][Gate:003_support_geometry]
        # Expected improvement is measured outside training by full-volume
        # support Dice/HD95 and target-core coverage, not patch pseudo Dice.
        total = total + self.support_dice_lambda * self._dice_loss(support_prob, support_target)
        if not self.fullres_geometry_terms:
            return total

        total = total + self.support_tversky_lambda * self._tversky(
            support_prob, support_target, fp_weight=0.55, fn_weight=0.65
        )
        total = total + self.core_coverage_lambda * self._positive_coverage(
            support_prob, labels == 3
        )
        if (labels == 4).any():
            total = total + self.contact_coverage_lambda * self._positive_coverage(
                support_prob, labels == 4
            )
        total = total + self.support_topk_false_lambda * self._support_topk_false(
            support_prob, labels
        )
        return total


class BICMV62SemanticBoundaryLoss(nn.Module):
    """V62 semantic-boundary loss for the oracle-passing V5 target.

    [AUDIT][Risk:High][Scope:methodology_pivot]
    V38-V61 showed that adding branches, scalar schedules, peak seeds, and
    contact/edge heads to the factorized BICM objective still cannot align
    support, contact, and seed topology in full volume. V62 deliberately
    returns to the simpler 5-class V5 semantic head and changes only the loss:
    boundary/support/contact terms are optimized inside the oracle-passing
    semantic representation. This is a diagnostic to decide whether the V5
    semantic path deserves more work before any 159/20-case/fold0/fulltrain.

    V5 class contract:
        0 background, 1 exterior_context, 2 shell, 3 core, 4 contact_surface.

    [QA][Gate:case003_before_expansion][Status:Required]
    The only acceptance evidence is full-volume `bicm-v5-eval` on case003.
    Patch pseudo Dice and nnU-Net `checkpoint_best` are health signals only.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        fullres_terms: bool = True,
        soft_ce_lambda: float = 0.12,
        support_dice_lambda: float = 0.45,
        support_boundary_lambda: float = 0.18,
        contact_tversky_lambda: float = 0.55,
        contact_focal_lambda: float = 0.28,
        contact_false_topk_lambda: float = 0.08,
        core_tversky_lambda: float = 0.40,
        core_focal_lambda: float = 0.22,
        sparse_margin_lambda: float = 0.18,
        topk_fraction: float = 0.01,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.base = base_loss
        self.fullres_terms = bool(fullres_terms)
        self.soft_ce_lambda = float(soft_ce_lambda)
        self.support_dice_lambda = float(support_dice_lambda)
        self.support_boundary_lambda = float(support_boundary_lambda)
        self.contact_tversky_lambda = float(contact_tversky_lambda)
        self.contact_focal_lambda = float(contact_focal_lambda)
        self.contact_false_topk_lambda = float(contact_false_topk_lambda)
        self.core_tversky_lambda = float(core_tversky_lambda)
        self.core_focal_lambda = float(core_focal_lambda)
        self.sparse_margin_lambda = float(sparse_margin_lambda)
        self.topk_fraction = float(topk_fraction)
        self.smooth = float(smooth)

    @staticmethod
    def _labels(target: torch.Tensor) -> torch.Tensor:
        if isinstance(target, (tuple, list)):
            target = target[0]
        if target.ndim == 5 and target.shape[1] == 1:
            return target[:, 0].long()
        if target.ndim == 5 and target.shape[1] == 5:
            return target.argmax(dim=1).long()
        return target.long()

    def _check_contract(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        # [QC][Invariant:shape_label_contract]
        # This loss indexes V5 semantic classes directly. A class-count mismatch
        # would silently train a different task and invalidate IoU-F evidence.
        if logits.ndim != 5 or logits.shape[1] != 5:
            raise ValueError(f"BICMV62SemanticBoundaryLoss expects logits [B,5,D,H,W], got {tuple(logits.shape)}")
        if labels.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                "BICMV62SemanticBoundaryLoss target shape mismatch: "
                f"labels={tuple(labels.shape)} logits={tuple(logits.shape)}"
            )
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) > 4):
            raise ValueError(
                f"BICMV62SemanticBoundaryLoss labels must be in 0..4, got min={int(labels.min())} "
                f"max={int(labels.max())}"
            )

    def _soft_targets(self, labels: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        labels = labels.clamp(0, 4)
        one = F.one_hot(labels, num_classes=5).permute(0, 4, 1, 2, 3).to(dtype=dtype, device=device)
        soft = one.clone()
        # [DATA][Risk:Medium][Scope:semantic_adjacency]
        # Adjacent smoothing is allowed for broad body labels, but contact/core
        # are kept mostly one-hot so the sparse split signal is not swallowed
        # by shell as observed in earlier V5/V3 semantic runs.
        exterior = labels == 1
        soft[:, 0][exterior], soft[:, 1][exterior], soft[:, 2][exterior] = 0.08, 0.86, 0.06
        shell = labels == 2
        soft[:, 1][shell], soft[:, 2][shell], soft[:, 3][shell], soft[:, 4][shell] = 0.04, 0.86, 0.06, 0.04
        core = labels == 3
        soft[:, 2][core], soft[:, 3][core], soft[:, 4][core] = 0.04, 0.92, 0.04
        contact = labels == 4
        soft[:, 2][contact], soft[:, 3][contact], soft[:, 4][contact] = 0.06, 0.04, 0.90
        return soft

    def _soft_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        soft = self._soft_targets(labels, logits.dtype, logits.device)
        return -(soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))
        inter = (pred * target).sum(dim=dims)
        denom = pred.sum(dim=dims) + target.sum(dim=dims)
        return (1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean()

    def _tversky(self, pred: torch.Tensor, target: torch.Tensor,
                 fp_weight: float, fn_weight: float) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))
        tp = (pred * target).sum(dim=dims)
        fp = (pred * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - pred) * target).sum(dim=dims)
        return (1.0 - (tp + self.smooth) / (tp + fp_weight * fp + fn_weight * fn + self.smooth)).mean()

    def _focal(self, prob: torch.Tensor, target: torch.Tensor,
               pos_alpha: float, neg_alpha: float) -> torch.Tensor:
        prob = prob.clamp(self.smooth, 1.0 - self.smooth)
        pos = target > 0.5
        neg = ~pos
        total = prob.new_zeros(())
        if pos.any():
            total = total + (-float(pos_alpha) * ((1.0 - prob[pos]) ** 2.0) * torch.log(prob[pos])).mean()
        if neg.any():
            total = total + (-float(neg_alpha) * (prob[neg] ** 2.0) * torch.log(1.0 - prob[neg])).mean()
        return total

    def _support_boundary_proxy(self, support_prob: torch.Tensor, support_target: torch.Tensor) -> torch.Tensor:
        # [METRIC][Risk:Major][Scope:HD95_proxy]
        # HD95 is not directly differentiable. This proxy compares a soft
        # probability boundary against a one-voxel target support boundary,
        # giving the loss a local surface term while keeping actual HD95 in
        # the full-volume evaluator.
        target = support_target.float()
        eroded = -F.max_pool3d(-target, kernel_size=3, stride=1, padding=1)
        target_boundary = (target - eroded).clamp(0.0, 1.0)
        soft_dilate = F.max_pool3d(support_prob, kernel_size=3, stride=1, padding=1)
        soft_erode = -F.max_pool3d(-support_prob, kernel_size=3, stride=1, padding=1)
        pred_boundary = (soft_dilate - soft_erode).clamp(0.0, 1.0)
        return self._dice_loss(pred_boundary, target_boundary)

    def _sparse_margin(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sparse = (labels == 3) | (labels == 4)
        if not sparse.any():
            return logits.new_zeros(())
        correct = logits.gather(1, labels.clamp(0, 4).unsqueeze(1)).squeeze(1)
        other_logits = logits.clone()
        other_logits.scatter_(1, labels.clamp(0, 4).unsqueeze(1), float("-inf"))
        best_other = other_logits.amax(dim=1)
        return torch.relu(1.0 + best_other[sparse] - correct[sparse]).mean()

    def _topk_false_contact(self, contact_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        neg = labels != 4
        if not neg.any():
            return contact_prob.new_zeros(())
        vals = contact_prob[:, 0][neg]
        if vals.numel() == 0:
            return contact_prob.new_zeros(())
        k = max(1, int(round(float(vals.numel()) * self.topk_fraction)))
        return torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        labels = self._labels(target).to(device=pred_logits.device)
        self._check_contract(pred_logits, labels)
        total = self.base(pred_logits, target)
        logits = pred_logits.float()
        prob = torch.softmax(logits, dim=1)
        support_target = ((labels == 2) | (labels == 3) | (labels == 4)).float().unsqueeze(1)
        support_prob = prob[:, 2:5].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        core_target = (labels == 3).float().unsqueeze(1)
        contact_target = (labels == 4).float().unsqueeze(1)
        core_prob = prob[:, 3:4]
        contact_prob = prob[:, 4:5]

        total = total + self.soft_ce_lambda * self._soft_ce(logits, labels)
        total = total + self.support_dice_lambda * self._dice_loss(support_prob, support_target)
        if not self.fullres_terms:
            if not torch.isfinite(total):
                raise ValueError(f"BICMV62SemanticBoundaryLoss produced non-finite aux loss: {float(total.detach().cpu())}")
            return total

        total = total + self.support_boundary_lambda * self._support_boundary_proxy(support_prob, support_target)
        total = total + self.core_tversky_lambda * self._tversky(
            core_prob, core_target, fp_weight=0.35, fn_weight=0.75
        )
        total = total + self.core_focal_lambda * self._focal(
            core_prob, core_target, pos_alpha=0.85, neg_alpha=0.10
        )
        if contact_target.any():
            total = total + self.contact_tversky_lambda * self._tversky(
                contact_prob, contact_target, fp_weight=0.45, fn_weight=0.75
            )
            total = total + self.contact_focal_lambda * self._focal(
                contact_prob, contact_target, pos_alpha=0.85, neg_alpha=0.12
            )
        total = total + self.contact_false_topk_lambda * self._topk_false_contact(contact_prob, labels)
        total = total + self.sparse_margin_lambda * self._sparse_margin(logits, labels)
        if not torch.isfinite(total):
            raise ValueError(f"BICMV62SemanticBoundaryLoss produced non-finite loss: {float(total.detach().cpu())}")
        return total


class BICMV64TopologyPrecisionLoss(BICMV62SemanticBoundaryLoss):
    """V64 loss-only diagnostic for V5 semantic topology precision.

    [AUDIT][Risk:High][Scope:single_variable_experiment]
    V62 40e case003 improved support Dice/HD95 but failed IoU-F because contact
    overfired in no-contact ROIs, LeftHip contact ratio stayed high, fragment
    57 missed its core seed, and several large LeftHip fragments produced
    multiple core components. V64 changes only the loss profile. Dataset,
    target, architecture, decoder, input channels, and 40e overfit protocol stay
    fixed so the result answers one question: can loss pressure alone correct
    the measured contact/core topology failures?

    [QA][Gate:case003_before_expansion][Status:Required]
    This class must not unlock 159/20-case/fold0 unless full-volume
    `bicm-v5-eval` on case003 passes the explicit IoU-F/contact/core gates.
    Patch pseudo Dice remains a health signal only.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        fullres_terms: bool = True,
        no_contact_roi_lambda: float = 0.70,
        contact_precision_tversky_lambda: float = 0.45,
        contact_support_false_lambda: float = 0.25,
        contact_positive_lambda: float = 0.12,
        core_positive_lambda: float = 0.25,
        core_false_topk_lambda: float = 0.25,
        core_support_margin_lambda: float = 0.20,
    ):
        super().__init__(
            base_loss,
            fullres_terms=fullres_terms,
            soft_ce_lambda=0.10,
            support_dice_lambda=0.42,
            support_boundary_lambda=0.18,
            contact_tversky_lambda=0.60,
            contact_focal_lambda=0.32,
            contact_false_topk_lambda=0.28,
            core_tversky_lambda=0.50,
            core_focal_lambda=0.32,
            sparse_margin_lambda=0.22,
            topk_fraction=0.03,
        )
        self.no_contact_roi_lambda = float(no_contact_roi_lambda)
        self.contact_precision_tversky_lambda = float(contact_precision_tversky_lambda)
        self.contact_support_false_lambda = float(contact_support_false_lambda)
        self.contact_positive_lambda = float(contact_positive_lambda)
        self.core_positive_lambda = float(core_positive_lambda)
        self.core_false_topk_lambda = float(core_false_topk_lambda)
        self.core_support_margin_lambda = float(core_support_margin_lambda)

    def _topk_mean_on_mask(self, prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of the highest probabilities under `mask`.

        [QC][Invariant:class_imbalance]
        Contact/core false positives occupy a tiny fraction of full patches but
        dominate IoU-F by creating extra ridges/seeds. Top-k focuses the loss on
        the exact high-confidence mistakes without changing inference
        thresholds.
        """
        if not mask.any():
            return prob.new_zeros(())
        vals = prob[:, 0][mask]
        if vals.numel() == 0:
            return prob.new_zeros(())
        k = max(1, int(round(float(vals.numel()) * self.topk_fraction)))
        return torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()

    def _positive_log_loss(self, prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sparse positive coverage term for tiny core/contact targets."""
        if not mask.any():
            return prob.new_zeros(())
        vals = prob[:, 0][mask]
        return -torch.log(vals.clamp(self.smooth, 1.0 - self.smooth)).mean()

    def _no_contact_roi_suppression(self, contact_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Suppress contact in samples whose target has no fracture/contact voxels.

        [DATA][Risk:High][Scope:no_contact_roi]
        Case003 Sacrum and RightHip have zero target contact but V62 predicts
        thousands of contact voxels there. Those false ridges drag contact
        precision down and can create unnecessary watershed cuts. This term is
        sample-wise: it activates only when the whole ROI has no class-4 target.
        """
        total = contact_prob.new_zeros(())
        count = 0
        for b in range(int(labels.shape[0])):
            if bool((labels[b] == 4).any()):
                continue
            roi_mask = (labels[b] == 1) | (labels[b] == 2) | (labels[b] == 3)
            if not bool(roi_mask.any()):
                continue
            vals = contact_prob[b, 0][roi_mask]
            if vals.numel() == 0:
                continue
            k = max(1, int(round(float(vals.numel()) * self.topk_fraction)))
            total = total + torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()
            count += 1
        if count == 0:
            return total
        return total / float(count)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = super().forward(pred_logits, target)
        if not self.fullres_terms:
            return total

        labels = self._labels(target).to(device=pred_logits.device)
        logits = pred_logits.float()
        prob = torch.softmax(logits, dim=1)
        core_prob = prob[:, 3:4]
        contact_prob = prob[:, 4:5]
        support_noncontact = (labels == 2) | (labels == 3)
        contact_target = labels == 4
        core_target = labels == 3
        core_false_mask = ((labels == 2) | (labels == 4)) & ~core_target

        # [METRIC][Scope:contact_precision]
        # V62 recall improved before precision; this extra Tversky term raises
        # FP cost while retaining a smaller FN cost so contact does not collapse
        # back to zero.
        total = total + self.contact_precision_tversky_lambda * self._tversky(
            contact_prob,
            contact_target.float().unsqueeze(1),
            fp_weight=0.85,
            fn_weight=0.55,
        )
        total = total + self.contact_support_false_lambda * self._topk_mean_on_mask(
            contact_prob, support_noncontact
        )
        total = total + self.no_contact_roi_lambda * self._no_contact_roi_suppression(
            contact_prob, labels
        )
        total = total + self.contact_positive_lambda * self._positive_log_loss(
            contact_prob, contact_target
        )

        # [METRIC][Scope:core_seed_topology]
        # Missing fragment 57 requires stronger positive seed coverage, while
        # multi-core large fragments require suppressing high-confidence core in
        # non-core support. This is still voxel-supervised; fragment-wise
        # topology is evaluated only in the full-volume JSON.
        total = total + self.core_positive_lambda * self._positive_log_loss(
            core_prob, core_target
        )
        total = total + self.core_false_topk_lambda * self._topk_mean_on_mask(
            core_prob, core_false_mask
        )
        support_prob = prob[:, 2:5].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        total = total + self.core_support_margin_lambda * F.relu(core_prob - support_prob).mean()
        if not torch.isfinite(total):
            raise ValueError(f"BICMV64TopologyPrecisionLoss produced non-finite loss: {float(total.detach().cpu())}")
        return total


class BICMV65BalancedContactCoreLoss(BICMV64TopologyPrecisionLoss):
    """V65 balanced contact/core loss after V64 contact collapse.

    [AUDIT][Risk:High][Scope:loss_ablation]
    V64 showed that aggressive precision/no-contact penalties can eliminate
    class-4 contact positives before the model learns the ridge. V65 keeps the
    same measured objective family but rebalances it: contact positive/recall
    is stronger than V64, while no-contact and top-k FP penalties are only
    mildly stronger than V62. This is still a loss-only experiment.
    """

    def __init__(self, base_loss: nn.Module, *, fullres_terms: bool = True):
        super().__init__(
            base_loss,
            fullres_terms=fullres_terms,
            no_contact_roi_lambda=0.18,
            contact_precision_tversky_lambda=0.16,
            contact_support_false_lambda=0.10,
            contact_positive_lambda=0.38,
            core_positive_lambda=0.32,
            core_false_topk_lambda=0.10,
            core_support_margin_lambda=0.12,
        )
        # [METRIC][Scope:contact_recall_first]
        # Parent V64 defaults were precision-first and collapsed contact.
        # These overrides restore recall pressure while keeping a small FP cost.
        self.contact_tversky_lambda = 0.70
        self.contact_focal_lambda = 0.42
        self.contact_false_topk_lambda = 0.12
        self.core_tversky_lambda = 0.55
        self.core_focal_lambda = 0.36
        self.sparse_margin_lambda = 0.22
        self.topk_fraction = 0.02


class BICMV67SemanticEdgePairLoss(BICMV62SemanticBoundaryLoss):
    """V67 semantic-head loss with local contact edge-pair ranking.

    [AUDIT][Risk:High][Scope:representation_after_v66]
    V66 proved the 5-class semantic head can learn support geometry on case003,
    but LeftHip remains blocked by contact precision and fragment-scale seed
    topology. The follow-up separability audit measured weak voxel contact
    separability (AUC 0.637 on case003) but strong local edge-pair separability
    (AUC 0.873). V67 therefore changes one thing: it keeps the V62 semantic
    support objective and adds a local contact-vs-support edge ranking term so
    true fracture/contact voxels must outrank nearby same-support non-contact
    voxels. This is not a threshold sweep and does not change the decoder.

    [QA][Gate:case003_before_expansion][Status:Required]
    Progression to 159/20-case/fold0/fulltrain remains forbidden unless
    full-volume `bicm-v5-eval` passes the original case003 gate.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        fullres_terms: bool = True,
        local_edge_rank_lambda: float = 0.55,
        absent_roi_contact_lambda: float = 0.10,
        contact_positive_floor_lambda: float = 0.18,
        support_false_contact_cap_lambda: float = 0.10,
        local_kernel_size: int = 7,
        local_pair_margin: float = 0.18,
        contact_floor: float = 0.65,
    ):
        super().__init__(
            base_loss,
            fullres_terms=fullres_terms,
            soft_ce_lambda=0.12,
            support_dice_lambda=0.45,
            support_boundary_lambda=0.18,
            contact_tversky_lambda=0.62,
            contact_focal_lambda=0.34,
            contact_false_topk_lambda=0.08,
            core_tversky_lambda=0.40,
            core_focal_lambda=0.22,
            sparse_margin_lambda=0.18,
            topk_fraction=0.01,
        )
        self.local_edge_rank_lambda = float(local_edge_rank_lambda)
        self.absent_roi_contact_lambda = float(absent_roi_contact_lambda)
        self.contact_positive_floor_lambda = float(contact_positive_floor_lambda)
        self.support_false_contact_cap_lambda = float(support_false_contact_cap_lambda)
        self.local_kernel_size = int(local_kernel_size)
        self.local_pair_margin = float(local_pair_margin)
        self.contact_floor = float(contact_floor)

    def _local_edge_rank_loss(self, contact_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Rank contact voxels above nearby support-internal non-contact voxels."""
        contact_mask = labels == 4
        if not contact_mask.any():
            return contact_prob.new_zeros(())
        support_noncontact = (labels == 2) | (labels == 3)
        if not support_noncontact.any():
            return contact_prob.new_zeros(())
        k = max(3, int(self.local_kernel_size))
        if k % 2 == 0:
            k += 1
        neg_prob = contact_prob[:, 0] * support_noncontact.float()
        local_hard_neg = F.max_pool3d(
            neg_prob.unsqueeze(1),
            kernel_size=k,
            stride=1,
            padding=k // 2,
        )[:, 0]
        pos_prob = contact_prob[:, 0][contact_mask]
        hard_neg = local_hard_neg[contact_mask]
        # [METRIC][Scope:local_topology]
        # The loss is local: a true contact voxel only has to outrank nearby
        # support non-contact voxels. This targets the V66 LeftHip pattern
        # where recall is high but the contact ridge spreads across shell.
        return F.relu(float(self.local_pair_margin) + hard_neg - pos_prob).pow(2).mean()

    def _absent_roi_contact_suppression(self, contact_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Suppress tiny contact islands in no-contact anatomy ROI samples."""
        total = contact_prob.new_zeros(())
        count = 0
        for b in range(int(labels.shape[0])):
            if bool((labels[b] == 4).any()):
                continue
            roi_mask = (labels[b] == 1) | (labels[b] == 2) | (labels[b] == 3)
            if not bool(roi_mask.any()):
                continue
            vals = contact_prob[b, 0][roi_mask]
            if vals.numel() == 0:
                continue
            k = max(1, int(round(float(vals.numel()) * self.topk_fraction)))
            total = total + torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean()
            count += 1
        return total if count == 0 else total / float(count)

    def _contact_positive_floor(self, contact_prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        contact_mask = labels == 4
        if not contact_mask.any():
            return contact_prob.new_zeros(())
        vals = contact_prob[:, 0][contact_mask]
        # [QC][Invariant:fixed_threshold_training]
        # Full-volume decoding still uses argmax labels. This floor does not
        # tune inference thresholds; it simply keeps true contact logits from
        # falling behind shell/core during early epochs.
        return F.relu(float(self.contact_floor) - vals).pow(2).mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        labels = self._labels(target).to(device=pred_logits.device)
        self._check_contract(pred_logits, labels)
        total = super().forward(pred_logits, target)
        if not self.fullres_terms:
            return total

        prob = torch.softmax(pred_logits.float(), dim=1)
        contact_prob = prob[:, 4:5]
        support_noncontact = (labels == 2) | (labels == 3)
        total = total + self.local_edge_rank_lambda * self._local_edge_rank_loss(contact_prob, labels)
        total = total + self.absent_roi_contact_lambda * self._absent_roi_contact_suppression(contact_prob, labels)
        total = total + self.contact_positive_floor_lambda * self._contact_positive_floor(contact_prob, labels)
        total = total + self.support_false_contact_cap_lambda * self._topk_false_contact(contact_prob, labels)
        if support_noncontact.any():
            # A small global cap complements the local rank term without the
            # strong V64-style pressure that collapsed all contact positives.
            total = total + 0.05 * contact_prob[:, 0][support_noncontact].mean()
        if not torch.isfinite(total):
            raise ValueError(f"BICMV67SemanticEdgePairLoss produced non-finite loss: {float(total.detach().cpu())}")
        return total


class BICMV5SparseDeepSupervisionWrapper(nn.Module):
    """Apply V5 sparse rare-class terms only at full resolution.

    [AUDIT][Risk:High][Scope:deep_supervision]
    The V5 core markers are often single-digit voxels. Low-resolution deep
    supervision can erase those markers during downsampling, so sparse
    core/contact terms are restricted to the full-resolution output. Auxiliary
    scales still receive base DC+CE plus support Dice as a coarse geometry
    health check.
    """

    def __init__(self, fullres_loss: nn.Module, aux_loss: nn.Module, weight_factors):
        super().__init__()
        self.fullres_loss = fullres_loss
        self.aux_loss = aux_loss
        self.weight_factors = tuple(float(x) for x in weight_factors)
        if not any(x != 0 for x in self.weight_factors):
            raise ValueError("At least one deep-supervision weight must be nonzero")

    def forward(self, *args):
        if not all(isinstance(i, (tuple, list)) for i in args):
            raise AssertionError(f"all args must be tuple/list, got {[type(i) for i in args]}")
        total = None
        for idx, inputs in enumerate(zip(*args)):
            weight = self.weight_factors[idx]
            if weight == 0:
                continue
            loss = self.fullres_loss(*inputs) if idx == 0 else self.aux_loss(*inputs)
            total = weight * loss if total is None else total + weight * loss
        return total


# =============================================================================
# 1. Boundary DoU loss (3D) — surface-aware regularizer
# =============================================================================
class BoundaryDoULoss3D(nn.Module):
    """3D adaptation of Sun et al. MICCAI'23 Boundary DoU loss.

    Algorithm (per foreground class c, each batch item):
      1. Identify "interior" voxels of the target: voxels whose 6 axis-aligned
         neighbors are ALL inside the target. Boundary band = target − interior.
      2. Compute boundary fraction α_c = 1 − boundary / total (per class).
         Clip to ≤ 0.8 for numerical stability (paper recommendation).
      3. Loss_c = (z + y − 2·intersect) / (z + y − (1 + α)·intersect)
         where z = ||score||², y = ||target||², intersect = ⟨score, target⟩.
         When α → 1 (target mostly interior), denom → numer, loss saturates.
         When α → 0 (target is mostly boundary), loss reduces to DoU acting
         like standard Dice but emphasizing boundary alignment.
      4. Mean across foreground classes (background excluded).

    Why 3D adaptation matters:
        Original paper does per-slice 2D conv on (B, H, W). For volumetric CT
        (B, C, D, H, W), 2D convs would only see in-plane neighbors; we use a
        7-tap 3D cross kernel so erosion respects through-plane connectivity.
        This is critical for PENGWIN where fracture surfaces span multiple
        slices.

    Args:
        n_classes: total number of classes including background. Background
            (class 0) is skipped in the loss sum.
        smooth: numerical smoothing constant (1e-5 matches paper).
        background_index: class index treated as background (default 0).

    Forward:
        logits:      (B, C, D, H, W) raw network outputs (pre-softmax)
        target:      (B, D, H, W) integer class labels OR (B, 1, D, H, W)
    Returns: scalar loss (mean over foreground classes).
    """

    def __init__(self, n_classes: int, smooth: float = 1e-5,
                 background_index: int = 0):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.background_index = background_index
        # 6-connected 3D cross kernel (D=H=W=3). 7 ones total: center + 6 neighbors.
        # Convolving binary target with this counts how many of {self ∪ 6 axis
        # neighbors} are foreground. Maximum value 7 = "fully interior".
        kernel = torch.zeros(1, 1, 3, 3, 3)
        kernel[0, 0, 1, 1, 1] = 1   # center
        kernel[0, 0, 0, 1, 1] = 1   # -z
        kernel[0, 0, 2, 1, 1] = 1   # +z
        kernel[0, 0, 1, 0, 1] = 1   # -y
        kernel[0, 0, 1, 2, 1] = 1   # +y
        kernel[0, 0, 1, 1, 0] = 1   # -x
        kernel[0, 0, 1, 1, 2] = 1   # +x
        # Buffer (not parameter) — moves with .to(device) automatically.
        self.register_buffer("kernel", kernel)
        self._kernel_sum = float(kernel.sum().item())

    def _alpha(self, target_c: torch.Tensor) -> torch.Tensor:
        """Boundary fraction-derived α coefficient for one class.

        target_c: (B, D, H, W) binary {0,1} mask.
        Returns scalar α ∈ [-1, 0.8] (clipped per paper).
        """
        if target_c.sum() == 0:
            # Empty class in this batch — degenerate; return α=0 (≡ standard Dice form).
            return target_c.new_zeros(())

        x = target_c.unsqueeze(1).float()                # (B,1,D,H,W) — float32
        # AMP+device safe: kernel buffer was registered without .cuda(), and
        # nnU-Net's `loss = loss.to(device)` may not always propagate to
        # buffers added in subclasses. Explicitly match BOTH device + dtype.
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        conv = F.conv3d(x, kernel, padding=1)       # neighbor sum
        # Voxel is "interior" iff ALL 7 kernel positions land on foreground.
        # Strict equality avoids surface tangents counting as interior.
        interior = (conv >= self._kernel_sum) & (x > 0)
        n_interior = interior.float().sum()
        n_total = x.sum() + self.smooth
        boundary_frac = 1.0 - (n_interior / n_total)
        # α = 1 - boundary_frac (so high α = "mostly interior" → loss less aggressive)
        alpha = 1.0 - boundary_frac
        # Map [0,1] → [-1, 1] then clamp ≤ 0.8 (paper Sec. 3.2 stability).
        alpha = (2 * alpha - 1).clamp(max=0.8)
        return alpha

    def _per_class(self, score_c: torch.Tensor,
                   target_c: torch.Tensor) -> torch.Tensor:
        """DoU-like loss for one class (paper Eq. 8 in 3D form)."""
        alpha = self._alpha(target_c)
        target_f = target_c.float()
        intersect = (score_c * target_f).sum()
        y_sum = (target_f * target_f).sum()
        z_sum = (score_c * score_c).sum()
        num = z_sum + y_sum - 2 * intersect + self.smooth
        den = z_sum + y_sum - (1.0 + alpha) * intersect + self.smooth
        return num / den

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Compute mean boundary DoU loss across foreground classes.

        Inputs are aligned with nnU-Net convention:
            logits:  (B, C, D, H, W)  -- pre-softmax (any dtype: AMP fp16 OK)
            target:  (B, D, H, W) or (B, 1, D, H, W) integer labels

        AMP-safe: disables `torch.cuda.amp.autocast` inside, so all internal
        arithmetic (softmax + conv3d + DoU formula) runs in float32 regardless
        of the surrounding autocast context. Without this guard, calling
        `tensor.float()` inside autocast still gets re-cast to half on
        autocast-enabled ops (conv3d, mul, etc.), which then mismatches our
        float32 kernel buffer. nnU-Net's GradScaler handles the float32 scalar
        return correctly. Boundary loss is <5% of total compute so float32 is
        cheap.
        """
        if target.dim() == 5:
            target = target.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            logits_f32 = logits.float()
            probs = F.softmax(logits_f32, dim=1)
            n_fg = 0
            loss = logits_f32.new_zeros(())
            for c in range(self.n_classes):
                if c == self.background_index:
                    continue
                target_c = (target == c).long()
                score_c = probs[:, c]
                # Skip empty-on-both-sides classes to avoid degenerate gradients.
                if target_c.sum() == 0 and score_c.sum() < self.smooth:
                    continue
                loss = loss + self._per_class(score_c, target_c)
                n_fg += 1
            if n_fg == 0:
                return loss
            return loss / n_fg


# =============================================================================
# 2. Compound loss (Dice + CE + BoundaryDoU) wrapper
# =============================================================================
class DC_CE_BD_loss(nn.Module):
    """Compound loss: w_dc·Dice + w_ce·CE + w_bd·BoundaryDoU.

    Designed as a drop-in replacement for nnU-Net's `DC_and_CE_loss` so the
    deep-supervision wrapper continues to work unchanged. Wraps an existing
    `DC_and_CE_loss` instance (passed in) and adds a BoundaryDoU3D term.

    Default weights chosen for PENGWIN:
        w_dc = 1.0, w_ce = 1.0, w_bd = 0.3
        — Sun et al. recommend ~0.3 for additive boundary; matches typical
        compound loss budget where boundary pulls without dominating.

    Why we wrap rather than re-implement DC+CE:
        nnU-Net's DC_and_CE_loss handles `ignore_label`, `batch_dice`,
        DDP smoothing, and edge cases. Re-implementing risks subtle bugs.

    Args:
        dc_ce_loss:  pre-built nnU-Net DC_and_CE_loss instance.
        n_classes:   number of segmentation classes (incl. background).
        weight_dc_ce: weight for the DC+CE term (default 1.0).
        weight_bd:   weight for the boundary DoU term (default 0.3).

    Forward:
        logits:  (B, C, D, H, W)
        target:  (B, 1, D, H, W) integer labels (nnU-Net convention)
    """

    def __init__(self, dc_ce_loss: nn.Module, n_classes: int,
                 weight_dc_ce: float = 1.0, weight_bd: float = 0.3):
        super().__init__()
        self.dc_ce = dc_ce_loss
        self.bd = BoundaryDoULoss3D(n_classes=n_classes)
        self.weight_dc_ce = weight_dc_ce
        self.weight_bd = weight_bd
        # Expose `.dc` so nnU-Net's torch.compile path (which calls
        # `loss.dc = torch.compile(loss.dc)` in nnUNetTrainer._build_loss)
        # still works without modification.
        if hasattr(dc_ce_loss, "dc"):
            self.dc = dc_ce_loss.dc

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        l_dc_ce = self.dc_ce(logits, target)
        l_bd = self.bd(logits, target)
        return self.weight_dc_ce * l_dc_ce + self.weight_bd * l_bd


# =============================================================================
# 3. Tversky loss (3D) — weak-class / recall-oriented ablation
# =============================================================================
class SoftTverskyLoss3D(nn.Module):
    """Weighted multi-class Tversky loss for 3D anatomy segmentation.

    Tversky generalizes Dice with separate false-positive and false-negative
    weights:

        T = TP / (TP + alpha * FP + beta * FN)

    By default this keeps foreground-only behavior. Ablations can pass
    `active_classes` to include a targeted subset, including background.

    Args:
        n_classes: total classes including background.
        alpha: false-positive weight. Smaller values tolerate extra foreground.
        beta: false-negative weight. Larger values emphasize recall.
        smooth: numerical smoothing.
        background_index: class skipped only when active_classes is omitted.
        active_classes: explicit class IDs to include. If None, use all classes
            except background_index.
        class_weights: optional dense list or {class_id: weight}; weights are
            normalized by the sum of weights actually used in the batch.
    """

    def __init__(self, n_classes: int, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1e-5, background_index: int = 0,
                 active_classes: list[int] | tuple[int, ...] | None = None,
                 class_weights: list[float] | tuple[float, ...] | dict[int, float] | None = None):
        super().__init__()
        self.n_classes = int(n_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.background_index = int(background_index)
        if active_classes is None:
            active_classes = [c for c in range(self.n_classes) if c != self.background_index]
        self.active_classes = tuple(int(c) for c in active_classes if 0 <= int(c) < self.n_classes)

        weight_arr = torch.ones(self.n_classes, dtype=torch.float32)
        if class_weights is not None:
            if isinstance(class_weights, dict):
                for k, v in class_weights.items():
                    k = int(k)
                    if 0 <= k < self.n_classes:
                        weight_arr[k] = float(v)
            else:
                arr = torch.as_tensor(class_weights, dtype=torch.float32)
                if arr.numel() != self.n_classes:
                    raise ValueError(
                        f"class_weights must have {self.n_classes} values, got {arr.numel()}"
                    )
                weight_arr = arr.reshape(self.n_classes)
        self.register_buffer("class_weights", weight_arr)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 5:
            target = target.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            logits_f32 = logits.float()
            probs = F.softmax(logits_f32, dim=1)
            loss = logits_f32.new_zeros(())
            weight_sum = logits_f32.new_zeros(())
            for c in self.active_classes:
                target_c = (target == c).float()
                score_c = probs[:, c]
                # Empty-on-both classes add noise but no useful gradient.
                if target_c.sum() == 0 and score_c.sum() < self.smooth:
                    continue
                tp = (score_c * target_c).sum()
                fp = (score_c * (1.0 - target_c)).sum()
                fn = ((1.0 - score_c) * target_c).sum()
                tversky = (tp + self.smooth) / (
                    tp + self.alpha * fp + self.beta * fn + self.smooth
                )
                weight = self.class_weights.to(device=logits_f32.device, dtype=logits_f32.dtype)[c]
                loss = loss + weight * (1.0 - tversky)
                weight_sum = weight_sum + weight
            if weight_sum <= 0:
                return loss
            return loss / weight_sum


class ContactEnergyLoss3D(nn.Module):
    """Non-Dice objective for Contact-Energy ABBC training.

    Contact-Energy V2 should not optimize the class-3 fracture/contact surface
    as a broad overlap mask. The decoder uses contact probability as a crossing
    cost, so false-positive contact ridges are expensive and stable core seeds
    are required for marker assignment. This loss therefore uses:

    - weighted cross entropy for the full semantic contract;
    - focal one-vs-rest contact supervision, with stronger negative penalty;
    - focal one-vs-rest core supervision for marker stability.

    No Dice/Tversky term is used here. Deep supervision can wrap this module in
    the same way it wraps nnU-Net's default loss.
    """

    def __init__(self, n_classes: int,
                 ce_weight: torch.Tensor | None = None,
                 ignore_label: int | None = None,
                 contact_class: int = 3,
                 core_class: int = 2,
                 contact_weight: float = 2.0,
                 contact_fp_weight: float = 3.0,
                 contact_pos_weight: float = 1.0,
                 core_weight: float = 0.75,
                 core_fp_weight: float = 1.0,
                 core_pos_weight: float = 1.25,
                 gamma: float = 2.0,
                 eps: float = 1e-6):
        super().__init__()
        self.n_classes = int(n_classes)
        self.ignore_label = ignore_label
        self.contact_class = int(contact_class)
        self.core_class = int(core_class)
        self.contact_weight = float(contact_weight)
        self.contact_fp_weight = float(contact_fp_weight)
        self.contact_pos_weight = float(contact_pos_weight)
        self.core_weight = float(core_weight)
        self.core_fp_weight = float(core_fp_weight)
        self.core_pos_weight = float(core_pos_weight)
        self.gamma = float(gamma)
        self.eps = float(eps)
        if ce_weight is None:
            self.register_buffer("ce_weight", None)
        else:
            self.register_buffer("ce_weight", ce_weight.detach().float().clone())

    def _binary_focal(self, prob: torch.Tensor, target: torch.Tensor,
                      valid: torch.Tensor, *, pos_weight: float,
                      neg_weight: float) -> torch.Tensor:
        prob = prob.clamp(self.eps, 1.0 - self.eps)
        target = target.bool() & valid
        negative = (~target) & valid
        loss = prob.new_zeros(())
        denom = prob.new_zeros(())
        if target.any():
            pos = -((1.0 - prob[target]) ** self.gamma) * torch.log(prob[target])
            loss = loss + float(pos_weight) * pos.mean()
            denom = denom + float(pos_weight)
        if negative.any():
            neg = -(prob[negative] ** self.gamma) * torch.log1p(-prob[negative])
            loss = loss + float(neg_weight) * neg.mean()
            denom = denom + float(neg_weight)
        if denom <= 0:
            return loss
        return loss / denom

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 5:
            target = target.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            logits_f32 = logits.float()
            target_long = target.long()
            valid = torch.ones_like(target_long, dtype=torch.bool)
            ce_target = target_long
            if self.ignore_label is not None:
                valid = target_long != int(self.ignore_label)
                ce_target = target_long.clone()
                ce_target[~valid] = 0

            ce_weight = None
            if self.ce_weight is not None:
                ce_weight = self.ce_weight.to(device=logits_f32.device, dtype=logits_f32.dtype)
            ce = F.cross_entropy(
                logits_f32,
                ce_target,
                weight=ce_weight,
                ignore_index=int(self.ignore_label) if self.ignore_label is not None else -100,
            )

            probs = F.softmax(logits_f32, dim=1)
            contact_target = ce_target == self.contact_class
            core_target = ce_target == self.core_class

            # Contact negatives are evaluated inside the valid region. Normalizing
            # positive and negative terms separately prevents background volume
            # from overwhelming the sparse contact surface.
            contact = self._binary_focal(
                probs[:, self.contact_class],
                contact_target,
                valid,
                pos_weight=self.contact_pos_weight,
                neg_weight=self.contact_fp_weight,
            )
            core = self._binary_focal(
                probs[:, self.core_class],
                core_target,
                valid,
                pos_weight=self.core_pos_weight,
                neg_weight=self.core_fp_weight,
            )
            return ce + self.contact_weight * contact + self.core_weight * core


class ContactFuseLoss3D(ContactEnergyLoss3D):
    """Contact-LegacyFuse objective with explicit hard-negative supervision.

    Dataset537 adds class 4 (`contact_hard_negative`) for voxels where contact
    false positives are especially damaging: outside-pelvis bone-like CT,
    foundation leakage, and non-contact cortical surfaces. The class is not a
    decoder fragment. It exists to teach the contact head where *not* to build
    an energy ridge.

    The objective keeps dense CE for the 5-class semantic contract, then adds:
        - precision-oriented contact focal loss;
        - core focal loss for marker stability;
        - class-4 focal loss for hard-negative recognition;
        - online top-k false-contact mining over non-contact voxels.

    No Dice/Tversky term is used for contact because overlap-heavy objectives
    encouraged high-recall, low-precision separator maps in the previous runs.
    """

    def __init__(self, n_classes: int,
                 ce_weight: torch.Tensor | None = None,
                 ignore_label: int | None = None,
                 contact_class: int = 3,
                 core_class: int = 2,
                 hard_negative_class: int = 4,
                 contact_weight: float = 2.5,
                 contact_fp_weight: float = 5.0,
                 contact_pos_weight: float = 0.75,
                 core_weight: float = 0.80,
                 core_fp_weight: float = 1.0,
                 core_pos_weight: float = 1.35,
                 hard_negative_weight: float = 1.25,
                 hard_negative_contact_penalty: float = 2.5,
                 topk_false_contact_weight: float = 1.5,
                 topk_fraction: float = 0.02,
                 gamma: float = 2.0,
                 eps: float = 1e-6):
        super().__init__(
            n_classes=n_classes,
            ce_weight=ce_weight,
            ignore_label=ignore_label,
            contact_class=contact_class,
            core_class=core_class,
            contact_weight=contact_weight,
            contact_fp_weight=contact_fp_weight,
            contact_pos_weight=contact_pos_weight,
            core_weight=core_weight,
            core_fp_weight=core_fp_weight,
            core_pos_weight=core_pos_weight,
            gamma=gamma,
            eps=eps,
        )
        self.hard_negative_class = int(hard_negative_class)
        self.hard_negative_weight = float(hard_negative_weight)
        self.hard_negative_contact_penalty = float(hard_negative_contact_penalty)
        self.topk_false_contact_weight = float(topk_false_contact_weight)
        self.topk_fraction = float(topk_fraction)

    def _topk_false_contact(self, contact_prob: torch.Tensor,
                            contact_target: torch.Tensor,
                            valid: torch.Tensor) -> torch.Tensor:
        negative = valid & ~contact_target.bool()
        if not negative.any():
            return contact_prob.new_zeros(())
        vals = contact_prob[negative].clamp(self.eps, 1.0 - self.eps)
        k = max(1, int(round(float(vals.numel()) * self.topk_fraction)))
        vals = torch.topk(vals, k=min(k, vals.numel()), largest=True).values
        return -((vals ** self.gamma) * torch.log1p(-vals)).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 5:
            target = target.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            logits_f32 = logits.float()
            target_long = target.long()
            valid = torch.ones_like(target_long, dtype=torch.bool)
            ce_target = target_long
            if self.ignore_label is not None:
                valid = target_long != int(self.ignore_label)
                ce_target = target_long.clone()
                ce_target[~valid] = 0

            ce_weight = None
            if self.ce_weight is not None:
                ce_weight = self.ce_weight.to(device=logits_f32.device, dtype=logits_f32.dtype)
            ce = F.cross_entropy(
                logits_f32,
                ce_target,
                weight=ce_weight,
                ignore_index=int(self.ignore_label) if self.ignore_label is not None else -100,
            )

            probs = F.softmax(logits_f32, dim=1)
            contact_prob = probs[:, self.contact_class]
            contact_target = ce_target == self.contact_class
            core_target = ce_target == self.core_class
            hard_target = ce_target == self.hard_negative_class

            contact = self._binary_focal(
                contact_prob,
                contact_target,
                valid,
                pos_weight=self.contact_pos_weight,
                neg_weight=self.contact_fp_weight,
            )
            core = self._binary_focal(
                probs[:, self.core_class],
                core_target,
                valid,
                pos_weight=self.core_pos_weight,
                neg_weight=self.core_fp_weight,
            )
            hard_negative = self._binary_focal(
                probs[:, self.hard_negative_class],
                hard_target,
                valid,
                pos_weight=1.5,
                neg_weight=0.5,
            )

            hard_contact_penalty = contact_prob.new_zeros(())
            if hard_target.any():
                hp = contact_prob[hard_target & valid].clamp(self.eps, 1.0 - self.eps)
                hard_contact_penalty = -((hp ** self.gamma) * torch.log1p(-hp)).mean()

            topk_false = self._topk_false_contact(contact_prob, contact_target, valid)
            return (
                ce
                + self.contact_weight * contact
                + self.core_weight * core
                + self.hard_negative_weight * hard_negative
                + self.hard_negative_contact_penalty * hard_contact_penalty
                + self.topk_false_contact_weight * topk_false
            )


class DC_CE_TV_BD_loss(nn.Module):
    """Compound loss: DC+CE plus optional Tversky and BoundaryDoU terms.

    This wrapper exists for controlled split-anatomy upgrades:

    - `tversky_07`: DC+CE + 0.5 * Tversky(alpha=0.3, beta=0.7)
    - `tversky_08`: DC+CE + 0.5 * Tversky(alpha=0.2, beta=0.8)
    - `combo_tversky_bd005`: DC+CE + 0.35 * Tversky + 0.05 * BoundaryDoU
    - `abbc_contact_energy_v1`: Contact-Energy ABBC core/contact Tversky for
      stable seeds and precise fracture surfaces

    We keep DC+CE as the anchor term so the optimization landscape remains
    close to the already-good baseline. Tversky/BD are small corrective forces,
    not replacements.
    """

    def __init__(self, dc_ce_loss: nn.Module, n_classes: int,
                 weight_dc_ce: float = 1.0,
                 weight_tversky: float = 0.0,
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7,
                 weight_bd: float = 0.0,
                 tversky_active_classes: list[int] | tuple[int, ...] | None = None,
                 tversky_class_weights: list[float] | tuple[float, ...] | dict[int, float] | None = None):
        super().__init__()
        self.dc_ce = dc_ce_loss
        self.weight_dc_ce = float(weight_dc_ce)
        self.weight_tversky = float(weight_tversky)
        self.weight_bd = float(weight_bd)
        self.tversky = SoftTverskyLoss3D(
            n_classes=n_classes,
            alpha=tversky_alpha,
            beta=tversky_beta,
            active_classes=tversky_active_classes,
            class_weights=tversky_class_weights,
        )
        self.bd = BoundaryDoULoss3D(n_classes=n_classes) if weight_bd > 0 else None
        if hasattr(dc_ce_loss, "dc"):
            self.dc = dc_ce_loss.dc

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.weight_dc_ce * self.dc_ce(logits, target)
        if self.weight_tversky > 0:
            loss = loss + self.weight_tversky * self.tversky(logits, target)
        if self.bd is not None and self.weight_bd > 0:
            loss = loss + self.weight_bd * self.bd(logits, target)
        return loss


# =============================================================================
# 4. Class weight estimation — frequency-inverse with PENGWIN tweaks
# =============================================================================
def compute_class_weights(label_array_or_iter, n_classes: int,
                          min_weight: float = 0.5,
                          max_weight: float = 2.0,
                          smoothing: float = 1.0) -> np.ndarray:
    """Compute frequency-inverse class weights for cross-entropy.

    Args:
        label_array_or_iter: Either a single np.ndarray of integer labels OR
            an iterable of arrays (will be summed). For PENGWIN we pass a
            list of training-set GT label arrays (one per case).
        n_classes: total class count (incl. background).
        min_weight, max_weight: clamp range. Default (0.5, 2.0) prevents
            the rare class (e.g., a 0.05% sec fragment) from dominating
            gradients while still up-weighting it ~2×.
        smoothing: pseudocount added to each class count to avoid div-by-zero
            for classes absent in the sample. 1.0 ≡ Laplace smoothing.

    Returns:
        np.ndarray of shape (n_classes,), dtype float32.

    Algorithm:
        counts[c] = #voxels labeled c (across all provided arrays)
        freq[c] = (counts[c] + smoothing) / sum(counts + smoothing)
        weight[c] = 1 / freq[c], normalized so mean(weight) = 1.0,
                    then clamped to [min_weight, max_weight].

    Why PENGWIN specifics:
        Background dominates most voxels. Standard inverse frequency can give
        extreme rare-class weights, which can hurt calibration and boundary
        quality. Clamping to [0.5, 2.0] keeps this an ablation knob rather than
        a hidden training regime change.
    """
    if isinstance(label_array_or_iter, np.ndarray):
        arrays = [label_array_or_iter]
    else:
        arrays = list(label_array_or_iter)

    counts = np.zeros(n_classes, dtype=np.float64)
    for arr in arrays:
        # bincount is O(N); minlength ensures we cover all classes even if some
        # don't appear in this particular case (Pelvic case has no Femur, etc.).
        bc = np.bincount(arr.ravel(), minlength=n_classes).astype(np.float64)
        if len(bc) > n_classes:
            bc = bc[:n_classes]
        counts[:len(bc)] += bc

    counts_smooth = counts + smoothing
    freq = counts_smooth / counts_smooth.sum()
    weights = 1.0 / freq
    weights /= weights.mean()                  # mean-1 normalization
    weights = np.clip(weights, min_weight, max_weight).astype(np.float32)
    return weights


def compute_median_frequency_class_weights(label_array_or_iter, n_classes: int,
                                           min_weight: float = 0.5,
                                           max_weight: float = 2.0,
                                           smoothing: float = 1.0) -> np.ndarray:
    """Compute clipped median-frequency CE weights.

    This is the trainer's `CE_CLASS_WEIGHTS="auto"` implementation. Median
    frequency weighting is less aggressive than raw inverse frequency:

        weight[c] = median(freq[present classes]) / freq[c]

    The result is clipped to [0.5, 2.0], because PENGWIN IoU-F is very sensitive
    to fragment topology. Extreme CE weights can create more foreground blobs
    and worsen unmatched predicted fragment counts even if anatomy Dice rises.
    """
    if isinstance(label_array_or_iter, np.ndarray):
        arrays = [label_array_or_iter]
    else:
        arrays = list(label_array_or_iter)

    counts = np.zeros(n_classes, dtype=np.float64)
    for arr in arrays:
        bc = np.bincount(arr.ravel(), minlength=n_classes).astype(np.float64)
        counts += bc[:n_classes]
    counts_smooth = counts + smoothing
    freq = counts_smooth / counts_smooth.sum()
    present = counts > 0
    median = np.median(freq[present]) if present.any() else np.median(freq)
    weights = median / freq
    weights = np.clip(weights, min_weight, max_weight).astype(np.float32)
    return weights


# =============================================================================
# 4. Self-test (smoke) — run as `python loss.py`
# =============================================================================
def _smoke_test():
    """Quick sanity check — verify shapes/finite values on small CPU tensors."""
    print("[loss.py smoke]")
    torch.manual_seed(0)
    B, C, D, H, W = 1, 4, 8, 16, 16
    logits = torch.randn(B, C, D, H, W, requires_grad=True)
    target = torch.randint(0, C, (B, D, H, W))

    # 1. BoundaryDoULoss3D
    bd = BoundaryDoULoss3D(n_classes=C)
    val = bd(logits, target)
    assert torch.isfinite(val), f"BD loss not finite: {val}"
    val.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print(f"  BoundaryDoULoss3D({C}c): {val.item():.4f}  ✓")

    # 2. compute_class_weights
    arr = np.random.randint(0, 4, size=(20, 20, 20))
    w = compute_class_weights(arr, n_classes=4)
    assert w.shape == (4,) and np.all(w > 0) and np.all(np.isfinite(w))
    print(f"  compute_class_weights: weights={w} (mean={w.mean():.3f})  ✓")

    print("[loss.py smoke] OK")


if __name__ == "__main__":
    _smoke_test()
