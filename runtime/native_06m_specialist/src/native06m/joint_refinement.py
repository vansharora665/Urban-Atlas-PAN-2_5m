"""Joint semantic refinement; a development stage, not a geographic-readiness bypass."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import NativeAttentionUNetPP
from .road_detail import conv_block, road_margin


STAGE = "limited_geography_joint_fiveclass_development_pilot"


def scoped_test_metrics(metrics):
    reference = np.asarray(metrics["confusion_matrix"]).sum(1)
    observed = np.flatnonzero(reference > 0).tolist()
    if not observed:
        raise ValueError("No reviewed test truth is available")
    return {**metrics, "classes_with_reviewed_test_truth": observed,
            "classes_without_reviewed_test_truth": np.flatnonzero(reference == 0).tolist(),
            "observed_class_mean_iou": float(np.mean([metrics["class_iou"][i] for i in observed])),
            "full_fiveclass_test_coverage": len(observed) == 5,
            "scope": "Sparse Chennai reference pixels; classes without reference have no measured recall/generalization. Formula mean_iou is not a five-class coverage claim."}


class JointNativeRefinement(NativeAttentionUNetPP):
    def __init__(self, encoder_name="resnet34", in_channels=4):
        super().__init__(encoder_name, in_channels)
        self.native_detail = nn.Sequential(conv_block(in_channels, 16), conv_block(16, 16, 2))
        self.native_fusion = nn.Sequential(conv_block(32, 32), conv_block(32, 16, 3))
        self.native_output = nn.Conv2d(16, 5, 1)
        nn.init.zeros_(self.native_output.weight)
        nn.init.zeros_(self.native_output.bias)

    def initialize_baseline(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith(("native_detail.", "native_fusion.", "native_output.")) for key in missing):
            raise ValueError("Baseline parameters do not match the expected native five-class architecture")

    def forward(self, image):
        decoded = self.decode_features(image)
        detail = self.native_detail(image)
        residual = self.native_output(self.native_fusion(torch.cat((decoded, detail), dim=1)))
        return {"semantic": self.network.segmentation_head(decoded) + residual,
                "road": self.road_head(decoded), "boundary": self.boundary_head(decoded)}

    def configure_training(self, epoch, thaw_epoch):
        self.train()
        self.network.encoder.requires_grad_(False)
        if epoch >= thaw_epoch:
            self.network.encoder.layer3.requires_grad_(True)
            self.network.encoder.layer4.requires_grad_(True)
        # Small, correlated native crops must not rewrite running statistics.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def road_hard_negative_loss(margin, target, negative_ratio=4, negative_only_limit=4096):
    """Mine known difficult negatives only; Unknown is never road background."""
    valid = target != 255
    positive = (target == 1) & valid
    negative = (target != 1) & valid
    logits = margin[:, 0]
    terms = []
    if positive.any():
        terms.append(F.softplus(-logits[positive]).mean())
    if negative.any():
        errors = F.softplus(logits[negative])
        limit = negative_ratio * int(positive.sum()) if positive.any() else negative_only_limit
        terms.append(errors.topk(min(len(errors), max(1, limit))).values.mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def reviewed_joint_loss(output, target, config):
    logits = output["semantic"]
    valid = target != 255
    if not valid.any():
        return logits.sum() * 0
    losses = F.cross_entropy(logits, target, ignore_index=255, reduction="none")
    per_class = torch.stack([losses[target == index].mean() for index in range(5) if (target == index).any()])
    mixture = config["class_balanced_ce_fraction"]
    ce = (1 - mixture) * losses[valid].mean() + mixture * per_class.mean()
    truth = F.one_hot(target.clamp_max(4), 5).permute(0, 3, 1, 2).float() * valid[:, None]
    probability = logits.softmax(1) * valid[:, None]
    intersection = (truth * probability).sum((0, 2, 3))
    union = (truth + probability).sum((0, 2, 3))
    present = truth.sum((0, 2, 3)) > 0
    dice = (1 - (2 * intersection + 1e-5) / (union + 1e-5))[present].mean()
    semantic_road = road_hard_negative_loss(road_margin(logits), target, config["hard_negative_ratio"])
    auxiliary = road_hard_negative_loss(output["road"], target, config["hard_negative_ratio"])
    return ce + config["dice_weight"] * dice + config["semantic_road_weight"] * semantic_road + config["road_aux_weight"] * auxiliary


def weak_view(tile, teacher_probability):
    with np.load(tile["array"]) as data:
        values = [data["image"].copy(), data["road_core"].astype(bool),
                  data["mapped_road_halo"].astype(bool), data["valid"].astype(bool), teacher_probability.copy()]
    rotation = int(np.random.randint(4))
    values = [np.rot90(value, rotation, axes=(-2, -1)) for value in values]
    for axis in (-1, -2):
        if np.random.rand() < .5:
            values = [np.flip(value, axis=axis) for value in values]
    values[0] = np.clip(values[0] * np.random.uniform(.85, 1.15) + np.random.uniform(-.06, .06), 0, 1) ** np.random.uniform(.85, 1.2)
    return tuple(torch.from_numpy(np.ascontiguousarray(value))[None] for value in values)


def joint_weak_loss(logits, core, halo, valid, teacher, config):
    valid, core, halo = valid.bool(), core.bool(), halo.bool()
    positive = core & valid
    margin = road_margin(logits)[:, 0]
    road = F.softplus(-margin[positive]).mean() if positive.any() else margin.sum() * 0
    maximum, dominant = teacher.max(1)
    safe = valid & ~halo & (maximum > .9) & (dominant != 1) & (teacher[:, 1] < .01)
    consistency = logits.sum() * 0
    if safe.any():
        kl = F.kl_div(logits.log_softmax(1), teacher.clamp_min(1e-8), reduction="none").sum(1)
        consistency = kl[safe].mean()
    return config["weak_road_weight"] * road + config["teacher_consistency_weight"] * consistency
