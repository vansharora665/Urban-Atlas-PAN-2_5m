"""Separate sparse OSM supervision: never accepted by the human-label trainer."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from .data import file_digest


def load_weak_roads(path, reviewed_manifest):
    weak = json.loads(path.read_text())
    if (weak.get("stage") != "assistant_checked_osm_weak_training_only" or
            weak.get("human_reviewed") is not False or weak.get("independent_ground_truth") is not False or
            weak.get("unmapped_pixels_are_negative_labels") is not False or weak.get("test_or_validation_source_used") is not False):
        raise ValueError("Weak supervision must remain explicitly separate from reviewed truth")
    training_sources = {(s["normalization_reference"], s["geographic_group"]) for s in reviewed_manifest["scenes"] if s["split"] == "train"}
    heldout_sources = {s["normalization_reference"] for s in reviewed_manifest["scenes"] if s["split"] in {"validation", "test"}}
    ids = set()
    for tile in weak["tiles"]:
        if (tile["split"] != "train" or tile["human_reviewed"] is not False or not tile["assistant_visual_checked"] or
                (tile["source_image"], tile["geographic_group"]) not in training_sources):
            raise ValueError("Weak tile is not an accepted training-only source")
        if tile["tile_id"] in ids or tile["source_image"] in heldout_sources:
            raise ValueError("Duplicate weak tile or held-out source")
        ids.add(tile["tile_id"])
        if file_digest(tile["array"]) != tile["array_sha256"]:
            raise ValueError("Weak tile hash changed")
        with np.load(tile["array"]) as data:
            image = data["image"]
            core, halo, valid = [data[key].astype(bool) for key in ("road_core", "mapped_road_halo", "valid")]
            if (image.ndim != 3 or image.shape[0] != 4 or image.shape[1:] != core.shape or
                    core.shape != halo.shape or halo.shape != valid.shape or not np.isfinite(image).all() or
                    image.min() < 0 or image.max() > 1 or not core.any() or
                    (core & (~halo | ~valid)).any()):
                raise ValueError("Invalid weak image/core/halo/validity arrays")
    return weak


def weak_sample(tile, augment=True):
    data = np.load(tile["array"])
    image = data["image"].copy()
    masks = [data[key].astype(bool) for key in ("road_core", "mapped_road_halo", "valid")]
    if augment:
        rotation = int(np.random.randint(4))
        image = np.rot90(image, rotation, axes=(-2, -1))
        masks = [np.rot90(mask, rotation) for mask in masks]
        if np.random.rand() < .5:
            image = np.flip(image, axis=-1)
            masks = [np.fliplr(mask) for mask in masks]
        if np.random.rand() < .5:
            image = np.flip(image, axis=-2)
            masks = [np.flipud(mask) for mask in masks]
        image = np.clip(image * np.random.uniform(.85, 1.15) + np.random.uniform(-.06, .06), 0, 1)
        image = image ** np.random.uniform(.85, 1.2)
    values = [image] + masks
    return tuple(torch.from_numpy(np.ascontiguousarray(value))[None] for value in values)


def weak_road_loss(output, core, halo, valid, distillation_weight=.5, residual_penalty=.005):
    margin = output["road"][:, 0]
    core, halo, valid = core.bool(), halo.bool(), valid.bool()
    positive = core & valid
    if not positive.any():
        return margin.sum() * 0
    positive_loss = F.softplus(-margin[positive]).mean()
    # Teacher consistency is not a negative ground-truth label or an OSM absence label.
    teacher = output["baseline_road_probability"][:, 0].detach()
    background = valid & ~halo & (teacher < .01) & (output["baseline_nonroad_confidence"][:, 0] > .9)
    consistency = margin.sum() * 0
    if background.any():
        p = teacher[background].clamp(1e-6, 1 - 1e-6)
        kl = F.binary_cross_entropy_with_logits(margin[background], p, reduction="none")
        kl += p * p.log() + (1 - p) * (1 - p).log()
        consistency = kl.clamp_min(0).mean()
    regularizer = output["road_delta"][:, 0][valid].square().mean()
    return positive_loss + distillation_weight * consistency + residual_penalty * regularizer
