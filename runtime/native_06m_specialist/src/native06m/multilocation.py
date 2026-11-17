"""Small-data, spatially blocked multi-location development training.

Existing reviewed labels are immutable. Previously inspected annotations remain
development references, even when withheld from this particular training run.
"""

from collections import defaultdict
from pathlib import Path
import random

import cv2
import numpy as np
import rasterio
from shapely.geometry import box
import torch
from torch.nn import functional as F

from .data import NativePatches
from .joint_refinement import reviewed_joint_loss
from .losses import confusion_metrics, boundaries, topology_loss


SPLITS = {
    "267372781_review_01": "train", "267372781_review_02": "train",
    "267372781_review_03": "test", "267372781_review_04": "validation",
    "267372781_water_candidate_01": "train", "267372781_water_candidate_04": "test",
    "2673727101_review_01": "train", "2673727101_review_02": "train",
    "2673727101_review_03": "validation", "2673727101_review_04": "test",
    "267372791_review_01": "validation", "267372791_review_02": "train",
    "267372791_review_03": "train", "267372791_review_04": "test",
}


def clear_weak_road_cores(raw, limits, core, valid, gsd_m=.6):
    """Do not teach mapped roads beneath obvious bright obscuration.

    This only excludes uncertain weak supervision; it does not classify clouds
    or modify reviewed labels. Bright roofs can also be excluded conservatively.
    """
    limits = np.asarray(limits).reshape(4, 2)
    scaled = (raw.astype(np.float32)-limits[:,0,None,None])/(limits[:,1,None,None]-limits[:,0,None,None])
    bright = (scaled[:3].min(0) > 1.02) & (scaled[:3].mean(0) > 1.1)
    radius = max(1, int(np.ceil(12/gsd_m)))
    uncertain = cv2.dilate(bright.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))).astype(bool)
    clear = np.asarray(core, bool) & np.asarray(valid, bool) & ~uncertain
    return clear, {"raw_mapped_core_pixels": int(np.count_nonzero(core)),
                   "retained_clear_core_pixels": int(clear.sum()),
                   "bright_obscuration_buffer_pixels": int(uncertain.sum()),
                   "method": "conservative_bright_obscuration_weak_supervision_exclusion_only"}


def spatial_manifest(manifest, minimum_buffer_m=150., protocol="three_way"):
    if {s["scene_id"] for s in manifest["scenes"]} != set(SPLITS):
        raise ValueError("Unexpected reviewed inventory: require an explicit new split version")
    if protocol not in {"three_way", "development_8020"}: raise ValueError("Unknown spatial protocol")
    held_out = {"267372781_water_candidate_04", "2673727101_review_01", "267372791_review_04"}
    splits = SPLITS if protocol == "three_way" else {key: "validation" if key in held_out else "train" for key in SPLITS}
    result = {**manifest, "scenes": [{**s, "split": splits[s["scene_id"]]} for s in manifest["scenes"]],
              "split_protocol": "whole_tile_spatial_blocks_within_three_parent_scenes_v4_"+protocol,
              "evaluation_scope": "Previously inspected development references; not an independent blind test"}
    grids = []
    for scene in result["scenes"]:
        if scene.get("label_provenance") != "human_reviewed":
            raise ValueError("This experiment accepts the existing reviewed references only")
        with rasterio.open(scene["images"][0]) as source, rasterio.open(scene["labels"]) as target:
            if (source.shape, source.crs, source.transform) != (target.shape, target.crs, target.transform):
                raise ValueError("Input and labels must have the identical native grid")
            if source.count != 4 or target.nodata != 255 or not source.crs.is_projected:
                raise ValueError("Expected four-band native imagery and Unknown=255 labels")
            if source.crs.linear_units != "metre" or any(abs(r-.6) > .03 for r in source.res):
                raise ValueError("Expected a 0.6 m metric grid")
            grids.append((scene, str(source.crs), box(*source.bounds)))
    distances = []
    for i, (a, crs, shape) in enumerate(grids):
        for b, other_crs, other in grids[i+1:]:
            if crs != other_crs or a["split"] == b["split"]:
                continue
            distance = shape.distance(other)
            if distance < minimum_buffer_m:
                raise ValueError(f"Spatial leakage: {a['scene_id']} and {b['scene_id']} are too close")
            distances.append(distance)
    group_splits = ("train", "validation", "test") if protocol == "three_way" else ("train", "validation")
    groups = {split: sorted({s["geographic_group"] for s in result["scenes"] if s["split"] == split})
              for split in group_splits}
    if any(len(v) != 3 for v in groups.values()):
        raise ValueError("All three source groups must appear in every development split")
    audit = {"minimum_cross_split_distance_m": min(distances), "required_buffer_m": minimum_buffer_m,
             "groups_per_split": groups, "split_counts": {k: sum(s["split"] == k for s in result["scenes"]) for k in groups},
             "random_patch_split": False, "source_labels_modified": False,
             "blind_test": False, "protocol": result["split_protocol"]}
    return result, audit


class CachedReviewed:
    def __init__(self, manifest, split, config):
        source = NativePatches(manifest, split, config)
        self.rows = []
        for index in range(len(source)):
            image, target = source[index]
            scene = source.items[index][0]
            self.rows.append((image.numpy().copy(), target.numpy().copy(), scene))
        self.counts = source.counts
        self.groups = defaultdict(list)
        for index, (_, _, scene) in enumerate(self.rows):
            self.groups[scene["geographic_group"]].append(index)

    def sample(self, group, size, road_focus, rng, photometric_strength="strong", return_weak=False):
        eligible = self.groups[group]
        if road_focus:
            positive = [i for i in eligible if np.count_nonzero(self.rows[i][1] == 1) >= 128]
            eligible = positive or eligible
        image, target, scene = self.rows[rng.choice(eligible)]
        weak = getattr(self, "weak_cores", {}).get(scene.get("scene_id"), np.zeros(target.shape, bool)).copy()
        if size < target.shape[0]:
            candidates = np.argwhere(target == 1 if road_focus and (target == 1).any() else target != 255)
            cy, cx = candidates[rng.randrange(len(candidates))]
            x = int(np.clip(cx-rng.randrange(size), 0, target.shape[1]-size))
            y = int(np.clip(cy-rng.randrange(size), 0, target.shape[0]-size))
            image, target = image[:, y:y+size, x:x+size], target[y:y+size, x:x+size]
            weak = weak[y:y+size, x:x+size]
        image, target = image.copy(), target.copy()
        k = rng.randrange(4)
        image, target = np.rot90(image, k, axes=(1, 2)), np.rot90(target, k)
        weak = np.rot90(weak, k)
        for axis in (0, 1):
            if rng.random() < .5:
                image, target = np.flip(image, axis+1), np.flip(target, axis)
                weak = np.flip(weak, axis)
        mild = photometric_strength == "mild"
        gain = rng.uniform(.9, 1.1) if mild else rng.uniform(.75, 1.25)
        offset = rng.uniform(-.025, .025) if mild else rng.uniform(-.07, .07)
        gamma = rng.uniform(.9, 1.1) if mild else rng.uniform(.7, 1.5)
        image = np.clip(image*gain+offset, 0, 1)**gamma
        if rng.random() < (.1 if mild else .25):
            image = np.stack([cv2.GaussianBlur(v, (3, 3), rng.uniform(.2, .7)) for v in image])
        if rng.random() < (.1 if mild else .25):
            generator = np.random.default_rng(rng.randrange(2**32))
            image = np.clip(image+generator.normal(0, .008, image.shape), 0, 1)
        result = (torch.from_numpy(np.ascontiguousarray(image)).float(), torch.from_numpy(np.ascontiguousarray(target)).long())
        return (*result, torch.from_numpy(np.ascontiguousarray(weak))) if return_weak else result


def joint_loss(output, target):
    config = {"class_balanced_ce_fraction": .35, "dice_weight": .5,
              "hard_negative_ratio": 4, "semantic_road_weight": .15, "road_aux_weight": .15}
    loss = reviewed_joint_loss(output, target, config)
    valid = target != 255
    if not valid.any():
        return loss
    edge, safe = boundaries(target)
    if safe.any():
        loss = loss+.08*F.binary_cross_entropy_with_logits(output["boundary"][:, 0][safe], edge[safe])
    probability = output["semantic"].softmax(1)[:, 1:2]
    return loss+.15*topology_loss(probability, target == 1, valid)


def evaluate_cached(model, data, device):
    matrices = defaultdict(lambda: np.zeros((5, 5), np.int64))
    predictions = {}
    model.eval()
    with torch.inference_mode():
        for image, target, scene in data.rows:
            prediction = model(torch.from_numpy(image)[None].to(device))["semantic"].argmax(1)[0].cpu().numpy()
            valid = target != 255
            matrices[scene["geographic_group"]] += np.bincount(
                target[valid].astype(np.int64)*5+prediction[valid], minlength=25).reshape(5, 5)
            predictions[scene["scene_id"]] = prediction.astype(np.uint8)
    per_group = {group: confusion_metrics(matrix) for group, matrix in matrices.items()}
    for group, metric in per_group.items():
        present = np.flatnonzero(matrices[group].sum(1) > 0).tolist()
        metric["reference_class_pixels"] = matrices[group].sum(1).tolist()
        metric["classes_with_reviewed_reference"] = present
        metric["observed_class_mean_iou"] = float(np.mean([metric["class_iou"][i] for i in present]))
        metric["full_five_class_reference_coverage"] = len(present) == 5
    total = confusion_metrics(sum(matrices.values()))
    total["per_location"] = per_group
    total["geographic_macro_mean_iou"] = float(np.mean([v["mean_iou"] for v in per_group.values()]))
    total["road_iou_mean_where_reference_exists"] = float(np.mean([
        v["class_iou"][1] for v in per_group.values() if v["class_recall"][1] is not None]))
    return total, predictions


def development_guard(baseline, candidate):
    failures = []
    if candidate["class_iou"][1] < baseline["class_iou"][1]+.01:
        failures.append("Road IoU gain below one percentage point")
    if candidate["class_precision"][1] is None or candidate["class_precision"][1] < (baseline["class_precision"][1] or 0):
        failures.append("Road precision regressed")
    if candidate["class_recall"][1] is None or candidate["class_recall"][1] < (baseline["class_recall"][1] or 0):
        failures.append("Road recall regressed")
    if candidate["mean_iou"] < baseline["mean_iou"]-.01:
        failures.append("Overall mean IoU dropped over one percentage point")
    for group, old in baseline["per_location"].items():
        new = candidate["per_location"][group]
        if new["pixel_accuracy"] < old["pixel_accuracy"]-.03:
            failures.append(f"{group}: accuracy dropped over three percentage points")
    return {"passed": not failures, "failures": failures}
