"""Conservative geometry and probability gates for road/water disagreements."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WaterRecoveryGeometry:
    eligible: np.ndarray
    protected: np.ndarray
    connected_to_core: np.ndarray
    stats: dict[str, int | float | str]


def water_recovery_geometry(
    labels: np.ndarray,
    gsd_m: float,
    *,
    max_road_width_m: float = 22.0,
    water_core_width_m: float = 32.0,
    road_connection_m: float = 18.0,
) -> WaterRecoveryGeometry:
    water = (labels == 0).astype(np.uint8)
    valid = labels != 255
    distance = cv2.distanceTransform(water, cv2.DIST_L2, 5) * gsd_m
    count, components, boxes, _ = cv2.connectedComponentsWithStats(water, 8)
    cores = water.astype(bool) & (distance >= water_core_width_m / 2.0)
    core_ids = np.unique(components[cores])
    protected_ids = np.zeros(count, dtype=bool)
    protected_ids[core_ids] = True
    protected_ids[0] = False
    connected_to_core = protected_ids[components]

    radius = max(1, int(np.ceil(road_connection_m / gsd_m)))
    near_road = cv2.dilate(
        (labels == 1).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    ).astype(bool)
    context_size = max(3, 2 * int(np.ceil(20.0 / gsd_m)) + 1)
    urban_context = cv2.blur(np.isin(labels, (1, 2)).astype(np.float32), (context_size, context_size))
    core_radius = max(1, int(np.ceil(32.0 / gsd_m)))
    core_neighbourhood = cv2.dilate(
        cores.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*core_radius+1, 2*core_radius+1)),
    ).astype(bool)
    protected = water.astype(bool) & (
        core_neighbourhood | (connected_to_core & ((urban_context < 0.50) | (distance > 6.0)))
    )
    possible = water.astype(bool) & (~protected) & (urban_context >= 0.30)
    possible &= (~connected_to_core) | ((urban_context >= 0.55) & (distance <= 6.0))
    count, components, boxes, _ = cv2.connectedComponentsWithStats(possible.astype(np.uint8), 8)
    eligible = np.zeros(labels.shape, dtype=bool)
    accepted_components = 0
    for idx in range(1, count):
        if int(boxes[idx, cv2.CC_STAT_AREA]) < 5:
            continue
        x, y, width, height, area = map(int, boxes[idx])
        region = np.s_[y : y + height, x : x + width]
        component = components[region] == idx
        local_width = 2.0 * distance[region][component]
        if float(np.quantile(local_width, 0.90)) > max_road_width_m:
            continue
        longest_m = max(width, height) * gsd_m
        fill_ratio = area / float(width * height)
        aspect = max(width, height) / float(max(1, min(width, height)))
        if longest_m < 24.0 or (aspect < 2.0 and fill_ratio > 0.45):
            continue
        if not np.any(near_road[region] & component):
            continue
        if float(np.mean(urban_context[region][component])) < 0.30:
            continue
        eligible[region] |= component & (urban_context[region] >= 0.20)
        accepted_components += 1

    eligible &= valid & (~protected)
    return WaterRecoveryGeometry(
        eligible=eligible,
        protected=protected,
        connected_to_core=connected_to_core,
        stats={
            "analysis_gsd_m": round(gsd_m, 4),
            "protected_water_cells": int(protected.sum()),
            "eligible_water_cells": int(eligible.sum()),
            "eligible_water_components": accepted_components,
            "eligible_cells_connected_to_core": int(np.count_nonzero(eligible & connected_to_core)),
            "max_road_width_m": max_road_width_m,
            "water_core_width_m": water_core_width_m,
        },
    )


def select_water_road_corrections(
    labels: np.ndarray,
    geometry: WaterRecoveryGeometry,
    road_probability: np.ndarray,
    specialist_water_probability: np.ndarray,
    base_water_probability: np.ndarray,
    agreement: np.ndarray,
    observations: np.ndarray,
    gsd_m: float,
    *,
    min_road_probability: float = 0.90,
    max_base_water_probability: float = 0.80,
    min_agreement: float = 0.67,
) -> np.ndarray:
    evidence = (
        geometry.eligible
        & (road_probability >= min_road_probability)
        & (specialist_water_probability <= 0.15)
        & (road_probability - specialist_water_probability >= 0.70)
        & (base_water_probability <= max_base_water_probability)
        & (agreement >= min_agreement)
        & (observations >= 2)
    )
    stricter_connected_evidence = (
        (road_probability >= max(0.95, min_road_probability))
        & (base_water_probability <= min(0.65, max_base_water_probability))
        & (specialist_water_probability <= 0.05)
        & (agreement >= 0.85)
    )
    evidence &= (~geometry.connected_to_core) | stricter_connected_evidence
    n, components, boxes, _ = cv2.connectedComponentsWithStats(evidence.astype(np.uint8), 8)
    near_road = cv2.dilate(
        (labels == 1).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    result = np.zeros(labels.shape, dtype=bool)
    for idx in range(1, n):
        x, y, width, height, area = map(int, boxes[idx])
        if area < 4 or max(width, height) * gsd_m < 12.0:
            continue
        region = np.s_[y : y + height, x : x + width]
        component = components[region] == idx
        aspect = max(width, height) / float(max(1, min(width, height)))
        if aspect < 1.8 and area / float(width * height) > 0.50:
            continue
        if not np.any(near_road[region] & component):
            continue
        result[region] |= component
    return result & geometry.eligible & (~geometry.protected) & (labels == 0)


class ProbabilityConsensus:
    """Commutative weighted evidence; processing order cannot grow new road seeds."""

    def __init__(self, shape: tuple[int, int]):
        self.road_sum = np.zeros(shape, dtype=np.float32)
        self.water_sum = np.zeros(shape, dtype=np.float32)
        self.base_water_sum = np.zeros(shape, dtype=np.float32)
        self.weight_sum = np.zeros(shape, dtype=np.float32)
        self.votes = np.zeros(shape, dtype=np.uint16)
        self.observations = np.zeros(shape, dtype=np.uint16)

    def add(
        self,
        region: tuple[slice, slice],
        road: np.ndarray,
        water: np.ndarray,
        base_water: np.ndarray,
        valid: np.ndarray,
        weights: np.ndarray,
        stable: np.ndarray,
    ) -> None:
        weighted_valid = weights * valid
        self.road_sum[region] += road * weighted_valid
        self.water_sum[region] += water * weighted_valid
        self.base_water_sum[region] += base_water * weighted_valid
        self.weight_sum[region] += weighted_valid
        self.votes[region] += (stable & valid).astype(np.uint16)
        self.observations[region] += valid.astype(np.uint16)

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        denominator = np.maximum(self.weight_sum, 1e-7)
        self.road_sum /= denominator
        self.water_sum /= denominator
        self.base_water_sum /= denominator
        agreement = self.votes.astype(np.float32) / np.maximum(self.observations, 1)
        return self.road_sum, self.water_sum, self.base_water_sum, agreement
