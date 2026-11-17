"""Conservative single-band cloud screening, not cloud removal or ground recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


CLOUD_COLOR = (205, 213, 222)
FRINGE_COLOR = (233, 225, 201)


@dataclass
class CloudMask:
    # 0=not screened out, 1=likely cloud, 2=cloud fringe/uncertain.
    codes: np.ndarray
    source_width: int
    source_height: int
    metadata: dict[str, Any]

    def window(self, x: int, y: int, width: int, height: int,
               out_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Project pixel centres directly; do not expand a large source-sized mask."""
        out_h, out_w = out_shape or (height, width)
        rows = np.floor((y + (np.arange(out_h) + .5) * height / out_h)
                        * self.codes.shape[0] / self.source_height).astype(int)
        cols = np.floor((x + (np.arange(out_w) + .5) * width / out_w)
                        * self.codes.shape[1] / self.source_width).astype(int)
        return self.codes[np.ix_(np.clip(rows, 0, self.codes.shape[0]-1),
                                 np.clip(cols, 0, self.codes.shape[1]-1))]

    @property
    def has_clouds(self) -> bool:
        return bool(np.any(self.codes))


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def detect_cloud_mask(raw: np.ndarray, valid: np.ndarray, *, source_width: int,
                      source_height: int, pixel_size_m: float | None,
                      normalization_stats: dict[str, Any]) -> CloudMask:
    """Require extreme brightness AND broad non-roof-like spatial support.

    The scene is screened on a bounded analysis grid. Bright rectangles, small
    glints and thin lines do not become cloud seeds. Shadows are not inferred
    from darkness: there is no reliable sun/view geometry in these input files.
    """
    raw = np.asarray(raw, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(raw)
    if raw.shape != valid.shape or raw.ndim != 2:
        raise ValueError("Cloud screening requires matching 2-D image and valid mask")
    codes = np.zeros(raw.shape, dtype=np.uint8)
    metadata = {
        "method": "single_band_spatial_cloud_screen_v1",
        "status": "no_cloud_candidates", "analysis_shape": list(raw.shape),
        "cloud_objects": 0, "bright_roof_objects_rejected": 0,
        "excluded_fraction_analysis": 0.0,
        "shadow_masking_applied": False,
        "limitation": "Heuristic screening, not validated cloud truth. Thin clouds and cloud shadows may remain; hidden ground is not reconstructed.",
    }
    result = CloudMask(codes, source_width, source_height, metadata)
    if pixel_size_m is None or not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        metadata["status"] = "skipped_unknown_ground_resolution"
        return result
    if not np.any(valid) or not normalization_stats.get("radiometric_tail_detected"):
        return result
    strong = normalization_stats.get("cloud_threshold")
    high = normalization_stats.get("high")
    if strong is None or high is None or strong <= high:
        return result
    gsd_x = pixel_size_m * source_width / raw.shape[1]
    gsd_y = pixel_size_m * source_height / raw.shape[0]
    gsd = float(max(gsd_x, gsd_y))
    if gsd > 20:
        metadata["status"] = "skipped_analysis_resolution_too_coarse"
        return result
    metadata["analysis_gsd_m"] = gsd
    minimum_area_m2 = 1500.0
    minimum_width_m = 15.0
    fringe_threshold = float(high + .15 * (strong - high))
    metadata.update({"seed_threshold": float(strong), "fringe_threshold": fringe_threshold,
                     "minimum_seed_area_m2": minimum_area_m2,
                     "minimum_seed_width_m": minimum_width_m, "uncertainty_buffer_m": 12.0})
    seed = ((raw >= strong) & valid).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, _ellipse(1))
    n, components, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    accepted = np.zeros_like(seed)
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area * gsd_x * gsd_y < minimum_area_m2:
            continue
        x, y, w, h = stats[idx, :4]
        local = (components[y:y+h, x:x+w] == idx).astype(np.uint8)
        padded = np.pad(local, 1)
        if 2 * float(cv2.distanceTransform(padded, cv2.DIST_L2, 5).max()) * min(gsd_x, gsd_y) < minimum_width_m:
            continue
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        rectangle = cv2.minAreaRect(contour)
        rw, rh = rectangle[1]
        if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > 8:
            continue
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / max(rw * rh, 1)
        polygon = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
        # Reject broad bright warehouses as well as tiny residential roof glints.
        if len(polygon) == 4 and rectangularity >= .88:
            metadata["bright_roof_objects_rejected"] += 1
            continue
        accepted[y:y+h, x:x+w][local.astype(bool)] = 1
        metadata["cloud_objects"] += 1
    if not np.any(accepted):
        return result
    # Grow only connected bright fringes, within 60 m of a vetted cloud seed.
    bright = ((raw >= fringe_threshold) & valid).astype(np.uint8)
    _, bright_components = cv2.connectedComponents(bright, 8)
    connected_ids = np.unique(bright_components[accepted.astype(bool)])
    connected_ids = connected_ids[connected_ids != 0]
    nearby = cv2.dilate(accepted, _ellipse(max(1, int(np.ceil(60 / gsd))))).astype(bool)
    grown = np.isin(bright_components, connected_ids) & nearby & valid
    core = accepted.astype(bool)
    halo = cv2.dilate(grown.astype(np.uint8), _ellipse(max(1, int(np.ceil(12 / gsd))))).astype(bool) & valid
    codes[halo] = 2
    codes[core] = 1
    excluded_fraction = float(np.count_nonzero(codes) / np.count_nonzero(valid))
    if excluded_fraction > .20:
        codes[:] = 0
        metadata["status"] = "review_required_excessive_cloud_candidate_area"
        metadata["candidate_fraction_analysis"] = excluded_fraction
        return result
    metadata["status"] = "applied"
    metadata["excluded_fraction_analysis"] = round(excluded_fraction, 6)
    return result


def color_cloud_mask(codes: np.ndarray, base_rgb: np.ndarray | None = None) -> np.ndarray:
    color = np.zeros((*codes.shape, 3), dtype=np.uint8) if base_rgb is None else base_rgb.copy()
    color[codes == 1] = CLOUD_COLOR
    color[codes == 2] = FRINGE_COLOR
    return color
