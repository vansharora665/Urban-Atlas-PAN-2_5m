"""Conservative native-grid obscuration exclusion and model-supported road recovery."""

import cv2
import numpy as np

from .cloud_mask import CloudMask


ROAD_RECIPES = {
    "conservative": {"minimum_road_probability": .08, "road_boost": 4., "maximum_width_m": 9.,
                     "minimum_length_m": 24., "minimum_pair_probability": .78},
    "balanced": {"minimum_road_probability": .05, "road_boost": 6., "maximum_width_m": 12.,
                 "minimum_length_m": 24., "minimum_pair_probability": .74},
    "strong_supported": {"minimum_road_probability": .04, "road_boost": 8., "maximum_width_m": 12.,
                         "minimum_length_m": 30., "minimum_pair_probability": .72},
}

# Physical thresholds, not a requested road-area quota. Long/thin components survive.
ROAD_CLEANUP_RECIPES = {
    "tiny_compact": {"maximum_area_m2": 4., "maximum_extent_m": 4.2},
    "small_compact": {"maximum_area_m2": 9., "maximum_extent_m": 6.},
    "compact": {"maximum_area_m2": 18., "maximum_extent_m": 9.},
}


def ellipse(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))


def detect_fourband_clouds(raw, valid, limits, *, source_width, source_height, pixel_size_m):
    """Display-normalized spectral/spatial screening, not calibrated Fmask or cloud truth."""
    raw = np.asarray(raw, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(raw).all(0)
    if raw.ndim != 3 or raw.shape[0] != 4 or valid.shape != raw.shape[1:]:
        raise ValueError("Cloud screening requires B/G/R/NIR and an aligned validity mask")
    limits = np.asarray(limits).reshape(4, 2)
    if not np.isfinite(limits).all() or np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError("Cloud screening requires valid parent-scene normalization limits")
    scaled = (raw-limits[:, 0, None, None])/(limits[:, 1, None, None]-limits[:, 0, None, None])
    vis = scaled[:3]
    brightness, minimum = vis.mean(0), vis.min(0)
    spread = vis.max(0)-minimum
    gsd_x, gsd_y = pixel_size_m*source_width/raw.shape[2], pixel_size_m*source_height/raw.shape[1]
    gsd = max(gsd_x, gsd_y)
    codes = np.zeros(valid.shape, np.uint8)
    metadata = {"method": "fourband_brightness_spatial_obscuration_v1", "status": "no_cloud_candidates",
                "cloud_objects": 0, "bright_roof_objects_rejected": 0, "analysis_shape": list(valid.shape),
                "candidate_shapes": [],
                "analysis_gsd_m": gsd, "shadow_masking_applied": False,
                "limitation": "Unvalidated conservative cloud/uncertainty screen. Thin haze and cloud shadows may remain. No hidden ground is reconstructed; normalized bands are not reflectance."}
    result = CloudMask(codes, source_width, source_height, metadata)
    if not valid.any():
        return result
    # Coarse full-scene samples cannot distinguish tiny irregular roof/glint shapes reliably.
    minimum_seed_area = max(900., 256*gsd_x*gsd_y)
    metadata["minimum_seed_area_m2"] = minimum_seed_area
    seed = (valid & (minimum > 1.02) & (brightness > 1.10)
            & (spread < .55*np.maximum(brightness, 1e-6)) & (scaled[3] > .75))
    seed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_OPEN, ellipse(max(1, int(round(3/gsd)))))
    count, components, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    accepted = np.zeros_like(seed)
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if area*gsd_x*gsd_y < minimum_seed_area:
            continue
        local = (components[y:y+height, x:x+width] == index).astype(np.uint8)
        distance = cv2.distanceTransform(np.pad(local, 1), cv2.DIST_L2, 5)
        if 2*distance.max()*min(gsd_x, gsd_y) < 18:
            continue
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        sides = cv2.minAreaRect(contour)[1]
        if min(sides) <= 0 or max(sides)/min(sides) > 8:
            continue
        rectangularity = cv2.contourArea(contour)/max(1., sides[0]*sides[1])
        corners = cv2.approxPolyDP(contour, .025*cv2.arcLength(contour, True), True)
        if len(corners) == 4 and rectangularity > .87:
            metadata["bright_roof_objects_rejected"] += 1
            continue
        accepted[y:y+height, x:x+width][local.astype(bool)] = 1
        metadata["cloud_objects"] += 1
        metadata["candidate_shapes"].append({"box_analysis": [x, y, width, height],
                                               "area_m2": area*gsd_x*gsd_y,
                                               "rectangularity": float(rectangularity), "polygon_corners": len(corners)})
    if not accepted.any():
        return result
    bright = valid & (minimum > .72) & (brightness > .85) & (scaled[3] > .50)
    _, groups = cv2.connectedComponents(bright.astype(np.uint8), 8)
    group_ids = np.unique(groups[accepted.astype(bool)])
    group_ids = group_ids[group_ids != 0]
    nearby = cv2.dilate(accepted, ellipse(max(1, int(np.ceil(90/gsd))))).astype(bool)
    grown = np.isin(groups, group_ids) & nearby & valid
    halo = cv2.dilate(grown.astype(np.uint8), ellipse(max(1, int(np.ceil(12/gsd))))).astype(bool) & valid
    codes[halo] = 2
    codes[accepted.astype(bool)] = 1
    metadata.update(status="applied", excluded_fraction_analysis=float(np.count_nonzero(codes)/valid.sum()),
                    uncertainty_buffer_m=12., classification_value=255)
    return result


def road_evidence(probabilities):
    """Mirror on-disk quantization used by large-scene inference."""
    probability = np.asarray(probabilities)
    quantized = np.round(np.stack((probability[1], probability[3], probability[[0, 2, 4]].max(0)))*255).clip(0, 255).astype(np.uint8)
    nonroad_ids = np.array([0, 2, 3, 4], np.uint8)
    alternative = nonroad_ids[probability[nonroad_ids].argmax(0)]
    return np.concatenate((quantized, alternative[None]))


def recover_narrow_roads(labels, evidence, valid, recipe, gsd_m=.6):
    """Only Bare -> Road; never classify a line as a road without semantic support."""
    labels = np.asarray(labels)
    evidence = np.asarray(evidence)
    valid = np.asarray(valid, dtype=bool) & (labels != 255)
    if evidence.shape not in ((3, *labels.shape), (4, *labels.shape)) or valid.shape != labels.shape:
        raise ValueError("Road evidence, labels and validity grid differ")
    config = ROAD_RECIPES[recipe] if isinstance(recipe, str) else recipe
    road, bare, other = evidence[:3].astype(np.float32)/255.
    eligible = (valid & np.isin(labels, [1, 3]) & (road >= config["minimum_road_probability"])
                & (road >= other) & (road+bare >= config["minimum_pair_probability"]))
    eligible |= valid & (labels == 1) & (road >= .35)
    # Close tiny gaps only in the geometry test; never add these gap pixels to the output.
    geometry = cv2.morphologyEx(eligible.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(1))
    geometry[~valid] = 0
    count, components, stats, _ = cv2.connectedComponentsWithStats(geometry, 8)
    recovered = np.zeros(labels.shape, bool)
    accepted_components = 0
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if max(width, height)*gsd_m < config["minimum_length_m"] or area < 20:
            continue
        local = components[y:y+height, x:x+width] == index
        padded = np.pad(local.astype(np.uint8), 1)
        diameter = 2*float(cv2.distanceTransform(padded, cv2.DIST_L2, 5).max())*gsd_m
        length = area*gsd_m*gsd_m/max(diameter, gsd_m)
        if diameter > config["maximum_width_m"] or length < config["minimum_length_m"] or length/max(diameter, gsd_m) < 3:
            continue
        local_road = road[y:y+height, x:x+width][local]
        # An isolated weak brown field trace cannot become a road just because it is thin.
        anchored = np.any(labels[y:y+height, x:x+width][local] == 1)
        if not anchored and (local_road.mean() < .12 or local_road.max() < .25):
            continue
        change = (local & eligible[y:y+height, x:x+width] & (labels[y:y+height, x:x+width] == 3)
                  & (road[y:y+height, x:x+width]*config["road_boost"] > bare[y:y+height, x:x+width]))
        if change.any():
            recovered[y:y+height, x:x+width] |= change
            accepted_components += 1
    output = labels.copy()
    output[recovered] = 1
    return output, {"bare_to_road_pixels": int(recovered.sum()), "accepted_narrow_components": accepted_components,
                    "road_recipe": recipe if isinstance(recipe, str) else "custom",
                    "only_bare_to_road_changes": True}


def remove_road_specks(labels, evidence, valid, recipe, gsd_m=.6):
    """Reject isolated compact road blobs; never open/erode or blindly connect roads.

    The fourth evidence band identifies the actual best non-road class. A weak
    alternative is Unknown, not a fabricated neighbouring land-cover label.
    Components touching a window/Unknown edge or an established road survive.
    """
    labels, evidence = np.asarray(labels), np.asarray(evidence)
    valid = np.asarray(valid, dtype=bool) & (labels != 255)
    if evidence.shape != (4, *labels.shape) or valid.shape != labels.shape:
        raise ValueError("Road cleanup needs four-band learned evidence including the non-road alternative")
    if not np.isfinite(gsd_m) or gsd_m <= 0:
        raise ValueError("Road cleanup needs a positive metric pixel size")
    config = ROAD_CLEANUP_RECIPES[recipe] if isinstance(recipe, str) else recipe
    road_mask = valid & (labels == 1)
    count, components, stats, centroids = cv2.connectedComponentsWithStats(road_mask.astype(np.uint8), 8)
    output = labels.copy()
    empty = {"road_speck_pixels_removed": 0, "road_speck_pixels_reassigned": 0,
             "road_speck_unknown_pixels": 0, "road_speck_components_removed": 0,
             "road_cleanup_recipe": recipe if isinstance(recipe, str) else "custom"}
    if count <= 1:
        return output, empty
    width, height, area = stats[:, 2], stats[:, 3], stats[:, 4]
    candidate = ((area*gsd_m**2 <= config["maximum_area_m2"])
                 & (np.maximum(width, height)*gsd_m <= config["maximum_extent_m"]))
    candidate[0] = False
    if not candidate.any():
        return output, empty
    # Vectorized central moments avoid a Python loop for millions of one-pixel blobs.
    ys, xs = np.nonzero(road_mask)
    ids = components[ys, xs]
    dx, dy = xs-centroids[ids, 0], ys-centroids[ids, 1]
    xx = np.bincount(ids, weights=dx*dx, minlength=count)/np.maximum(area, 1)
    yy = np.bincount(ids, weights=dy*dy, minlength=count)/np.maximum(area, 1)
    xy = np.bincount(ids, weights=dx*dy, minlength=count)/np.maximum(area, 1)
    delta = np.sqrt(np.maximum(0., (xx-yy)**2+4*xy**2))
    # Include the finite area of a pixel, so a singleton is not infinitely thin.
    elongation_squared = ((xx+yy+delta)/2+1/12)/((xx+yy-delta)/2+1/12)
    linear = (elongation_squared >= 9.) & (np.maximum(width, height) >= 3)
    candidate[linear] = False
    edges = np.unique(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])))
    candidate[edges] = False
    unknown_edge = cv2.dilate((~valid).astype(np.uint8), ellipse(1)).astype(bool)
    candidate[np.unique(components[unknown_edge & road_mask])] = False
    established = road_mask & ~candidate[components]
    nearby = cv2.dilate(established.astype(np.uint8), ellipse(max(1, round(1.2/gsd_m)))).astype(bool)
    candidate[np.unique(components[nearby & road_mask])] = False
    mean_road = np.bincount(ids, weights=evidence[0, ys, xs]/255., minlength=count)/np.maximum(area, 1)
    # A short component can be a real, interrupted road: retain strong evidence.
    candidate[mean_road >= .60] = False
    rejected = road_mask & candidate[components]
    if not rejected.any():
        return output, empty
    alternative = evidence[3]
    if not np.isin(alternative[rejected], [0, 2, 3, 4]).all():
        raise ValueError("Invalid learned non-road alternative class")
    alternate_probability = np.maximum(evidence[1], evidence[2]).astype(np.float32)/255.
    supported = rejected & (alternate_probability >= .20)
    output[supported] = alternative[supported]
    output[rejected & ~supported] = 255
    return output, {**empty, "road_speck_pixels_removed": int(rejected.sum()),
                    "road_speck_pixels_reassigned": int(supported.sum()),
                    "road_speck_unknown_pixels": int(np.count_nonzero(rejected & ~supported)),
                    "road_speck_components_removed": int(candidate.sum())}


def bridge_short_road_gaps(labels, evidence, valid, gsd_m=.6,maximum_gap_m=1.2):
    """Fill only bounded metric gaps between aligned, model-supported road runs.

    No nearest-neighbour or long-range line drawing; Water, Vegetation and
    Unknown cannot become roads. Endpoints come from the original mask only.
    """
    labels, evidence = np.asarray(labels), np.asarray(evidence)
    valid = np.asarray(valid, dtype=bool) & (labels != 255)
    if evidence.shape not in ((3, *labels.shape), (4, *labels.shape)) or valid.shape != labels.shape:
        raise ValueError("Gap repair needs aligned learned road evidence")
    if not np.isfinite(gsd_m) or gsd_m <= 0:
        raise ValueError("Gap repair needs a positive metric pixel size")
    if not np.isfinite(maximum_gap_m) or not 0 <= maximum_gap_m <= 6:
        raise ValueError("Gap repair needs a bounded metric gap between 0 and 6 m")
    roads = valid & (labels == 1)
    probability = evidence[0].astype(np.float32)/255.
    alternative = np.maximum(evidence[1], evidence[2]).astype(np.float32)/255.
    eligible = valid & np.isin(labels, [2, 3]) & (probability >= .20) & (probability >= .4*alternative)
    repaired = np.zeros(labels.shape, bool)
    if not roads.any() or not eligible.any():
        return labels.copy(), {"road_gap_pixels_added": 0, "maximum_gap_m": maximum_gap_m}

    def shift(array, dy, dx):
        result = np.zeros_like(array)
        if abs(dy) >= array.shape[0] or abs(dx) >= array.shape[1]:
            return result
        sy = slice(max(0, -dy), min(array.shape[0], array.shape[0]-dy))
        sx = slice(max(0, -dx), min(array.shape[1], array.shape[1]-dx))
        ty = slice(max(0, dy), min(array.shape[0], array.shape[0]+dy))
        tx = slice(max(0, dx), min(array.shape[1], array.shape[1]+dx))
        result[ty, tx] = array[sy, sx]
        return result

    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        maximum_gap = int(np.floor(maximum_gap_m/(gsd_m*np.hypot(dy, dx))+1e-6))
        for gap in range(1, maximum_gap+1):
            start = np.ones(labels.shape, bool)
            for run in range(3):
                start &= shift(roads, -run*dy, -run*dx)
                start &= shift(roads, (gap+1+run)*dy, (gap+1+run)*dx)
            for offset in range(1, gap+1):
                start &= shift(eligible, offset*dy, offset*dx)
            for offset in range(1, gap+1):
                repaired |= shift(start, -offset*dy, -offset*dx)
    output = labels.copy()
    output[repaired] = 1
    return output, {"road_gap_pixels_added": int(repaired.sum()), "maximum_gap_m": maximum_gap_m}
