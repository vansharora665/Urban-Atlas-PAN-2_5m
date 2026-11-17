"""Metric, evidence-constrained road graph decoding without map overlays."""

from dataclasses import asdict, dataclass
import heapq
import math

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


NEIGHBOURS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


@dataclass(frozen=True)
class NetworkConfig:
    seed_probability: float = .65
    support_probability: float = .30
    minimum_length_m: float = 24.
    preferred_width_m: float = 7.
    maximum_width_m: float = 28.
    maximum_gap_m: float = 9.
    minimum_gap_probability: float = .18
    mean_gap_probability: float = .40
    maximum_detour_ratio: float = 1.35
    minimum_alignment_cosine: float = .75
    maximum_gap_candidates: int = 256

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Network parameters must be finite")
        for name in ("seed_probability", "support_probability", "minimum_gap_probability", "mean_gap_probability"):
            if not 0 < values[name] <= 1:
                raise ValueError("Probability thresholds must lie in (0, 1]")
        if self.seed_probability < self.support_probability:
            raise ValueError("Seeds must be at least as certain as support")
        if min(self.minimum_length_m, self.preferred_width_m, self.maximum_width_m) <= 0:
            raise ValueError("Road sizes must be positive")
        if self.maximum_gap_m < 0 or self.maximum_detour_ratio < 1:
            raise ValueError("Invalid path constraints")
        if not 0 <= self.minimum_alignment_cosine <= 1 or self.maximum_gap_candidates < 0:
            raise ValueError("Invalid endpoint constraints")


def thin(mask):
    """Zhang-Suen thinning, retaining loops and using an explicit zero border."""
    image = np.pad(np.asarray(mask, dtype=bool), 1)
    for _ in range(max(image.shape)):
        changed = False
        for phase in (0, 1):
            p = [image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:], image[2:, 2:],
                 image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2], image[:-2, :-2]]
            number = sum(v.astype(np.uint8) for v in p)
            transitions = sum((~p[i] & p[(i+1) % 8]).astype(np.uint8) for i in range(8))
            if phase == 0:
                corner = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                corner = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = image[1:-1, 1:-1] & (number >= 2) & (number <= 6) & (transitions == 1) & corner
            if remove.any():
                image[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            break
    return image[1:-1, 1:-1].copy()


def adjacency(skeleton):
    pixels = {tuple(map(int, p)) for p in np.argwhere(skeleton)}
    result = {}
    for y, x in pixels:
        adjacent = []
        for dy, dx in NEIGHBOURS:
            point = (y+dy, x+dx)
            if point not in pixels:
                continue
            # Avoid triangular shortcuts around orthogonal turns/junctions.
            if dy and dx and ((y+dy, x) in pixels or (y, x+dx) in pixels):
                continue
            adjacent.append(point)
        result[(y, x)] = sorted(adjacent)
    return result


def graph_from_mask(mask, probability, gsd_m, transform=None, crs=None):
    if not np.isfinite(gsd_m) or gsd_m <= 0:
        raise ValueError("A positive metre grid spacing is required")
    mask = np.asarray(mask, bool)
    probability = np.asarray(probability, np.float32)
    if mask.ndim != 2 or not mask.size or probability.shape != mask.shape or not np.isfinite(probability).all():
        raise ValueError("Road probability and mask must be aligned finite 2D arrays")
    skeleton = thin(mask)
    links = adjacency(skeleton)
    anchors = {point for point, neighbours in links.items() if len(neighbours) != 2}
    components, count = ndimage.label(skeleton, np.ones((3, 3)))
    # A closed ring needs one anchor; otherwise it disappears during tracing.
    for component in range(1, count+1):
        pixels = np.argwhere(components == component)
        if len(pixels) and not any(tuple(p) in anchors for p in pixels):
            anchors.add(tuple(map(int, pixels[0])))
    ordered = sorted(anchors)
    identifiers = {point: i for i, point in enumerate(ordered)}
    widths = 2*ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]*gsd_m

    def coordinate(point):
        y, x = point
        return list(transform*(x+.5, y+.5)) if transform is not None else [(x+.5)*gsd_m, (y+.5)*gsd_m]

    nodes = [{"id": identifiers[p], "row": p[0], "col": p[1], "xy": coordinate(p),
              "degree": len(links[p])} for p in ordered]
    visited, edges = set(), []
    for start in ordered:
        for neighbour in links[start]:
            if frozenset((start, neighbour)) in visited:
                continue
            path = [start, neighbour]
            visited.add(frozenset((start, neighbour)))
            while path[-1] not in anchors:
                options = [p for p in links[path[-1]] if p != path[-2]]
                if not options:
                    break
                following = options[0]
                visited.add(frozenset((path[-1], following)))
                path.append(following)
            end = path[-1]
            if end not in identifiers:
                raise RuntimeError("Skeleton tracing lost a terminal node")
            ys, xs = np.asarray(path).T
            length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))*gsd_m
            edges.append({"id": len(edges), "source": identifiers[start], "target": identifiers[end],
                          "length_m": float(length), "confidence_mean": float(probability[ys, xs].mean()),
                          "estimated_width_m": float(np.median(widths[ys, xs])),
                          "coordinates": [coordinate(p) for p in path]})
    graph = {"schema_version": 1, "crs": str(crs) if crs is not None else None,
             "coordinate_units": "metre", "gsd_m": gsd_m, "nodes": nodes, "edges": edges,
             "total_length_m": float(sum(edge["length_m"] for edge in edges)),
             "not_navigation_ready": True,
             "crossing_policy": "2D image graph; grade separation and turn restrictions are not inferred"}
    return graph, skeleton


def _outward(point, links, distance=8):
    path = [point]
    for _ in range(distance):
        options = [q for q in links[path[-1]] if len(path) < 2 or q != path[-2]]
        if len(options) != 1:
            break
        path.append(options[0])
    direction = np.asarray(point, float)-path[-1]
    norm = np.linalg.norm(direction)
    return direction/norm if norm else None


def _supported_path(start, end, probability, allowed, config, gsd_m, orientation=None):
    margin = max(2, int(np.ceil(config.maximum_gap_m/gsd_m)))
    y0, x0 = np.maximum(np.minimum(start, end)-margin, 0)
    y1, x1 = np.minimum(np.maximum(start, end)+margin+1, probability.shape)
    start, end = (start[0]-y0, start[1]-x0), (end[0]-y0, end[1]-x0)
    local = probability[y0:y1, x0:x1]
    safe = allowed[y0:y1, x0:x1] & (local >= config.minimum_gap_probability)
    if not safe[start] or not safe[end]:
        return None
    costs = {start: 0.}
    parents = {}
    queue = [(0., start)]
    while queue:
        cost, point = heapq.heappop(queue)
        if cost > costs[point]:
            continue
        if point == end:
            break
        for dy, dx in NEIGHBOURS:
            q = (point[0]+dy, point[1]+dx)
            if not (0 <= q[0] < local.shape[0] and 0 <= q[1] < local.shape[1]) or not safe[q]:
                continue
            if dy and dx and (not safe[point[0]+dy, point[1]] or not safe[point[0], point[1]+dx]):
                continue
            orientation_cost = 0.
            if orientation is not None:
                vector = orientation[:, q[0]+y0, q[1]+x0]
                norm = np.linalg.norm(vector)
                if norm > .1:
                    expected = np.array([dx*dx-dy*dy, 2*dx*dy], float)/(dx*dx+dy*dy)
                    orientation_cost = .4*(1-float(expected@(vector/norm)))
            proposed = cost+math.hypot(dy, dx)*(1-math.log(max(float(local[q]), 1e-6))+orientation_cost)
            if proposed < costs.get(q, float("inf")):
                costs[q], parents[q] = proposed, point
                heapq.heappush(queue, (proposed, q))
    if end not in costs:
        return None
    path = [end]
    while path[-1] != start:
        path.append(parents[path[-1]])
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    if length > config.maximum_detour_ratio*math.dist(start, end):
        return None
    ys, xs = np.asarray(path).T
    if local[ys, xs].mean() < config.mean_gap_probability:
        return None
    return [(int(y+y0), int(x+x0)) for y, x in reversed(path)]


def decode_network(probability, allowed, gsd_m=.6, config=NetworkConfig(), transform=None, crs=None,
                   *, centreline_probability=None, orientation=None, width_m=None):
    """Decode only evidence-supported roads; width is a ranking prior, not a buffer."""
    if isinstance(config, dict):
        config = NetworkConfig(**config)
    probability = np.asarray(probability, np.float32)
    allowed = np.asarray(allowed, bool)
    if probability.ndim != 2 or not probability.size or probability.shape != allowed.shape:
        raise ValueError("Expected aligned 2D road evidence and permission mask")
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Road probabilities must be finite and in [0, 1]")
    if not np.isfinite(gsd_m) or gsd_m <= 0:
        raise ValueError("Road decoding needs a positive metre grid")
    for name, value, expected in (("centreline", centreline_probability, probability.shape),
                                  ("orientation", orientation, (2, *probability.shape)),
                                  ("width", width_m, probability.shape)):
        if value is not None and (np.shape(value) != expected or not np.isfinite(value).all()):
            raise ValueError(f"Invalid learned {name} field")
    if centreline_probability is not None and (np.min(centreline_probability) < 0 or np.max(centreline_probability) > 1):
        raise ValueError("Centreline probability must lie in [0, 1]")
    if width_m is not None and np.min(width_m) < 0:
        raise ValueError("Learned widths cannot be negative")
    support = allowed & (probability >= config.support_probability)
    seeds = support & (probability >= config.seed_probability)
    if centreline_probability is not None:
        seeds |= support & (probability >= config.support_probability+.1) & (centreline_probability >= .7)
    grown = ndimage.binary_propagation(seeds, mask=support, structure=np.ones((3, 3)))
    components, count = ndimage.label(grown, np.ones((3, 3)))
    accepted = np.zeros_like(allowed)
    rejected_components = 0
    for index, slices in enumerate(ndimage.find_objects(components), 1):
        if slices is None:
            continue
        piece = components[slices] == index
        centreline = thin(piece)
        length = np.count_nonzero(centreline)*gsd_m
        width = 2*ndimage.distance_transform_edt(np.pad(piece, 1))[1:-1, 1:-1]*gsd_m
        median_width = float(np.median(width[centreline])) if centreline.any() else 0.
        prior_width = median_width
        if width_m is not None and centreline.any():
            estimate = float(np.median(width_m[slices][centreline]))
            if gsd_m <= estimate <= 2*config.maximum_width_m:
                prior_width = estimate
        ys, xs = np.nonzero(piece)
        covariance = np.cov(np.stack((ys, xs))) if len(ys) > 2 else np.eye(2)
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = math.sqrt(max(eigenvalues[-1], 0)/max(eigenvalues[0], .1))
        fill = piece.mean()
        preferred = math.exp(-.5*(math.log(max(prior_width, gsd_m)/config.preferred_width_m)/.85)**2)
        mean_score = float(probability[slices][piece].mean())
        valid_shape = length >= config.minimum_length_m and median_width <= config.maximum_width_m
        valid_shape &= elongation >= 2. or fill < .35
        # Thin streets are less favoured, not widened or declared non-roads.
        valid_shape &= mean_score*(.8+.2*preferred) >= config.support_probability
        if valid_shape:
            accepted[slices] |= piece
        else:
            rejected_components += 1
    skeleton = thin(accepted)
    links = adjacency(skeleton)
    endpoints = sorted(p for p, neighbours in links.items() if len(neighbours) == 1)
    pairs = []
    if len(endpoints) > 1 and config.maximum_gap_m:
        tree = cKDTree(np.asarray(endpoints)*gsd_m)
        pairs = sorted(tree.query_pairs(config.maximum_gap_m),
                       key=lambda pair: math.dist(endpoints[pair[0]], endpoints[pair[1]]))
    component_map, _ = ndimage.label(accepted, np.ones((3, 3)))
    used, gaps = set(), []
    for a, b in pairs[:config.maximum_gap_candidates]:
        if a in used or b in used:
            continue
        start, end = endpoints[a], endpoints[b]
        if component_map[start] == component_map[end]:
            continue
        direction = np.asarray(end, float)-start
        direction /= np.linalg.norm(direction)
        first, second = _outward(start, links), _outward(end, links)
        if first is None or second is None:
            continue
        if first@direction < config.minimum_alignment_cosine or second@(-direction) < config.minimum_alignment_cosine:
            continue
        path = _supported_path(start, end, probability, allowed, config, gsd_m, orientation)
        if path is None:
            continue
        gap_mask = np.zeros_like(accepted)
        ys, xs = np.asarray(path).T
        gap_mask[ys, xs] = True
        # Repair a narrow centreline only. Surface width needs image evidence.
        gap_mask = ndimage.binary_dilation(gap_mask, iterations=1)
        gap_mask &= allowed & (probability >= config.minimum_gap_probability)
        accepted |= gap_mask
        used.update((a, b))
        gaps.append({"start_row_col": start, "end_row_col": end,
                     "length_m": float(sum(math.dist(p, q) for p, q in zip(path, path[1:]))*gsd_m),
                     "mean_probability": float(probability[ys, xs].mean())})
    if np.any(accepted & ~allowed):
        raise AssertionError("Graph crossed a protected pixel")
    graph, skeleton = graph_from_mask(accepted, probability, gsd_m, transform, crs)
    graph.update(config=asdict(config), gaps=gaps, rejected_components=rejected_components,
                 network_is_image_derived=True, map_overlay_used=False,
                 learned_geometry_heads_used=centreline_probability is not None or orientation is not None or width_m is not None,
                 width_prior_is_not_forced_surface_width=True)
    return accepted, skeleton, graph


def road_metrics(prediction, target, gsd_m=.6, tolerance_m=1.2):
    """Exact pixel IoU and a separately named, tolerance-based centreline metric."""
    prediction, target = np.asarray(prediction, bool), np.asarray(target)
    if prediction.shape != target.shape or target.ndim != 2 or not np.isin(target, [0, 1, 255]).all():
        raise ValueError("Use aligned binary road references with Unknown=255")
    if gsd_m <= 0 or not np.isfinite(gsd_m) or tolerance_m < 0:
        raise ValueError("Invalid metric grid or tolerance")
    valid, truth = target != 255, target == 1
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth & valid).sum())
    fn = int((~prediction & truth).sum())
    predicted_skeleton = thin(prediction) & valid
    true_skeleton = thin(truth)
    # Ignore skeletons next to unknown instead of inventing reference connectivity.
    safe = ndimage.distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1]*gsd_m > tolerance_m
    predicted_skeleton &= safe
    true_skeleton &= safe
    near_truth = ndimage.distance_transform_edt(~true_skeleton)*gsd_m <= tolerance_m if true_skeleton.any() else np.zeros_like(valid)
    near_pred = ndimage.distance_transform_edt(~predicted_skeleton)*gsd_m <= tolerance_m if predicted_skeleton.any() else np.zeros_like(valid)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "road_iou": tp/(tp+fp+fn) if tp+fp+fn else None,
            "road_precision": tp/(tp+fp) if tp+fp else None,
            "road_recall": tp/(tp+fn) if tp+fn else None,
            "evaluated_pixels": int(valid.sum()), "reference_road_pixels": int(truth.sum()),
            "centreline_precision": float(near_truth[predicted_skeleton].mean()) if predicted_skeleton.any() else None,
            "centreline_recall": float(near_pred[true_skeleton].mean()) if true_skeleton.any() else None,
            "centreline_tolerance_m": tolerance_m, "apls": None,
            "apls_status": "Requires a reviewed graph including junctions and grade separation; not substituted by a pixel proxy"}
