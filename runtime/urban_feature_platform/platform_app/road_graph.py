"""Pixel-supported road topology for a clearly labelled cartographic derivative."""

import math

import cv2
import numpy as np
from scipy.spatial import cKDTree


OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KERNEL = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], np.uint16)


def _thinning_tables():
    bits = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(bool)
    transitions = ((~bits) & np.roll(bits, -1, axis=1)).sum(1)
    common = (bits.sum(1) >= 2) & (bits.sum(1) <= 6) & (transitions == 1)
    p2, p4, p6, p8 = bits[:, 0], bits[:, 2], bits[:, 4], bits[:, 6]
    return (common & ~(p2 & p4 & p6) & ~(p4 & p6 & p8),
            common & ~(p2 & p4 & p8) & ~(p2 & p6 & p8))


TABLES = _thinning_tables()


def skeletonize(mask):
    """Zhang-Suen thinning; topology is retained, not a newly predicted road."""
    current = np.ascontiguousarray(mask, dtype=np.uint8)
    for _ in range(256):
        changed = 0
        for table in TABLES:
            code = cv2.filter2D(current, cv2.CV_16U, KERNEL, borderType=cv2.BORDER_CONSTANT)
            remove = (current != 0) & table[code]
            changed += int(np.count_nonzero(remove))
            current[remove] = 0
        if not changed:
            return current.astype(bool)
    raise ValueError("Road thinning exceeded its safety bound; no incomplete network was published")


def degree(skeleton):
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    return cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16U, kernel, borderType=cv2.BORDER_CONSTANT).astype(np.uint8)


def filter_components(mask, gsd, minimum_length_m=35., minimum_area_m2=20.):
    """Reject dots and broad compact blobs, while retaining long thin components."""
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    width, height, area = stats[:, 2], stats[:, 3], stats[:, 4]
    diagonal = np.hypot(width, height) * gsd
    elongation = np.maximum(width, height) / np.maximum(1, np.minimum(width, height))
    fill = area / np.maximum(1, width * height)
    compact_blob = (fill > .72) & (elongation < 2.) & (np.minimum(width, height) * gsd > 18.)
    keep = (diagonal >= minimum_length_m) & (area * gsd**2 >= minimum_area_m2) & ~compact_blob
    keep[0] = False
    return keep[labels]


def prune_spurs(skeleton, gsd, maximum_m=10.):
    current = skeleton.copy()
    for _ in range(2):
        degrees = degree(current)
        remove = []
        for y, x in np.argwhere(current & (degrees == 1)):
            path, previous, point = [], None, (int(y), int(x))
            length = 0.
            while True:
                path.append(point)
                py, px = point
                neighbors = [(py + dy, px + dx) for dy, dx in OFFSETS
                             if 0 <= py + dy < current.shape[0] and 0 <= px + dx < current.shape[1]
                             and current[py + dy, px + dx] and (py + dy, px + dx) != previous]
                if len(neighbors) != 1:
                    if len(neighbors) > 1 and length < maximum_m:
                        remove.extend(path[:-1])
                    break
                following = neighbors[0]
                length += math.hypot(following[0] - py, following[1] - px) * gsd
                if length >= maximum_m:
                    break
                previous, point = point, following
        if not remove:
            break
        yy, xx = np.asarray(remove).T
        current[yy, xx] = False
    return current


def _outward(skeleton, endpoint, steps):
    point, previous = endpoint, None
    for _ in range(steps):
        y, x = point
        neighbors = [(y + dy, x + dx) for dy, dx in OFFSETS
                     if 0 <= y + dy < skeleton.shape[0] and 0 <= x + dx < skeleton.shape[1]
                     and skeleton[y + dy, x + dx] and (y + dy, x + dx) != previous]
        if len(neighbors) != 1:
            break
        previous, point = point, neighbors[0]
    vector = np.asarray(endpoint, float) - point
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def connect_short_gaps(surface, valid, water, gsd, probability=None, maximum_gap_m=14.):
    """Only opposing endpoints are connected; Unknown/water barriers are never crossed."""
    skeleton = skeletonize(surface)
    ends = np.argwhere(skeleton & (degree(skeleton) == 1))
    inferred = np.zeros(surface.shape, bool)
    if len(ends) < 2:
        return surface.copy(), inferred, {"candidate_pairs": 0, "accepted_links": 0}
    if len(ends) > 100_000:
        raise ValueError("Road endpoint count exceeds the cartographic safety limit")
    _, components = cv2.connectedComponents(surface.astype(np.uint8), connectivity=8)
    radius = cv2.distanceTransform(surface.astype(np.uint8), cv2.DIST_L2, 5)
    tree = cKDTree(ends)
    pairs = sorted(tree.query_pairs(maximum_gap_m / gsd),
                   key=lambda pair: float(np.linalg.norm(ends[pair[0]] - ends[pair[1]])))
    tangents = [_outward(skeleton, tuple(point), max(2, round(9 / gsd))) for point in ends]
    used, accepted = set(), 0
    result = surface.copy()
    for a, b in pairs:
        if a in used or b in used:
            continue
        p, q = ends[a], ends[b]
        if components[tuple(p)] == components[tuple(q)]:
            continue
        delta = q - p
        distance = np.linalg.norm(delta)
        if distance == 0:
            continue
        direction = delta / distance
        cosine = .82 if probability is not None else .94
        if np.dot(tangents[a], direction) < cosine or np.dot(tangents[b], -direction) < cosine:
            continue
        samples = np.rint(np.linspace(p, q, max(3, int(distance * 2) + 1))).astype(int)
        yy, xx = samples.T
        missing_length = float((~surface[yy, xx]).sum()) * distance * gsd / max(1, len(samples) - 1)
        if probability is None and missing_length > 6.:
            continue
        if not valid[yy, xx].all() or water[yy, xx].any():
            continue
        if probability is not None:
            support = probability[yy, xx]
            if np.any(~np.isfinite(support)) or np.any(support < 0) or np.mean(support) < .04:
                continue
        # A small local corridor is checked too, rather than only its centre pixel.
        half_width = max(1, min(3, round(min(radius[tuple(p)], radius[tuple(q)]))))
        y0, x0 = np.maximum(0, np.minimum(p, q) - half_width - 1)
        y1, x1 = np.minimum(surface.shape, np.maximum(p, q) + half_width + 2)
        corridor = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.line(corridor, (int(p[1] - x0), int(p[0] - y0)),
                 (int(q[1] - x0), int(q[0] - y0)), 1, 2 * half_width - 1)
        addition = corridor.astype(bool) & ~surface[y0:y1, x0:x1]
        if np.any(addition & (~valid[y0:y1, x0:x1] | water[y0:y1, x0:x1])):
            continue
        result[y0:y1, x0:x1] |= addition
        inferred[y0:y1, x0:x1] |= addition
        accepted += 1
        used.update((a, b))
    return result, inferred, {"candidate_pairs": len(pairs), "accepted_links": accepted}


def trace_graph(skeleton, surface, inferred, transform, gsd):
    """Trace every skeleton edge once, including cycles, into a planar GIS graph."""
    degrees = degree(skeleton)
    special = skeleton & (degrees != 2)
    _, node_labels = cv2.connectedComponents(special.astype(np.uint8), connectivity=8)
    nodes, lookup = [], {}
    special_points = np.argwhere(special)
    values = node_labels[special]
    order = np.argsort(values, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(values[order])) + 1) if len(order) else []
    for group in groups:
        label = int(values[group[0]])
        points = special_points[group]
        mean = points.mean(0)
        y, x = points[np.argmin(np.sum((points - mean)**2, axis=1))]
        ident = len(nodes) + 1
        wx, wy = transform * (float(x) + .5, float(y) + .5)
        nodes.append({"id": ident, "x": wx, "y": wy, "degree": 0})
        lookup[int(label)] = ident
    visited = np.zeros(skeleton.shape, np.uint8)
    radius = cv2.distanceTransform(surface.astype(np.uint8), cv2.DIST_L2, 5)
    edges = []

    def neighbors(point):
        y, x = point
        return [(i, (y + dy, x + dx)) for i, (dy, dx) in enumerate(OFFSETS)
                if 0 <= y + dy < skeleton.shape[0] and 0 <= x + dx < skeleton.shape[1]
                and skeleton[y + dy, x + dx]]

    def mark(point, following, direction):
        visited[point] |= 1 << direction
        visited[following] |= 1 << ((direction + 4) % 8)

    def emit(start, direction, following, start_id):
        path, point, previous = [start, following], following, start
        mark(start, following, direction)
        while not node_labels[point]:
            choices = [(i, p) for i, p in neighbors(point) if p != previous and not visited[point] & (1 << i)]
            if not choices:
                break
            i, nxt = choices[0]
            mark(point, nxt, i)
            previous, point = point, nxt
            path.append(point)
            if point == start:
                break
        end_id = lookup.get(int(node_labels[point]), start_id)
        coords = [[nodes[start_id - 1]["x"], nodes[start_id - 1]["y"]]]
        coords.extend([list(transform * (x + .5, y + .5)) for y, x in path[1:-1]])
        coords.append([nodes[end_id - 1]["x"], nodes[end_id - 1]["y"]])
        # Remove consecutive duplicates before length and GIS validation.
        coords = [p for i, p in enumerate(coords) if i == 0 or p != coords[i - 1]]
        if len(coords) < 2:
            return
        length = sum(math.dist(a, b) for a, b in zip(coords, coords[1:]))
        if length < gsd * .5:
            return
        yy, xx = np.asarray(path).T
        inferred_count = int(inferred[yy, xx].sum())
        edge = {"id": len(edges) + 1, "from_id": start_id, "to_id": end_id,
                "length_m": length, "width_m": float(np.median(radius[yy, xx]) * 2 * gsd),
                "inferred": bool(inferred_count), "inferred_fraction": inferred_count / len(path), "coordinates": coords}
        edges.append(edge)
        nodes[start_id - 1]["degree"] += 1
        nodes[end_id - 1]["degree"] += 1

    for y, x in np.argwhere(special):
        point = (int(y), int(x))
        for direction, following in neighbors(point):
            if node_labels[following] == node_labels[point] or visited[point] & (1 << direction):
                continue
            emit(point, direction, following, lookup[int(node_labels[point])])
    # Components that are closed loops have no endpoint/junction node.
    next_label = len(lookup) + 1
    for y, x in np.argwhere(skeleton):
        point = (int(y), int(x))
        choices = [(i, p) for i, p in neighbors(point) if not visited[point] & (1 << i)]
        if not choices or node_labels[point]:
            continue
        ident = len(nodes) + 1
        wx, wy = transform * (x + .5, y + .5)
        nodes.append({"id": ident, "x": wx, "y": wy, "degree": 0})
        node_labels[point] = next_label
        next_label += 1
        lookup[int(node_labels[point])] = ident
        emit(point, choices[0][0], choices[0][1], ident)
    return {"nodes": nodes, "edges": edges, "planar_graph": True,
            "routing_validated": False, "grade_separation_known": False,
            "warning": "Image-derived centreline topology; no one-way, access, overpass or turn information. Not a navigation network."}
