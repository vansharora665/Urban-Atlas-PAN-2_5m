"""Optional cartographic generalisation. Raw predictions and accuracy stay unchanged."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.features import sieve, shapes
from rasterio.windows import Window
from rasterio.warp import reproject, transform as transform_points
from scipy.ndimage import gaussian_filter, distance_transform_edt

from .model_roads import atomic_json, result_directory
from .road_graph import connect_short_gaps, filter_components, prune_spurs, skeletonize, trace_graph


VERSION = "usable_map_v3"
LATEST = "usable_map_latest.json"
COLORS = np.array([[31, 119, 180], [250, 249, 241], [225, 181, 70], [165, 125, 94], [71, 139, 93]], np.uint8)
NAMES = ["Water Bodies", "Roads", "Urban Built-up", "Barren / Empty Land", "Vegetation"]
ARTIFACTS = {"map_geotiff_url": "landcover_native.tif", "road_geotiff_url": "roads_native.tif",
             "inferred_geotiff_url": "inferred_links_native.tif", "edits_geotiff_url": "edits_native.tif",
             "map_preview_url": "map_preview.png", "overlay_url": "overlay.png",
             "road_preview_url": "roads_preview.png", "graph_url": "road_graph.json",
             "network_geojson_url": "road_network_wgs84.geojson", "summary_url": "summary.json",
             "vectors_url": "usable_map_vectors.zip"}
SOURCE_NAMES = ("segmentation_labels_georef.tif", "model_roads_mask_georef.tif",
                "model_roads_probability_georef.tif", "cloud_quality_georef.tif", "prediction_summary.json",
                "input_reference.json")


@dataclass(frozen=True)
class MapOptions:
    minimum_region_m2: float = 300.
    smoothing_m: float = 4.
    context_m: float = 9.
    minimum_road_length_m: float = 35.
    maximum_gap_m: float = 14.
    maximum_working_pixels: int = 32_000_000
    uncertainty_fill_policy: str = "narrow_verified_interior_v2"


def remember_input(settings, identifier, source_path):
    """A server-owned reference for distinguishing valid abstentions from no-data."""
    directory = result_directory(settings, identifier)
    source_path = Path(source_path).resolve()
    if not source_path.is_relative_to(settings.upload_dir.resolve()):
        raise ValueError("Input reference must be a local platform upload")
    stat = source_path.stat()
    atomic_json(directory / "input_reference.json", {"path": str(source_path), "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "purpose": "Input validity only; not ground truth"})


def domain_reference(settings, directory, source, summary):
    """Only a verified input can authorise filling an internal low-confidence hole."""
    native = directory / "native_input_stack.vrt"
    candidates = []
    if native.is_file() and not native.is_symlink():
        candidates.append((native, "saved native input stack", False))
    reference = directory / "input_reference.json"
    if reference.is_file() and not reference.is_symlink():
        value = json.loads(reference.read_text())
        path = Path(value["path"])
        if (path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(settings.upload_dir.resolve())
                and path.stat().st_size == value["bytes"] and path.stat().st_mtime_ns == value["mtime_ns"]):
            candidates.append((path, "saved platform upload reference", False))
    image = summary.get("image", {})
    name = image.get("normalization_reference", "")
    if (image.get("normalization_method") == "pixel_verified_parent_scene_original_pixels_v1"
            and image.get("normalization_reference_verified_pixels", 0) >= 4096 and Path(name).name == name):
        for folder in ("JNPA", "Banda"):
            candidate = settings.root_dir.parent / folder / name
            if candidate.is_file() and not candidate.is_symlink():
                candidates.append((candidate, "legacy parent scene verified against saved preview", True))
    for path, reason, compare_preview in candidates:
        with rasterio.open(path) as original:
            if (original.shape, original.crs, original.transform) != (source.shape, source.crs, source.transform):
                continue
        if compare_preview:
            from .image_utils import _read_raster_grayscale, to_uint8
            saved = np.asarray(Image.open(directory / "input_preview.png").convert("RGB"))
            limits = summary.get("image", {})
            low, high = limits.get("normalization_low"), limits.get("normalization_high")
            if low is None or high is None or high <= low:
                continue
            with rasterio.open(path) as original:
                gray, valid = _read_raster_grayscale(original, out_shape=saved.shape[:2])
            normalized = np.clip((gray - low) / (high - low), 0, 1)
            normalized[~valid] = 0
            preview = np.repeat(to_uint8(normalized)[..., None], 3, axis=2)
            if saved.shape != preview.shape or np.abs(saved.astype(float) - preview).mean() > .5:
                continue
        return path, reason
    return None, "No verified original input; original Unknown remains protected"


def source_domain(source, window=None, shape=None):
    options = {"window": window} if window is not None else {}
    if shape is not None:
        options.update(out_shape=shape, resampling=Resampling.nearest)
    value = source.read(1, **options)
    masks = source.read_masks(1, **options)
    return (masks > 0) & np.isfinite(value)


def fill_uncertain_holes(labels, domain, gsd, blocked=None):
    valid = labels < 5
    allowed = domain & ~valid
    if blocked is not None:
        allowed &= ~blocked
    count, components, stats, _ = cv2.connectedComponentsWithStats(allowed.astype(np.uint8), connectivity=4)
    keep = np.zeros(count, bool)
    if count > 1:
        # Never fill outer footprints or more than an 8 m inset. Long narrow
        # abstention chains can be filled too, but only inside verified valid data.
        s = stats[1:]
        keep[1:] = ((s[:, cv2.CC_STAT_LEFT] > 0) & (s[:, cv2.CC_STAT_TOP] > 0)
                    & (s[:, cv2.CC_STAT_LEFT] + s[:, cv2.CC_STAT_WIDTH] < labels.shape[1])
                    & (s[:, cv2.CC_STAT_TOP] + s[:, cv2.CC_STAT_HEIGHT] < labels.shape[0]))
    candidate = allowed & keep[components]
    del components
    if not candidate.any() or not valid.any():
        return labels.copy(), np.zeros_like(valid)
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    candidate &= distance * gsd <= 8.
    result = labels.copy()
    result[candidate] = labels[nearest[0][candidate], nearest[1][candidate]]
    # A context-infilled uncertainty hole must not manufacture a road by nearest neighbour.
    candidate &= result != 1
    result[~valid & ~candidate] = 255
    return result, candidate


def fingerprint(directory):
    records = []
    for name in SOURCE_NAMES:
        path = directory / name
        if path.is_symlink():
            raise ValueError("Linked source artifacts are not supported")
        if path.is_file():
            stat = path.stat()
            records.append({"file": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    value = {"version": VERSION, "options": asdict(MapOptions()), "inputs": records}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), value


def capability(settings, identifier):
    directory = result_directory(settings, identifier)
    path = directory / "segmentation_labels_georef.tif"
    if not path.is_file() or path.is_symlink():
        raise ValueError("This analysis needs a georeferenced five-class raster first")
    summary = json.loads((directory / "prediction_summary.json").read_text())
    classes = {int(row["class_id"]): row["class_name"].lower() for row in summary.get("class_percentages", [])}
    if 1 in classes and "road" not in classes[1]:
        raise ValueError("This older class schema is not the five-class road/land-cover model")
    with rasterio.open(path) as source:
        if not source.crs or not source.crs.is_projected or source.crs.linear_units != "metre":
            raise ValueError("Usable-map processing needs a projected metric CRS. Reproject this analysis to metres first.")
        return {"width": source.width, "height": source.height, "crs": str(source.crs)}


def saved_layer(settings, identifier):
    directory = result_directory(settings, identifier)
    pointer = directory / LATEST
    if not pointer.is_file() or pointer.is_symlink():
        return None
    try:
        value = json.loads(pointer.read_text())
        folder = value["folder"]
        if not folder.startswith(VERSION + "_") or Path(folder).name != folder:
            return None
        home = directory / folder
        if home.is_symlink() or not home.is_dir():
            return None
        current, _ = fingerprint(directory)
        summary_path = home / "summary.json"
        if summary_path.is_symlink():
            return None
        summary = json.loads(summary_path.read_text())
        if summary.get("input_fingerprint") != current or summary.get("version") != VERSION:
            return None
        required = [ARTIFACTS[key] for key in ("map_geotiff_url", "road_geotiff_url", "inferred_geotiff_url", "edits_geotiff_url", "map_preview_url", "road_preview_url", "graph_url")]
        if not all((home / name).is_file() and not (home / name).is_symlink() for name in required):
            return None
        summary["folder"] = folder
        summary["artifacts"] = {key: f"/results/{identifier}/{folder}/{name}" if (home / name).is_file() else None
                                for key, name in ARTIFACTS.items()}
        summary["artifacts"]["viewer_url"] = f"/maps/{identifier}/usable"
        summary["tiles_url"] = f"/api/results/{identifier}/usable-map/tiles/{{layer}}/{{z}}/{{x}}/{{y}}.png?v={current[:12]}"
        return summary
    except (OSError, ValueError, TypeError, KeyError):
        return None


def smooth_landcover(labels, gsd, options):
    valid = np.isin(labels, [0, 1, 2, 3, 4])
    result = np.full(labels.shape, 255, np.uint8)
    # Roads are reinserted as a separate, topology-filtered surface, not majority-voted away.
    for y in range(0, labels.shape[0], 768):
        for x in range(0, labels.shape[1], 768):
            halo = max(4, math.ceil(4 * options.context_m / gsd))
            y0, x0 = max(0, y - halo), max(0, x - halo)
            y1, x1 = min(labels.shape[0], y + 768 + halo), min(labels.shape[1], x + 768 + halo)
            part = labels[y0:y1, x0:x1]
            best = np.full(part.shape, -1, np.float32)
            chosen = np.full(part.shape, 255, np.uint8)
            for class_id in (0, 2, 3, 4):
                evidence = (part == class_id).astype(np.float32)
                score = (.72 * gaussian_filter(evidence, options.smoothing_m / gsd, mode="nearest")
                         + .28 * gaussian_filter(evidence, options.context_m / gsd, mode="nearest")
                         + .10 * evidence)
                replace = (score > best) & (score > .005)
                best[replace], chosen[replace] = score[replace], class_id
            cy, cx = y - y0, x - x0
            h, w = min(768, labels.shape[0] - y), min(768, labels.shape[1] - x)
            result[y:y+h, x:x+w] = chosen[cy:cy+h, cx:cx+w]
    result[~valid] = 255
    mask = (valid & (result != 255)).astype(np.uint8)
    if not mask.any():
        return result
    sieved = sieve(result, size=max(2, math.ceil(options.minimum_region_m2 / gsd**2)), mask=mask, connectivity=4)
    sieved[~valid] = 255
    # GDAL cannot remove tiny islands with no sufficiently large immediate neighbour.
    # Assign those remaining fragments to the nearest established region instead.
    anchors = np.zeros(labels.shape, bool)
    for class_id in (0, 2, 3, 4):
        count, regions, stats, _ = cv2.connectedComponentsWithStats((sieved == class_id).astype(np.uint8), connectivity=4)
        keep = stats[:, cv2.CC_STAT_AREA] * gsd**2 >= options.minimum_region_m2
        keep[0] = False
        anchors |= keep[regions]
        del regions
    fragments = valid & (sieved != 255) & ~anchors
    if fragments.any() and anchors.any():
        nearest = distance_transform_edt(~anchors, return_distances=False, return_indices=True)
        sieved[fragments] = sieved[nearest[0][fragments], nearest[1][fragments]]
    return sieved


def refine_roads(mask, landcover, gsd, probability=None, options=None):
    options = options or MapOptions()
    valid = landcover != 255
    initial = (mask != 0) & valid
    # Keep dot removal separate from broad land-cover smoothing.
    road = filter_components(initial, gsd, options.minimum_road_length_m)
    water = landcover == 0
    road, inferred, gaps = connect_short_gaps(road, valid, water, gsd, probability, options.maximum_gap_m)
    road = filter_components(road, gsd, options.minimum_road_length_m)
    inferred &= road
    skeleton = prune_spurs(skeletonize(road), gsd)
    # A spur-pruned graph and its surface must agree spatially.
    if skeleton.any():
        from scipy.ndimage import distance_transform_edt
        distances, nearest = distance_transform_edt(~skeleton, return_indices=True)
        radius = cv2.distanceTransform(road.astype(np.uint8), cv2.DIST_L2, 5)
        road &= distances <= radius[nearest[0], nearest[1]] + .75
        del distances, nearest, radius
    else:
        road[:] = False
    inferred &= road
    return road, inferred, skeleton, {**gaps, "source_road_cells": int(initial.sum()),
                                    "removed_road_cells": int((initial & ~road).sum()),
                                    "inferred_cells": int(inferred.sum())}


def _profile(source, dtype="uint8", nodata=255):
    return {"driver": "GTiff", "height": source.height, "width": source.width, "count": 1,
            "crs": source.crs, "transform": source.transform, "dtype": dtype, "nodata": nodata,
            "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate", "BIGTIFF": "IF_SAFER"}


def _native_outputs(source, work, landcover, roads, inferred, work_transform, progress,
                    filled_holes=None, domain=None, clouds=None):
    counts = np.zeros(5, np.int64)
    changed = unknown = inferred_count = filled_count = 0
    road_array, inferred_array = roads.astype(np.uint8), inferred.astype(np.uint8)
    fill_array = np.zeros(landcover.shape, np.uint8) if filled_holes is None else filled_holes.astype(np.uint8)
    with ExitStack() as stack:
        outputs = {name: stack.enter_context(rasterio.open(work / name, "w", **_profile(source)))
                   for name in ("landcover_native.tif", "roads_native.tif", "inferred_links_native.tif", "edits_native.tif")}
        for y in range(0, source.height, 1024):
            for x in range(0, source.width, 1024):
                window = Window(x, y, min(1024, source.width - x), min(1024, source.height - y))
                original = source.read(1, window=window)
                valid = (original < 5) & (source.read_masks(1, window=window) > 0)
                shape = original.shape
                converted = []
                rx, ry = source.width / landcover.shape[1], source.height / landcover.shape[0]
                wx0, wy0 = max(0, math.floor(x / rx) - 1), max(0, math.floor(y / ry) - 1)
                wx1, wy1 = min(landcover.shape[1], math.ceil((x + window.width) / rx) + 1), min(landcover.shape[0], math.ceil((y + window.height) / ry) + 1)
                local_transform = work_transform * rasterio.Affine.translation(wx0, wy0)
                for array in (landcover, road_array, inferred_array, fill_array):
                    value = np.full(shape, 255, np.uint8)
                    reproject(array[wy0:wy1, wx0:wx1], value, src_transform=local_transform, src_crs=source.crs,
                              dst_transform=source.window_transform(window), dst_crs=source.crs,
                              src_nodata=255 if array is landcover else None, dst_nodata=255,
                              resampling=Resampling.nearest, num_threads=1)
                    converted.append(value)
                mapped, road, added, fill = converted
                filled = np.zeros(shape, bool)
                if domain is not None:
                    filled = (original == 255) & (fill == 1) & source_domain(domain, window=window)
                    if clouds is not None:
                        filled &= ~np.isin(clouds.read(1, window=window), [1, 2, 255])
                    valid |= filled
                mapped[~valid] = road[~valid] = added[~valid] = 255
                # If no non-road evidence was available, do not invent a background class.
                road[(mapped == 255) | ~valid] = 255
                added[(mapped == 255) | ~valid] = 255
                mapped[road == 1] = 1
                edits = np.zeros(shape, np.uint8)
                edits[valid & (mapped != original)] = 1
                edits[valid & (original == 1) & (mapped != 1)] = 2
                edits[valid & (road == 1) & (original != 1)] = 3
                edits[added == 1] = 4
                edits[filled & (added != 1)] = 5
                edits[~valid | (mapped == 255)] = 255
                for name, value in zip(outputs, (mapped, road, added, edits)):
                    outputs[name].write(value, 1, window=window)
                counts += np.bincount(mapped[mapped < 5], minlength=5)
                changed += int((valid & (mapped != original)).sum())
                unknown += int((mapped == 255).sum())
                inferred_count += int((added == 1).sum())
                filled_count += int((filled & (mapped < 5)).sum())
            progress(.65 + .23 * min(1., (y + 1024) / source.height), "Writing native-grid map, road surfaces and edit audit")
        for output in outputs.values():
            output.update_tags(product=VERSION, cartographic_derivative="true", ground_truth="false",
                               accuracy_measured="false", raw_prediction_unchanged="true")
            levels = [level for level in (2, 4, 8, 16, 32, 64) if min(output.shape) // level >= 32]
            if levels:
                output.build_overviews(levels, Resampling.nearest)
            write_palette(output)
    return counts, {"changed_pixels": changed, "unknown_pixels": unknown, "inferred_road_pixels": inferred_count,
                    "context_filled_low_confidence_pixels": filled_count}


def write_palette(output):
    name = Path(output.name).name
    if name == "landcover_native.tif":
        palette = {i: tuple(map(int, color)) + (255,) for i, color in enumerate(COLORS)}
    elif name == "roads_native.tif":
        palette = {0: (38, 51, 59, 255), 1: (244, 169, 66, 255)}
    elif name == "inferred_links_native.tif":
        palette = {0: (38, 51, 59, 0), 1: (169, 106, 227, 255)}
    else:
        palette = {0: (38, 51, 59, 255), 1: (225, 181, 70, 255), 2: (224, 102, 97, 255),
                   3: (71, 139, 93, 255), 4: (169, 106, 227, 255), 5: (63, 180, 192, 255)}
    palette[255] = (218, 222, 225, 0)
    output.write_colormap(1, palette)


def fragmentation(labels, gsd, minimum_m2):
    total = small = 0
    for class_id in (0, 2, 3, 4):
        count, components, stats, _ = cv2.connectedComponentsWithStats((labels == class_id).astype(np.uint8), connectivity=4)
        total += count - 1
        small += int((stats[1:, cv2.CC_STAT_AREA] * gsd**2 < minimum_m2).sum())
        del components
    return {"nonroad_regions": total, "regions_below_mapping_unit": small,
            "grid": "generalisation working grid; not accuracy"}


def _colors(labels):
    result = np.full((*labels.shape, 3), [218, 222, 225], np.uint8)
    valid = labels < 5
    result[valid] = COLORS[labels[valid]]
    return result


def previews(directory, work, maximum=1800):
    with rasterio.open(work / "landcover_native.tif") as source:
        ratio = min(1., maximum / max(source.shape))
        shape = (max(1, round(source.height * ratio)), max(1, round(source.width * ratio)))
        labels = source.read(1, out_shape=shape, resampling=Resampling.nearest)
    colors = _colors(labels)
    Image.fromarray(colors).save(work / "map_preview.png")
    raw = np.array(Image.open(directory / "input_preview.png").convert("RGB").resize((shape[1], shape[0])))
    Image.fromarray((raw.astype(float) * .35 + colors * .65).astype(np.uint8)).save(work / "overlay.png")
    with rasterio.open(work / "roads_native.tif") as source:
        mask = source.read(1, out_shape=shape, resampling=Resampling.nearest)
    road_colors = np.full_like(raw, [38, 51, 59])
    road_colors[mask == 1], road_colors[mask == 255] = [244, 169, 66], [218, 222, 225]
    Image.fromarray(road_colors).save(work / "roads_preview.png")


def write_network(work, graph, crs):
    graph["crs"] = str(crs)
    atomic_json(work / "road_graph.json", graph)
    features = []
    for edge in graph["edges"]:
        xs, ys = zip(*edge["coordinates"])
        lon, lat = transform_points(crs, "EPSG:4326", xs, ys)
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": list(map(list, zip(lon, lat)))},
                         "properties": {key: value for key, value in edge.items() if key != "coordinates"}})
    atomic_json(work / "road_network_wgs84.geojson", {"type": "FeatureCollection", "features": features,
                "name": "Cartographic road centreline candidates; inferred links flagged; not navigation validated"})


def build(settings, identifier, progress):
    started = time.perf_counter()
    info = capability(settings, identifier)
    cached = saved_layer(settings, identifier)
    if cached:
        return {**cached, "cached_result": True}
    directory = result_directory(settings, identifier)
    signature, provenance = fingerprint(directory)
    destination = directory / (VERSION + "_" + signature[:12])
    if destination.exists():
        raise ValueError("An incomplete derivative directory already exists; preserve it for review before retrying")
    if shutil.disk_usage(directory).free < info["width"] * info["height"] * 2 + 1_500_000_000:
        raise ValueError("Keep at least 1.5 GB spare plus room for native-grid cartographic exports")
    options = MapOptions()
    progress(.02, "Reading existing predictions only; no model or external map is being run")
    with tempfile.TemporaryDirectory(prefix=".usable_map_", dir=directory) as temporary:
        work = Path(temporary)
        with ExitStack() as stack:
            stack.enter_context(rasterio.Env(GDAL_CACHEMAX=128 * 1024**2))
            source = stack.enter_context(rasterio.open(directory / "segmentation_labels_georef.tif"))
            factor = max(1., math.sqrt(source.width * source.height / options.maximum_working_pixels))
            shape = (max(1, math.ceil(source.height / factor)), max(1, math.ceil(source.width / factor)))
            work_transform = source.transform * rasterio.Affine.scale(source.width / shape[1], source.height / shape[0])
            gsd = math.sqrt(abs(work_transform.a * work_transform.e - work_transform.b * work_transform.d))
            labels = source.read(1, out_shape=shape, resampling=Resampling.nearest)
            invalid = source.read_masks(1, out_shape=shape, resampling=Resampling.nearest) == 0
            labels[invalid] = 255
            unexpected = set(np.unique(labels)) - {0, 1, 2, 3, 4, 255}
            if unexpected:
                raise ValueError("Unsupported saved class values: " + repr(unexpected))
            summary = json.loads((directory / "prediction_summary.json").read_text())
            domain_path, domain_reason = domain_reference(settings, directory, source, summary)
            domain = stack.enter_context(rasterio.open(domain_path)) if domain_path else None
            cloud_path = directory / "cloud_quality_georef.tif"
            clouds = stack.enter_context(rasterio.open(cloud_path)) if cloud_path.is_file() else None
            blocked = np.zeros(shape, bool)
            if clouds:
                if (clouds.shape, clouds.crs, clouds.transform) != (source.shape, source.crs, source.transform):
                    raise ValueError("Cloud exclusion grid differs from the saved class raster")
                blocked = np.isin(clouds.read(1, out_shape=shape, resampling=Resampling.nearest), [1, 2, 255])
            filled_holes = np.zeros(shape, bool)
            if domain is not None:
                uncertain = summary.get("image", {}).get("uncertain_model_pixels", 0)
                if uncertain > 0 or clouds is not None:
                    labels, filled_holes = fill_uncertain_holes(labels, source_domain(domain, shape=shape), gsd, blocked)
            progress(.08, "Building coherent class regions with metric smoothing and minimum mapping units")
            landcover = smooth_landcover(labels, gsd, options)
            region_audit = {"before_stage": "after optional confidence-hole fill; before spatial regularisation",
                            "before": fragmentation(labels, gsd, options.minimum_region_m2),
                            "after": fragmentation(landcover, gsd, options.minimum_region_m2)}
            progress(.28, "Removing isolated roads and compact blobs; preserving Unknown barriers")
            road_source = "saved five-class road pixels; no pixel probability raster available"
            probability = None
            mask = labels == 1
            path = directory / "model_roads_mask_georef.tif"
            if path.is_file():
                with rasterio.open(path) as roads:
                    if (roads.shape, roads.crs, roads.transform) != (source.shape, source.crs, source.transform):
                        raise ValueError("Saved specialist road grid does not match the land-cover raster")
                    mask = roads.read(1, out_shape=shape, resampling=Resampling.nearest) == 1
                road_source = "saved model-only CoANet road surface; no external map"
                path = directory / "model_roads_probability_georef.tif"
                if path.is_file():
                    with rasterio.open(path) as scores:
                        if (scores.shape, scores.crs, scores.transform) != (source.shape, source.crs, source.transform):
                            raise ValueError("Road score raster does not match the saved prediction grid")
                        probability = scores.read(1, out_shape=shape, resampling=Resampling.average).astype(np.float32)
            road, inferred, skeleton, audit = refine_roads(mask, landcover, gsd, probability, options)
            del probability
            progress(.49, "Tracing connected centreline candidates; marking inferred links separately")
            graph = trace_graph(skeleton, road, inferred, work_transform, gsd)
            write_network(work, graph, source.crs)
            graph_summary = {"nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
                             "total_length_m": sum(edge["length_m"] for edge in graph["edges"]),
                             "inferred_edges": sum(edge["inferred"] for edge in graph["edges"]),
                             "routing_validated": False, "grade_separation_known": False}
            del graph, skeleton
            counts, pixel_audit = _native_outputs(source, work, landcover, road, inferred, work_transform, progress,
                                                  filled_holes, domain, clouds)
            native = {"width": source.width, "height": source.height, "crs": str(source.crs),
                      "transform": list(source.transform)[:6]}
            working = {"shape": list(shape), "gsd_m": gsd, "transform": list(work_transform)[:6],
                       "note": "Generalised working grid, reprojected to the original output grid; not newly measured native detail"}
        progress(.91, "Preparing previews, full-resolution zoom overviews and change accounting")
        previews(directory, work, settings.preview_max_dimension)
        total = int(counts.sum())
        summary = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                   "input_fingerprint": signature, "provenance": provenance, "native_grid": native,
                   "working_grid": working, "parameters": asdict(options), "road_source": road_source,
                   "class_percentages": [{"class_id": i, "class_name": NAMES[i], "pixel_count": int(count),
                                          "percentage": round(100 * int(count) / max(1, total), 3)} for i, count in enumerate(counts)],
                   "change_audit": pixel_audit, "road_cleanup": audit, "network": graph_summary,
                   "region_coherence": region_audit,
                   "input_validity_reference": domain_reason,
                   "raw_prediction_unchanged": True, "external_maps_used": False, "accuracy_measured": False,
                   "iou": None, "confidence_available": "CoANet surface scores only" if "CoANet" in road_source else "No per-pixel confidence available",
                   "tile_size": 256, "maximum_tile_zoom": max(0, math.ceil(math.log2(max(native["width"], native["height"]) / 256))),
                   "limitations": ["Cartographic regularisation is not improved measured IoU or ground truth.",
                                   "Regions smaller than the minimum mapping unit can be merged away, including real small features.",
                                   "Short inferred connections may be wrong; they are separately downloadable and flagged in vectors.",
                                   "Roof/road semantics, overpasses and occluded pavement cannot be established from geometry alone.",
                                   "Small low-confidence holes may be context-filled only inside a verified input footprint; audit code 5 identifies these inferred classes.",
                                   "Known cloud/no-data pixels remain Unknown. No navigation or planning safety guarantee."],
                   "seconds": time.perf_counter() - started}
        atomic_json(work / "summary.json", summary)
        # Source stat signatures detect an accidental concurrent replacement during processing.
        if fingerprint(directory)[0] != signature:
            raise ValueError("Source artifacts changed during processing; derivative was not published")
        work.rename(destination)
        atomic_json(directory / LATEST, {"folder": destination.name})
    progress(1., "Usable-map layer ready; raw predictions and measured scores are unchanged")
    return saved_layer(settings, identifier)


@lru_cache(maxsize=256)
def _tile(path, layer, z, x, y, maximum_zoom, signature):
    factor = 2 ** (maximum_zoom - z)
    with rasterio.open(path) as source:
        window = Window(x * 256 * factor, y * 256 * factor, 256 * factor, 256 * factor)
        value = source.read(1, window=window, out_shape=(256, 256), boundless=True,
                            fill_value=255, resampling=Resampling.nearest)
    rgba = np.zeros((256, 256, 4), np.uint8)
    rgba[..., :3], rgba[..., 3] = 218, np.where(value == 255, 0, 255)
    if layer in {"classes", "raw"}:
        rgba[..., :3] = _colors(value)
    elif layer == "roads":
        rgba[value == 0, :3], rgba[value == 1, :3] = [38, 51, 59], [244, 169, 66]
    else:
        rgba[value == 0, :3] = [38, 51, 59]
        for code, color in {1: [225, 181, 70], 2: [224, 102, 97], 3: [71, 139, 93], 4: [169, 106, 227], 5: [63, 180, 192]}.items():
            rgba[value == code, :3] = color
    output = io.BytesIO()
    Image.fromarray(rgba).save(output, format="PNG")
    return output.getvalue()


def tile(settings, identifier, layer, z, x, y):
    names = {"classes": "landcover_native.tif", "roads": "roads_native.tif", "edits": "edits_native.tif", "raw": None}
    addon = saved_layer(settings, identifier)
    if addon is None or layer not in names:
        raise ValueError("Build this usable-map layer first")
    maximum = addon["maximum_tile_zoom"]
    if not 0 <= z <= maximum or not 0 <= x < 2**z or not 0 <= y < 2**z:
        raise ValueError("Tile coordinates outside the published raster pyramid")
    directory = result_directory(settings, identifier)
    path = directory / addon["folder"] / names[layer] if names[layer] else directory / "segmentation_labels_georef.tif"
    return _tile(str(path), layer, z, x, y, maximum, addon["input_fingerprint"])


def export_vectors(settings, identifier, progress, addon=None):
    """Export generalised parcels and an explicitly non-navigational line graph."""
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    from shapely import make_valid
    from shapely.geometry import shape, LineString, Point

    addon = addon if addon is not None else saved_layer(settings, identifier)
    if not addon:
        raise ValueError("Build the usable-map layer first")
    home = result_directory(settings, identifier) / addon["folder"]
    destination = home / ARTIFACTS["vectors_url"]
    if destination.is_file() and not destination.is_symlink():
        return {"vectors_url": f"/results/{identifier}/{addon['folder']}/{destination.name}", "cached_result": True}
    with tempfile.TemporaryDirectory(prefix=".usable_vectors_", dir=home) as temporary:
        work = Path(temporary)
        gpkg = work / "usable_map.gpkg"
        rows = {i: [] for i in range(5)}
        totals, areas = np.zeros(5, np.int64), np.zeros(5, float)
        with rasterio.open(home / "landcover_native.tif") as source:
            crs = source.crs

            def flush(class_id):
                if not rows[class_id]:
                    return
                frame = gpd.GeoDataFrame(rows[class_id], geometry="geometry", crs=crs)
                path = work / f"class_{class_id}_{['water', 'roads', 'urban', 'bare_land', 'vegetation'][class_id]}.shp"
                shp_frame = frame.drop(columns="area_m2").assign(area_ha=frame["area_m2"] / 10000.)
                shp_frame.to_file(path, driver="ESRI Shapefile", engine="pyogrio", encoding="UTF-8", append=path.exists())
                layer = "class_" + str(class_id)
                exists = gpkg.exists() and layer in pyogrio.list_layers(gpkg)[:, 0]
                frame.to_file(gpkg, layer=layer, driver="GPKG", engine="pyogrio", append=exists)
                rows[class_id].clear()

            progress(.08, "Polygonising continuous class regions, not per-tile fragments")
            for geometry, value in shapes(rasterio.band(source, 1), connectivity=4, transform=source.transform):
                class_id = int(value)
                if class_id not in rows:
                    continue
                polygon = shape(geometry)
                if not polygon.is_valid:
                    polygon = make_valid(polygon)
                if polygon.geom_type not in {"Polygon", "MultiPolygon"} or polygon.is_empty:
                    raise ValueError("Unverified parcel geometry; no vector ZIP was published")
                rows[class_id].append({"class_id": class_id, "class_name": NAMES[class_id],
                                       "area_m2": polygon.area, "edited": 1, "geometry": polygon})
                totals[class_id] += 1
                areas[class_id] += polygon.area
                if len(rows[class_id]) >= 2048:
                    flush(class_id)
            pixel_area = abs(source.transform.a * source.transform.e - source.transform.b * source.transform.d)
            expected_areas = np.array([row["pixel_count"] for row in addon["class_percentages"]]) * pixel_area
            if not np.allclose(areas, expected_areas, rtol=1e-7, atol=.05):
                raise ValueError("Parcel areas did not preserve the cartographic raster")
            for class_id in rows:
                flush(class_id)
                path = work / f"class_{class_id}_{['water', 'roads', 'urban', 'bare_land', 'vegetation'][class_id]}.shp"
                if not path.exists():
                    empty = gpd.GeoDataFrame({"class_id": pd.Series(dtype="int64"), "class_name": pd.Series(dtype="str"),
                                              "area_m2": pd.Series(dtype="float64"), "edited": pd.Series(dtype="int64")},
                                             geometry=gpd.GeoSeries([], crs=crs))
                    empty.drop(columns="area_m2").assign(area_ha=pd.Series(dtype="float64")).to_file(path, driver="ESRI Shapefile", engine="pyogrio", encoding="UTF-8")
                    empty.to_file(gpkg, layer="class_" + str(class_id), driver="GPKG", engine="pyogrio")
        progress(.77, "Exporting road edges and junction nodes with inferred-link flags")
        graph = json.loads((home / "road_graph.json").read_text())
        edges = [{"edge_id": e["id"], "from_id": e["from_id"], "to_id": e["to_id"], "length_m": e["length_m"],
                  "width_m": e["width_m"], "inferred": int(e["inferred"]), "infer_frac": e["inferred_fraction"],
                  "routing_ok": 0, "geometry": LineString(e["coordinates"])} for e in graph["edges"]]
        nodes = [{"node_id": n["id"], "degree": n["degree"], "geometry": Point(n["x"], n["y"])} for n in graph["nodes"]]
        for layer, records, schema in (
            ("road_network_edges", edges, {"edge_id": "int64", "from_id": "int64", "to_id": "int64", "length_m": "float64", "width_m": "float64", "inferred": "int64", "infer_frac": "float64", "routing_ok": "int64"}),
            ("road_network_nodes", nodes, {"node_id": "int64", "degree": "int64"})):
            frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs) if records else gpd.GeoDataFrame(
                {name: pd.Series(dtype=dtype) for name, dtype in schema.items()}, geometry=gpd.GeoSeries([], crs=crs))
            frame.to_file(work / (layer + ".shp"), driver="ESRI Shapefile", engine="pyogrio", encoding="UTF-8")
            frame.to_file(gpkg, layer=layer, driver="GPKG", engine="pyogrio")
        for path in work.glob("*.shp"):
            info = pyogrio.read_info(path)
            if rasterio.crs.CRS.from_user_input(info["crs"]) != crs:
                raise ValueError("Exported CRS differs from the native grid")
            if not all(path.with_suffix(ext).is_file() for ext in (".shx", ".dbf", ".prj", ".cpg")):
                raise ValueError("Incomplete shapefile sidecars")
            frame = pyogrio.read_dataframe(path)
            if not frame.geometry.is_valid.all():
                raise ValueError("An exported geometry failed round-trip validation")
            if path.name.startswith("class_"):
                class_id = int(path.name.split("_")[1])
                if len(frame) != totals[class_id] or abs(frame.geometry.area.sum() - areas[class_id]) > max(.05, areas[class_id] * 1e-7):
                    raise ValueError("Exported parcel count/area differs from the cleaned raster")
        report = {"status": "PASS", "crs": str(crs), "native_class_area_preserved": True,
                  "class_features": totals.tolist(), "class_area_m2": areas.tolist(),
                  "network_edges": len(edges), "network_nodes": len(nodes), "routing_validated": False}
        atomic_json(work / "VECTOR_VERIFICATION.json", report)
        (work / "README.txt").write_text(
            "CARTOGRAPHIC DERIVATIVE, NOT GROUND TRUTH OR A NEW ACCURACY MEASUREMENT.\n"
            "Open usable_map.gpkg in QGIS for all layers, or load each class shapefile and network edges/nodes.\n"
            "Known cloud/no-data regions are excluded; small real objects may be generalised away.\n"
            "Small valid-input uncertainty holes may be context-filled and are marked code 5 in the audit raster.\n"
            "Parcel boundaries exactly follow the regularised raster, with no independently simplified overlaps/gaps.\n"
            "Shapefiles record area_ha (hectares) to avoid DBF precision overflow; GeoPackage retains area_m2.\n"
            "Road widths are raster-derived estimates. inferred=1 means an edge contains a heuristic gap repair.\n"
            "routing_ok=0: no verified access, one-way, turn or overpass information. NOT for navigation.\n"
            "All original predictions, metrics and optional external-map layers remain separate and unchanged.\n")
        if addon.get('product_kind') == 'structured_zones':
            (work / 'README.txt').write_text((work / 'README.txt').read_text() +
                '\nSTRUCTURED NEIGHBOURHOOD MODEL: region-level pre-argmax neural scores, a learned urban-context head, '
                'and spatial consistency. These dominant-use zones are not pixel-level cover truth or cadastral parcels. '
                'The accepted road surface and graph are unchanged. The four-class context-head candidate was rejected.\n')
        temporary_zip = work / destination.name
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.iterdir()):
                if path != temporary_zip:
                    archive.write(path, path.name)
        temporary_zip.replace(destination)
    progress(1., "GIS package ready: five class layers, road edges and junction nodes")
    return {"vectors_url": f"/results/{identifier}/{addon['folder']}/{destination.name}", **report}
