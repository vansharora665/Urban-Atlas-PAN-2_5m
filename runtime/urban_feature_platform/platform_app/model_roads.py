"""Optional frozen CoANet roads: separate from five-class labels and OSM priors."""

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject

from .native06m_inference import is_native_grid
from .saved_results import RESULT_ID


MODE = "coanet_native06m_model_only_roads_v1"
CHECKPOINT = "road_transfer_v5_mainroad_precision/experiments/precision_surface_512_v1/BEST_DEVELOPMENT_WEIGHTS.pt"
CHECKPOINT_SHA256 = "d75cc7608a3c7b49d8b1e5a8c1ec60c8e131ccadd3a617b42f19e3d985f2da5b"
SUMMARY = "model_roads_summary.json"
FILES = {"overlay_url": "model_roads_overlay.png", "mask_preview_url": "model_roads_mask_preview.png",
         "mask_geotiff_url": "model_roads_mask_georef.tif", "probability_geotiff_url": "model_roads_probability_georef.tif",
         "summary_url": SUMMARY, "shapefile_url": "model_roads_shapefile.zip"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def result_directory(settings, identifier):
    if not RESULT_ID.fullmatch(identifier):
        raise ValueError("Invalid result identifier")
    directory = settings.result_dir / identifier
    summary = directory / "prediction_summary.json"
    if directory.is_symlink() or not summary.is_file() or summary.is_symlink():
        raise ValueError("Open a completed analysis first")
    return directory


def capability(settings, identifier):
    directory = result_directory(settings, identifier)
    summary = json.loads((directory / "prediction_summary.json").read_text())
    source_path = directory / "native_input_stack.vrt"
    if summary.get("selected_model", {}).get("model_type") != "native_06m":
        return {"available": False, "reason": "The tested road specialist needs native 0.6 m RGB bands; it is not validated for 2.5 m grayscale PAN."}
    if not source_path.is_file() or source_path.is_symlink():
        return {"available": False, "reason": "Original four-band input is unavailable. Upload the TIFF again to rebuild its verified band stack."}
    try:
        with rasterio.open(source_path) as source:
            if source.count != 4 or not is_native_grid(source):
                raise ValueError("Not a native four-band 0.6 m grid")
            return {"available": True, "reason": "Separate model-only road surface; five-class labels remain unchanged.",
                    "width": source.width, "height": source.height}
    except (OSError, rasterio.errors.RasterioError, ValueError):
        return {"available": False, "reason": "The original native input or companion bands are unavailable. Upload the TIFF again."}


def saved_addon(directory, identifier):
    path = Path(directory) / SUMMARY
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text())
        required = [FILES[key] for key in ("overlay_url", "mask_preview_url", "mask_geotiff_url", "probability_geotiff_url")]
        if (value.get("mode") != MODE or value.get("checkpoint_sha256") != CHECKPOINT_SHA256
                or not all((Path(directory) / name).is_file() and not (Path(directory) / name).is_symlink() for name in required)):
            return None
        value["artifacts"] = {key: f"/results/{identifier}/{name}" if (Path(directory) / name).is_file() else None
                              for key, name in FILES.items()}
        return value
    except (OSError, ValueError, TypeError):
        return None


def validate_limits(limits):
    limits = np.asarray(limits, np.float32)
    if limits.shape != (4, 2) or not np.isfinite(limits).all() or np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError("Invalid fixed four-band normalization limits")
    return limits


class ModelRoadService:
    def __init__(self, settings, registry):
        self.settings, self.registry = settings, registry
        self._model = None
        self._limits = {}
        for path in (settings.root_dir.parent, settings.root_dir.parent / "native_06m_specialist/src"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

    def load(self):
        import torch
        from road_transfer_v3_coanet.model import load_checkpoint
        path = self.settings.root_dir.parent / CHECKPOINT
        if self._model is None:
            if digest(path) != CHECKPOINT_SHA256:
                raise ValueError("Road specialist checkpoint fingerprint differs from the evaluated model")
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
            model, verification = load_checkpoint(path)
            model.to(device).eval().requires_grad_(False)
            self._model = (model, device, verification)
        return self._model

    def normalization(self, directory, summary):
        if summary.get("image", {}).get("normalization_limits") is not None:
            return validate_limits(summary["image"]["normalization_limits"]), "Persisted original scene limits"
        from native06m.data import normalization
        source_path, quality = directory / "native_input_stack.vrt", None
        product = summary.get("image", {}).get("matched_local_product")
        if product:
            record = self.registry.get(summary["selected_model"]["id"])
            manifest = json.loads((self.settings.root_dir / record.summary["source_manifest"]).read_text())
            scene = next((row for row in manifest["scenes"] if row["scene_id"] == product), None)
            if scene is None:
                raise ValueError("Original normalization product is no longer registered")
            source_path, quality = Path(scene["images"][0]), scene.get("quality_mask")
        key = (str(source_path), source_path.stat().st_mtime_ns)
        if key not in self._limits:
            self._limits[key] = validate_limits(normalization(source_path, quality))
        return self._limits[key], "Original parent-scene percentiles; no label-based or test-specific tuning"

    def build(self, identifier, progress):
        import torch
        from road_transfer_v3_coanet.model import prepare_input
        from road_transfer_v5_mainroad_precision.full_scene import pad_native, starts, stream_probabilities, surface_only

        started = time.perf_counter()
        directory = result_directory(self.settings, identifier)
        available = capability(self.settings, identifier)
        if not available["available"]:
            raise ValueError(available["reason"])
        cached = saved_addon(directory, identifier)
        if cached:
            return {**cached, "cached_result": True, "cache_access_seconds": time.perf_counter() - started}
        summary_path = directory / "prediction_summary.json"
        summary = json.loads(summary_path.read_text())
        protected = {name: digest(directory / name) for name in
                     ("prediction_summary.json", "class_percentages.json", "segmentation_labels_georef.tif",
                      "input_preview.png", "marked_preview.png", "segmentation_preview.png")}
        progress(.01, "Loading the frozen road specialist; original land-cover map stays unchanged")
        tick = time.perf_counter()
        cold_load = self._model is None
        model, device, verification = self.load()
        load_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        limits, normalization_source = self.normalization(directory, summary)
        normalization_seconds = time.perf_counter() - tick
        source_path = directory / "native_input_stack.vrt"
        stats = {"valid_pixels": 0, "road_pixels": 0, "unknown_pixels": 0, "cloud_excluded_pixels": 0}
        times = {"read_normalize_seconds": 0., "neural_seconds": 0., "raster_write_seconds": 0.}
        with tempfile.TemporaryDirectory(prefix=".model_roads_", dir=directory) as temporary:
            work = Path(temporary)
            with ExitStack() as stack:
                stack.enter_context(rasterio.Env(GDAL_CACHEMAX=128 * 1024 * 1024))
                source = stack.enter_context(rasterio.open(source_path))
                width, height = source.width, source.height
                source_metadata = {"width": width, "height": height, "crs": str(source.crs), "resolution_m": list(source.res)}
                if shutil.disk_usage(directory).free < width * height * 5 + 2_000_000_000:
                    raise ValueError("Not enough free disk for full-resolution road mask and probabilities; keep at least 2 GB spare")
                patch = min(768, max(32, ((max(width, height) + 31) // 32) * 32))
                stride = max(32, patch * 3 // 4)
                quality_path = directory / "native_quality.vrt"
                quality = stack.enter_context(rasterio.open(quality_path)) if quality_path.is_file() else None
                cloud_path = directory / "cloud_quality_georef.tif"
                clouds = stack.enter_context(rasterio.open(cloud_path)) if cloud_path.is_file() else None
                for mask in (quality, clouds):
                    if mask and (mask.shape, mask.crs, mask.transform) != (source.shape, source.crs, source.transform):
                        raise ValueError("Source/cloud quality grid differs from native imagery")
                profile = {"driver": "GTiff", "width": width, "height": height, "count": 1,
                           "crs": source.crs, "transform": source.transform, "compress": "lzw", "tiled": True,
                           "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
                mask_dst = stack.enter_context(rasterio.open(work / FILES["mask_geotiff_url"], "w", **profile, dtype="uint8", nodata=255))
                prob_dst = stack.enter_context(rasterio.open(work / FILES["probability_geotiff_url"], "w", **profile, dtype="float32", nodata=-1.))

                def read_patch(x, y):
                    tick = time.perf_counter()
                    h, w = min(patch, height - y), min(patch, width - x)
                    window = Window(x, y, w, h)
                    raw = source.read(window=window).astype(np.float32)
                    valid = np.isfinite(raw).all(0) & (source.read_masks(window=window) > 0).all(0) & np.any(raw != 0, axis=0)
                    if quality:
                        valid &= quality.read(1, window=window) == 0
                    normalized = np.clip((raw - limits[:, 0, None, None]) / (limits[:, 1, None, None] - limits[:, 0, None, None]), 0, 1)
                    normalized = np.nan_to_num(normalized, nan=0., posinf=1., neginf=0.)
                    image = prepare_input(pad_native(normalized, patch))
                    padded_valid = np.pad(valid, ((0, patch - h), (0, patch - w)), constant_values=False)
                    times["read_normalize_seconds"] += time.perf_counter() - tick
                    return image, padded_valid

                # Skip unused connectivity heads only after checking the true author forward.
                example, _ = read_patch(max(0, (width - patch) // 2), max(0, (height - patch) // 2))
                check_started = time.perf_counter()
                tensor = torch.from_numpy(example[None]).to(device)
                with torch.inference_mode():
                    expected = model(tensor)[0].cpu().numpy()
                    observed = surface_only(model, tensor).cpu().numpy()
                    repeated = surface_only(model, tensor.repeat(2, 1, 1, 1)).cpu().numpy()[:1] if device.type != "cpu" else observed
                equivalence = {"surface_max_absolute_difference": float(np.abs(expected - observed).max()),
                               "batch_max_absolute_difference": float(np.abs(expected - repeated).max()),
                               "batch_threshold_disagreements": int(((expected >= .1) != (repeated >= .1)).sum())}
                if equivalence["surface_max_absolute_difference"] > 1e-6:
                    raise ValueError("Optimized road surface does not reproduce the evaluated architecture")
                batch_size = 2 if (device.type != "cpu" and equivalence["batch_max_absolute_difference"] <= 1e-5
                                   and equivalence["batch_threshold_disagreements"] == 0) else 1
                equivalence_seconds = time.perf_counter() - check_started
                inference_started = time.perf_counter()
                next_row = 0

                def predict(images):
                    tick = time.perf_counter()
                    with torch.inference_mode():
                        values = surface_only(model, torch.from_numpy(images).to(device))[:, 0].cpu().numpy()
                    times["neural_seconds"] += time.perf_counter() - tick
                    return np.clip(values, 0, 1)

                def emit(y, probability, valid):
                    nonlocal next_row
                    tick = time.perf_counter()
                    if y != next_row:
                        raise AssertionError("Missing or duplicate native output rows")
                    out_window = Window(0, y, width, probability.shape[0])
                    if clouds:
                        excluded = np.isin(clouds.read(1, window=out_window), [1, 2])
                        stats["cloud_excluded_pixels"] += int((valid & excluded).sum())
                        valid &= ~excluded
                    labels = np.where(valid, probability >= .1, 255).astype(np.uint8)
                    mask_dst.write(labels, 1, window=out_window)
                    prob_dst.write(np.where(valid, probability, -1).astype(np.float32), 1, window=out_window)
                    stats["valid_pixels"] += int(valid.sum())
                    stats["road_pixels"] += int((labels == 1).sum())
                    stats["unknown_pixels"] += int((~valid).sum())
                    next_row += labels.shape[0]
                    times["raster_write_seconds"] += time.perf_counter() - tick

                def report(done, total):
                    elapsed = time.perf_counter() - inference_started
                    remaining = elapsed * (total - done) / max(1, done)
                    progress(.06 + .88 * done / total,
                             f"CoANet roads: {done}/{total} patches; elapsed {elapsed:.0f}s; estimated remaining {remaining:.0f}s (estimate)")

                stream_probabilities(width, height, read_patch, predict, emit, window=patch, stride=stride,
                                     batch_size=batch_size, progress=report)
                if next_row != height or stats["valid_pixels"] + stats["unknown_pixels"] != width * height:
                    raise AssertionError("Incomplete native road raster")
                for output in (mask_dst, prob_dst):
                    output.update_tags(mode=MODE, checkpoint_sha256=CHECKPOINT_SHA256, threshold="0.1", osm_used="false",
                                       ground_truth="false", original_landcover_unchanged="true")
                mask_dst.set_band_description(1, "0=not predicted as road; 1=predicted vehicle road; 255=Unknown")
                prob_dst.set_band_description(1, "Road surface score, not calibrated accuracy; -1=Unknown")
                inference_seconds = time.perf_counter() - inference_started
                patches = len(starts(width, patch, stride)) * len(starts(height, patch, stride))
            progress(.96, "Preparing separate road preview and measured runtime")
            tick = time.perf_counter()
            render_preview(source_path, work, limits, self.settings.preview_max_dimension)
            preview_seconds = time.perf_counter() - tick
            for key in ("mask_geotiff_url", "probability_geotiff_url"):
                with rasterio.open(work / FILES[key]) as output, rasterio.open(source_path) as original:
                    if (output.shape, output.crs, output.transform) != (original.shape, original.crs, original.transform):
                        raise AssertionError("Road output lost its original georeferencing")
            if not all(digest(directory / name) == expected for name, expected in protected.items()):
                raise AssertionError("Original land-cover artifacts changed during road inference")
            runtime = {"total_seconds": time.perf_counter() - started, "model_load_seconds": load_seconds,
                       "normalization_seconds": normalization_seconds, "equivalence_check_seconds": equivalence_seconds,
                       "inference_and_raster_seconds": inference_seconds, "preview_seconds": preview_seconds,
                       **times, "cold_model_load": cold_load, "device": str(device), "batch_size": batch_size,
                       "patch_size": patch, "stride": stride, "patches": patches,
                       "scope": "Road addon only, including loading, checks, streaming and exports; excludes original five-class prediction, upload, queue wait and optional shapefiles"}
            value = {"mode": MODE, "result_id": identifier, "created_utc": datetime.now(timezone.utc).isoformat(),
                     "checkpoint_sha256": CHECKPOINT_SHA256, "model": "Frozen CoANet vehicle-road surface specialist",
                     "threshold": .1, "normalization_limits": limits.tolist(), "normalization_source": normalization_source,
                     "image": source_metadata,
                     **stats, "road_coverage_percent": 100 * stats["road_pixels"] / max(1, stats["valid_pixels"]),
                     "coverage_is_not_accuracy": True, "cloud_screen": "Existing likely-cloud/fringe mask only" if clouds else "No cloud mask available; source footprint only",
                     "runtime": runtime, "surface_equivalence": equivalence, "strict_model_verification": verification,
                     "validation": {"local_reused_development_road_iou": .41253207752457677,
                                    "external_loveda_48_pooled_road_iou": .48945966062410445,
                                    "external_urban_road_iou": .5644451886396756, "external_rural_road_iou": .2854797757724171},
                     "warning": "Experimental binary road layer, not five-class accuracy or a complete connected network. Local scores reuse development crops; external RGB benchmark is not an Indian TIFF guarantee. No map priors, forced widths, smoothing or gap filling.",
                     "license": "CoANet GPL-3.0 source; upstream specifies non-commercial research use",
                     "source_url": "https://github.com/mj129/CoANet", "original_artifact_hashes": protected,
                     "original_landcover_unchanged": True, "osm_used": False, "labels_loaded": False,
                     "artifacts": {key: f"/results/{identifier}/{name}" if key != "shapefile_url" else None for key, name in FILES.items()}}
            for name in FILES.values():
                if (work / name).is_file():
                    (work / name).replace(directory / name)
            atomic_json(directory / SUMMARY, value)
        progress(1., f"Model-only roads complete in {runtime['total_seconds']:.1f}s; original classes unchanged")
        return value


def render_preview(source_path, work, limits, maximum):
    with rasterio.open(source_path) as source, rasterio.open(work / FILES["mask_geotiff_url"]) as labels:
        scale = min(1., maximum / max(source.shape))
        shape = (max(1, round(source.height * scale)), max(1, round(source.width * scale)))
        raw = source.read(out_shape=(4, *shape), resampling=Resampling.average).astype(np.float32)
        normalized = np.clip((raw - limits[:, 0, None, None]) / (limits[:, 1, None, None] - limits[:, 0, None, None]), 0, 1)
        rgb = (np.nan_to_num(normalized[[2, 1, 0]].transpose(1, 2, 0)) * 255).astype(np.uint8)
        # Maximum overview preserves thin positive pixels; coverage uses full-resolution counts instead.
        mask = np.full(shape, 255, np.uint8)
        reproject(rasterio.band(labels, 1), mask, src_crs=labels.crs, src_transform=labels.transform,
                  dst_crs=labels.crs,
                  dst_transform=labels.transform * rasterio.Affine.scale(labels.width / shape[1], labels.height / shape[0]),
                  src_nodata=255, dst_nodata=255, resampling=Resampling.max, num_threads=2)
    overlay = rgb.copy()
    roads = mask == 1
    overlay[roads] = (.35 * overlay[roads] + .65 * np.array([255, 110, 20])).astype(np.uint8)
    overlay[mask == 255] = 180
    colors = np.full_like(rgb, [31, 42, 49])
    colors[roads], colors[mask == 255] = [255, 110, 20], [180, 180, 180]
    Image.fromarray(overlay).save(work / FILES["overlay_url"])
    Image.fromarray(colors).save(work / FILES["mask_preview_url"])


def export_shapefile(settings, identifier):
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    from rasterio.features import shapes
    from shapely.geometry import shape

    directory = result_directory(settings, identifier)
    addon = saved_addon(directory, identifier)
    if not addon:
        raise ValueError("Run the model-only road specialist first")
    destination = directory / FILES["shapefile_url"]
    if destination.is_file() and not destination.is_symlink():
        return {"shapefile_url": f"/results/{identifier}/{destination.name}", "cached_result": True}
    count, area = 0, 0.
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".road_vector_", dir=directory) as temporary:
        work = Path(temporary)
        with rasterio.open(directory / FILES["mask_geotiff_url"]) as source:
            if not source.crs or not source.crs.is_projected or source.crs.linear_units != "metre":
                raise ValueError("Road shapefile requires the original projected metric CRS")
            rows = []
            path = work / "model_vehicle_roads.shp"
            def flush():
                if not rows:
                    return
                frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=source.crs)
                frame.to_file(path, driver="ESRI Shapefile", engine="pyogrio", encoding="UTF-8", append=path.exists())
                rows.clear()
            for _, window in source.block_windows(1):
                values = source.read(1, window=window)
                for geometry, value in shapes(values, mask=values == 1, transform=source.window_transform(window)):
                    polygon = shape(geometry)
                    if not polygon.is_valid:
                        raise ValueError("A road polygon is invalid; no unverified shapefile was published")
                    rows.append({"class_id": 1, "class_name": "Predicted vehicle road", "area_m2": polygon.area, "geometry": polygon})
                    count, area = count + 1, area + polygon.area
                    if len(rows) >= 2048:
                        flush()
            flush()
            if not path.exists():
                frame = gpd.GeoDataFrame({"class_id": pd.Series(dtype="int64"), "class_name": pd.Series(dtype="str"),
                                          "area_m2": pd.Series(dtype="float64")}, geometry=gpd.GeoSeries([], crs=source.crs))
                frame.to_file(path, driver="ESRI Shapefile", engine="pyogrio", encoding="UTF-8")
            expected = addon["road_pixels"] * abs(source.transform.a * source.transform.e - source.transform.b * source.transform.d)
            if abs(area - expected) > max(.01, expected * 1e-7):
                raise AssertionError("Road polygons do not preserve predicted native road area")
            expected_crs = source.crs
        check = pyogrio.read_info(path)
        if check["features"] != count or rasterio.crs.CRS.from_string(check["crs"]) != expected_crs:
            raise AssertionError("Road shapefile failed CRS/count round-trip")
        if not all(path.with_suffix(suffix).is_file() for suffix in (".shp", ".shx", ".dbf", ".prj")):
            raise AssertionError("Road shapefile sidecars are incomplete")
        (work / "README.txt").write_text("Model-predicted road surface polygons, not ground truth, measured pavement, centrelines or OSM.\n"
                                         "Native CRS and pixel area retained. Raster-block boundaries may split connected roads into adjacent polygons.\n"
                                         "0.10 threshold; no small-component filtering or forced 7 m buffers. CoANet research-use restrictions apply.\n")
        temporary_zip = work / destination.name
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.iterdir()):
                if path != temporary_zip:
                    archive.write(path, path.name)
        temporary_zip.replace(destination)
    return {"shapefile_url": f"/results/{identifier}/{destination.name}", "features": count,
            "area_m2": area, "seconds": time.perf_counter() - started, "georeferencing_verified": True}
