"""Native four-band testing adapter, isolated from legacy PAN corrections."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import torch

from .cloud_mask import color_cloud_mask
from .native_refinement_io import refine_native_raster, screen_reference_clouds


NATIVE_STAGE = "limited_geography_joint_fiveclass_development_pilot"
BAND_NAMES = ["Blue", "Green", "Red", "NearInfrared"]


def is_native_grid(source):
    return bool(source.crs and source.crs.is_projected and source.crs.linear_units == "metre"
                and all(abs(value - .6) <= .03 for value in source.res)
                and not source.transform.b and not source.transform.d
                and source.transform.a > 0 and source.transform.e < 0)


def source_window(uploaded, parent):
    if uploaded.crs != parent.crs or not np.allclose(uploaded.res, parent.res, rtol=0, atol=1e-6):
        return None
    x, y = (~parent.transform) * (uploaded.transform.c, uploaded.transform.f)
    xi, yi = round(x), round(y)
    if abs(x-xi) > 1e-3 or abs(y-yi) > 1e-3:
        return None
    if xi < 0 or yi < 0 or xi + uploaded.width > parent.width or yi + uploaded.height > parent.height:
        return None
    return Window(xi, yi, uploaded.width, uploaded.height)


def matching_bands(uploaded, parent, window):
    samples, originals = [], []
    size = min(12, uploaded.width, uploaded.height)
    for yf in (.1, .5, .9):
        for xf in (.1, .5, .9):
            x = round((uploaded.width-size)*xf)
            y = round((uploaded.height-size)*yf)
            samples.append(uploaded.read(window=Window(x, y, size, size)).reshape(uploaded.count, -1))
            originals.append(parent.read(window=Window(window.col_off+x, window.row_off+y, size, size)).reshape(4, -1))
    samples, originals = np.concatenate(samples, 1), np.concatenate(originals, 1)
    if np.count_nonzero(np.isfinite(samples) & (samples != 0)) < 32 or np.std(samples) == 0:
        return None
    if uploaded.count == 4:
        return [1, 2, 3, 4] if np.array_equal(samples, originals) else None
    for band in range(4):
        if np.array_equal(samples[0], originals[band]):
            return [band+1]
    return None


def write_window_vrt(path, source, window, transform, bands, descriptions):
    types = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16", "uint32": "UInt32",
             "int32": "Int32", "float32": "Float32", "float64": "Float64"}
    with rasterio.open(source) as parent:
        root = ET.Element("VRTDataset", rasterXSize=str(int(window.width)), rasterYSize=str(int(window.height)))
        ET.SubElement(root, "SRS").text = parent.crs.to_wkt()
        ET.SubElement(root, "GeoTransform").text = ", ".join(map(str, transform.to_gdal()))
        for index, band in enumerate(bands, 1):
            layer = ET.SubElement(root, "VRTRasterBand", dataType=types[parent.dtypes[band-1]], band=str(index))
            ET.SubElement(layer, "Description").text = descriptions[index-1]
            if parent.nodatavals[band-1] is not None:
                ET.SubElement(layer, "NoDataValue").text = str(parent.nodatavals[band-1])
            simple = ET.SubElement(layer, "SimpleSource")
            ET.SubElement(simple, "SourceFilename", relativeToVRT="0").text = str(Path(source).resolve())
            ET.SubElement(simple, "SourceBand").text = str(band)
            ET.SubElement(simple, "SrcRect", xOff=str(int(window.col_off)), yOff=str(int(window.row_off)),
                          xSize=str(int(window.width)), ySize=str(int(window.height)))
            ET.SubElement(simple, "DstRect", xOff="0", yOff="0", xSize=str(int(window.width)), ySize=str(int(window.height)))
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def resolve_native_input(upload_path, record, settings, result_dir, declared_order=None):
    manifest_path = record.summary.get("source_manifest")
    scenes = []
    if manifest_path:
        manifest_path = (settings.root_dir / manifest_path).resolve()
        scenes = json.loads(manifest_path.read_text())["scenes"]
    with rasterio.open(upload_path) as uploaded:
        if uploaded.driver != "GTiff" or not is_native_grid(uploaded):
            raise ValueError("The 0.6 m model requires a north-up GeoTIFF on an approximately 0.6 m metric grid.")
        if uploaded.count not in (1, 4):
            raise ValueError("Use a four-band B/G/R/NIR TIFF, or one original band from a registered local product.")
        for scene in scenes:
            parent_path = Path(scene["images"][0])
            if not parent_path.exists():
                continue
            with rasterio.open(parent_path) as parent:
                window = source_window(uploaded, parent)
                matched = matching_bands(uploaded, parent, window) if window is not None else None
            if matched:
                stack = write_window_vrt(result_dir / "native_input_stack.vrt", parent_path, window,
                                         uploaded.transform, [1, 2, 3, 4], BAND_NAMES)
                quality = None
                if scene.get("quality_mask") and Path(scene["quality_mask"]).is_file():
                    quality = write_window_vrt(result_dir / "native_quality.vrt", scene["quality_mask"], window,
                                               uploaded.transform, [1], ["Footprint exclusion; not cloud truth"])
                return {"source": stack, "quality": quality, "normalization_reference": parent_path,
                        "normalization_quality": scene.get("quality_mask"), "matched_product": scene["scene_id"],
                        "uploaded_source_bands": matched, "companion_bands_recovered": uploaded.count == 1,
                        "parent_origin": [int(window.col_off), int(window.row_off)],
                        "band_order": ["blue", "green", "red", "nir"]}
        if uploaded.count == 1:
            raise ValueError("This single-band TIFF does not match the three registered 0.6 m products. "
                             "The new model needs all four B/G/R/NIR bands; it cannot infer NIR from BAND3. "
                             "Upload a four-band TIFF or the unchanged original band/crop from a registered product.")
        names = [re.sub(r"[^a-z]", "", str(value or "").lower()) for value in uploaded.descriptions]
        names = ["nir" if value in {"nearinfrared", "nearir"} else value for value in names]
        if set(names) == {"blue", "green", "red", "nir"}:
            bands = [names.index(value)+1 for value in ("blue", "green", "red", "nir")]
        elif declared_order == "bgrn":
            bands = [1, 2, 3, 4]
        else:
            raise ValueError("Four-band TIFF has no identifiable band descriptions. Confirm B/G/R/NIR order in the upload form first.")
        stack = write_window_vrt(result_dir / "native_input_stack.vrt", upload_path,
                                 Window(0, 0, uploaded.width, uploaded.height), uploaded.transform, bands, BAND_NAMES)
        return {"source": stack, "quality": None, "normalization_reference": stack,
                "normalization_quality": None, "matched_product": None,
                "uploaded_source_bands": bands, "companion_bands_recovered": False,
                "parent_origin": [0, 0],
                "band_order": ["blue", "green", "red", "nir"]}


class Native06mInference:
    def __init__(self, settings):
        self.settings = settings
        self._models = {}
        self._limits = {}
        self._clouds = {}
        self._lock = threading.Lock()
        source = settings.root_dir.parent / "native_06m_specialist/src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

    def load_model(self, record):
        if record.id not in self._models:
            from native06m.joint_refinement import JointNativeRefinement
            from native06m.model import choose_device
            from native06m.data import file_digest
            if not record.summary.get("experimental_testing_enabled"):
                raise ValueError("This partial checkpoint is not explicitly enabled for platform testing.")
            if file_digest(record.model_path) != record.summary.get("checkpoint_sha256"):
                raise ValueError("0.6 m checkpoint checksum does not match the approved testing model.")
            saved = torch.load(record.model_path, map_location="cpu", weights_only=True)
            if (saved.get("stage") != NATIVE_STAGE or not saved.get("pilot_complete")
                    or saved.get("architecture") not in {"JointNativeRefinement","NativeRoadContinuity"}
                    or saved.get("supported_classes") != list(range(5)) or saved["config"].get("in_channels") != 4):
                raise ValueError("Incompatible native 0.6 m testing checkpoint.")
            if saved.get("architecture")=="NativeRoadContinuity":
                from native06m.road_continuity import NativeRoadContinuity
                model=NativeRoadContinuity(saved["config"]["encoder_name"],4,
                    saved["config"].get("residual_bound",6.),saved["config"].get("road_correction_scale",1.),
                    saved["config"].get("road_head_mode","residual"),saved["config"].get("road_ensemble_weight",.3),
                    saved["config"].get("protected_base_classes",()), saved["config"].get("protection_confidence",.75),
                    saved["config"].get("road_margin_bias",0.))
            else:
                model = JointNativeRefinement(saved["config"]["encoder_name"], 4)
            model.load_state_dict(saved["state_dict"], strict=True)
            device = choose_device()
            self._models[record.id] = model.to(device).eval().requires_grad_(False), device
        return self._models[record.id]

    def predict(self, path, record, mode, declared_order=None, progress=None, refinement=True):
        with self._lock:
            return self._predict(path, record, mode, declared_order, progress or (lambda *_: None), refinement)

    def _predict(self, path, record, mode, declared_order, progress, refinement):
        from native06m.data import normalization
        from native06m.predict import native_stream_prediction
        result_id = f"{datetime.now():%Y%m%d_%H%M%S}_06m_{uuid.uuid4().hex[:8]}"
        directory = self.settings.result_dir / result_id
        directory.mkdir(parents=True, exist_ok=False)
        progress(.01, "Checking native grid and locating all four spectral bands")
        resolved = resolve_native_input(path, record, self.settings, directory, declared_order)
        progress(.03, "Loading the trained 0.6 m Attention U-Net++")
        model, device = self.load_model(record)
        reference = resolved["normalization_reference"]
        key = (str(reference), reference.stat().st_mtime_ns)
        if key not in self._limits:
            self._limits[key] = normalization(reference, resolved["normalization_quality"])
        limits = self._limits[key]
        destination = directory / "segmentation_labels_georef.tif"
        base_path = directory / "segmentation_labels_base_georef.tif" if refinement else destination
        road_recipe, evidence_path, cloud_quality_path, cloud_preview_path = None, None, None, None
        cleanup_recipe = None
        continuity_recipe=None
        strict_dot_extent_m = None
        maximum_gap_m = 1.2
        clouds, refinement_stats = None, {"enabled": False}
        if refinement:
            validation_path = (self.settings.root_dir / record.summary["refinement_validation_path"]).resolve()
            validation = json.loads(validation_path.read_text())
            if validation["source_checkpoint_sha256"] != record.summary["checkpoint_sha256"]:
                raise ValueError("Refinement validation does not match the selected checkpoint")
            road_recipe = validation.get("selected_road_recipe")
            cleanup_recipe = validation.get("selected_cleanup_recipe")
            continuity_recipe=validation.get("continuity_recipe")
            strict_dot_extent_m = validation.get("strict_dot_extent_m")
            maximum_gap_m = validation.get("maximum_gap_m", 1.2)
            if road_recipe or cleanup_recipe or continuity_recipe or strict_dot_extent_m:
                evidence_path = directory / "road_model_evidence_georef.tif"
            progress(.04, "Screening broad likely clouds; obscured areas will remain Unknown")
            if key not in self._clouds:
                self._clouds[key] = screen_reference_clouds(reference, limits, resolved["normalization_quality"])
            clouds = self._clouds[key]
        measured = native_stream_prediction(model, resolved["source"], base_path, device,
            patch=512, stride=int(record.summary.get("inference_stride",384)), batch_size=1,
            quality_path=resolved["quality"], normalization_limits=limits,
            evidence_path=evidence_path,
            progress=lambda fraction, message: progress(.05 + (.78 if refinement else .90) * fraction, message))
        if refinement:
            cloud_quality_path = directory / "cloud_quality_georef.tif"
            refinement_stats = {"enabled": True, **refine_native_raster(base_path, destination, cloud_quality_path,
                clouds, parent_origin=resolved["parent_origin"], evidence_path=evidence_path, road_recipe=road_recipe,
                cleanup_recipe=cleanup_recipe,
                continuity_recipe=continuity_recipe,
                strict_dot_extent_m=strict_dot_extent_m, maximum_gap_m=maximum_gap_m,
                progress=lambda fraction, message: progress(.83+.13*fraction, message))}
            measured["class_pixel_counts"] = refinement_stats["class_pixel_counts"]
        progress(.97, "Preparing marked image, segmentation and class coverage")
        with rasterio.open(resolved["source"]) as source, rasterio.open(destination) as labels:
            width, height = source.width, source.height
            scale = min(1., self.settings.preview_max_dimension / max(width, height))
            shape = max(1, round(height*scale)), max(1, round(width*scale))
            raw = source.read(out_shape=(4, *shape), resampling=Resampling.average).astype(np.float32)
            limits = np.asarray(limits).reshape(4, 2)
            normalized = np.clip((raw-limits[:, 0, None, None])/(limits[:, 1, None, None]-limits[:, 0, None, None]), 0, 1)
            rgb = np.round(normalized[[2, 1, 0]].transpose(1, 2, 0)*255).astype(np.uint8)
            preview = labels.read(1, out_shape=shape, resampling=Resampling.nearest)
            color = np.full((*shape, 3), 220, np.uint8)
            for index, value in record.display_class_colors.items():
                color[preview == index] = value
            # Keep clouds visible in the original input; only the classified map becomes Unknown.
            original_invalid = (source.read_masks(1, out_shape=shape) == 0) | ~np.isfinite(raw).all(0) | ~np.any(raw != 0, axis=0)
            rgb[original_invalid] = 220
            overlay = np.round(.57*rgb + .43*color).astype(np.uint8)
            overlay[preview == 255] = (220, 220, 220)
            if refinement:
                with rasterio.open(base_path) as baseline:
                    base_preview = baseline.read(1, out_shape=shape, resampling=Resampling.nearest)
                base_color = np.full((*shape, 3), 220, np.uint8)
                for index, value in record.display_class_colors.items():
                    base_color[base_preview == index] = value
                Image.fromarray(base_color).save(directory / "base_segmentation_preview.png")
                if refinement_stats["excluded_cloud_pixels"]:
                    codes = clouds.window(*resolved["parent_origin"], width, height, shape)
                    cloud_preview_path = directory / "cloud_screening_preview.png"
                    Image.fromarray(color_cloud_mask(codes, rgb)).save(cloud_preview_path)
            image_metadata = {"width": width, "height": height, "loader": "rasterio",
                              "processing_mode": "native_06m_fourband_streaming", "crs": str(source.crs),
                              "pixel_size_x": source.res[0], "pixel_size_y": source.res[1], "pixel_size_units": "metre",
                              "inference_target_gsd_m": .6, "inference_source_patch_size": 512,
                              "normalization_limits": limits.tolist(),
                              "band_order": resolved["band_order"], "matched_local_product": resolved["matched_product"],
                              "companion_bands_recovered": resolved["companion_bands_recovered"],
                              "uploaded_source_bands": resolved["uploaded_source_bands"], "filename": path.name,
                              "cloud_masking_applied": bool(refinement and refinement_stats["excluded_cloud_pixels"]),
                              "native_refinement": refinement_stats}
        for name, pixels in (("input_preview.png", rgb), ("segmentation_preview.png", color), ("marked_preview.png", overlay)):
            Image.fromarray(pixels).save(directory / name)
        counts = measured["class_pixel_counts"]
        rows = [{"class_id": index, "class_name": name, "pixel_count": counts[index],
                 "percentage": round(100*counts[index]/max(1, sum(counts)), 3)} for index, name in record.class_names.items()]
        rows.sort(key=lambda row: row["pixel_count"], reverse=True)
        (directory / "class_percentages.json").write_text(json.dumps(rows, indent=2))
        dominant = rows[0]["class_name"] if sum(counts) else "No valid pixels"
        base_metrics=record.summary.get("base_metrics") if not refinement else None
        selected = {"id": record.id, "display_name": record.display_name, "dataset_name": record.dataset_name,
                    "model_type": record.model_type,
                    "primary_metric_label": "Base reviewed development accuracy" if base_metrics else record.primary_metric_label,
                    "secondary_metric_label": "Base reviewed development mean IoU" if base_metrics else record.secondary_metric_label,
                    "validation_accuracy": base_metrics["val_accuracy"] if base_metrics else record.primary_metric_value,
                    "validation_macro_f1": base_metrics["mean_iou"] if base_metrics else record.secondary_metric_value,
                    "experimental": True,
                    "checkpoint_sha256":record.summary["checkpoint_sha256"],
                    "checkpoint_file":record.model_path.name,
                    "pipeline_version":record.summary.get("pipeline_version","native_06m_joint_v1")}
        scores = {"final_score": None, "confidence_mean": None, "domain_similarity": None,
                  "selection_basis": "Verified 0.6 m grid and four spectral bands; no fitted accuracy score"}
        warning = ("Experimental base 0.6 m checkpoint without automatic cloud exclusion or narrow-road refinement. "
                   "Roads remain unreliable; coverage percentages are predictions, not accuracy.")
        if refinement:
            warning = ("Experimental 0.6 m model with cloud-to-Unknown screening and model-supported road refinement. "
                       "Isolated compact road specks are screened and only short, supported gaps are repaired. "
                       "Roads still need visual review. Thin haze and cloud shadows may remain. Coverage is not accuracy.")
        if resolved["companion_bands_recovered"]:
            warning = f"Matched local product {resolved['matched_product']}; all four original bands were used. " + warning
        if record.summary.get("road_validation"):
            warning+=" "+record.summary["warning"]
        summary = {"selected_model": selected, "image": image_metadata, "selection_mode": mode,
                   "class_percentages": rows, "dominant_class": dominant, "note": warning,
                   "generalization_warning": warning, "model_fit_score": None, "score_breakdown": scores,
                   "checkpoint_sha256": record.summary["checkpoint_sha256"], "experimental_testing": True,
                   "training_complete": False, "production_approved": False,
                   "model_output_preserved": refinement_stats.get("model_output_preserved", True),
                   "legacy_pan_postprocessing_applied": False, "native_refinement": refinement_stats,
                   "class_percentage_scope": ("Classified pixels only; source no-data, detected cloud/uncertainty and unsupported road specks are excluded. Coverage is not accuracy."
                                              if refinement else "Valid source pixels only; base mode has no automatic cloud exclusion. Coverage is not accuracy."),
                   "vector_shapefiles": [], "vector_export_validation": {"ok": False, "message": "Generate on request"}}
        url = lambda name: f"/results/{result_id}/{name}"
        cloud_metadata = ({**clouds.metadata, **refinement_stats} if refinement else None)
        summary["cloud_screening"] = cloud_metadata
        (directory / "prediction_summary.json").write_text(json.dumps(summary, indent=2))
        legend = [{"class_id": index, "class_name": name, "color_rgb": list(record.display_class_colors[index])}
                  for index, name in record.class_names.items()]
        if refinement:
            legend.append({"class_id": 255, "class_name": "Unknown / cloud / uncertain / no-data", "color_rgb": [220, 220, 220]})
        return {"result_id": result_id, "image": image_metadata, "selection_mode": mode, "selected_model": selected,
                "score_breakdown": scores, "class_percentages": rows, "cloud_screening": cloud_metadata,
                "native_refinement": refinement_stats, "dominant_class": dominant,
                "candidate_rankings": [{"model_id": record.id, **selected, "score": None, "dominant_class": dominant,
                                        "score_breakdown": scores, "secondary_metric_value": record.secondary_metric_value}],
                "legend": legend,
                "artifacts": {"input_preview_url": url("input_preview.png"), "marked_preview_url": url("marked_preview.png"),
                              "segmentation_preview_url": url("segmentation_preview.png"), "geotiff_url": url(destination.name),
                              "summary_json_url": url("prediction_summary.json"), "class_percentages_url": url("class_percentages.json"),
                              "base_geotiff_url": url(base_path.name) if refinement else None,
                              "base_segmentation_preview_url": url("base_segmentation_preview.png") if refinement else None,
                              "cloud_quality_geotiff_url": url(cloud_quality_path.name) if cloud_quality_path else None,
                              "cloud_screening_preview_url": url(cloud_preview_path.name) if cloud_preview_path else None,
                              "class_shapefiles_zip_url": None}, "note": warning}
