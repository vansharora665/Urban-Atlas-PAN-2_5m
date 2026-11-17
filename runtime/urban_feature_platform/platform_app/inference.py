from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rasterio_features
from shapely.geometry import box as shapely_box
from shapely.geometry import shape as shapely_shape
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from .config import Settings
from .cloud_mask import CLOUD_COLOR, color_cloud_mask
from .image_utils import (
    LoadedImage,
    TiledImage,
    build_full_label_map,
    ensure_dir,
    load_uploaded_image,
    read_raster_below_threshold_mask,
    read_raster_valid_mask,
    read_normalized_raster_patch,
    robust_normalize,
    tile_image,
    to_uint8,
)
from .model_registry import ModelRecord, ModelRegistry
from .pan_overlap import blend_probabilities
from .pan_branches import PanWaterLand
from .pan_texture import FEATURES as PAN_TEXTURE_FEATURES, HALO as PAN_TEXTURE_HALO, texture_features
from .pan_texture_net import PanTextureNet
from .pan_radiometry import inherit_parent_normalization
from .pan_road_inference import apply_connected_pan_roads
from .road_water import ProbabilityConsensus, select_water_road_corrections, water_recovery_geometry
from .unet_model import UNet


NO_DATA_LABEL = 255


@dataclass
class ModelPrediction:
    record: ModelRecord
    patch_grid: np.ndarray
    patch_confidences: np.ndarray
    patch_labels: np.ndarray
    patch_coverages: np.ndarray
    class_percentages: list[dict[str, Any]]
    selected_score: float
    score_breakdown: dict[str, float]
    heuristic_water_overrides: int
    dominant_class_name: str
    full_label_map: np.ndarray | None = None
    reference_road_overlay_pixels: int = 0
    statistical_calibration_applied: bool = False
    contrast_enhancement_applied: bool = False
    road_specialist_pass_applied: bool = False
    learned_road_head_applied: bool = False
    learned_road_head_pixels_added: int = 0
    road_water_recovery: dict[str, Any] = field(default_factory=dict)
    road_water_correction_preview: np.ndarray | None = None
    topology_postprocessing_applied: bool = False
    water_coherence_cleanup_applied: bool = False
    water_coherence_pixels_restored: int = 0
    road_ratio_calibration_applied: bool = False
    road_buffer_meters: float | None = None
    road_target_share_of_urban: float | None = None


def compute_patch_features(
    patches: np.ndarray,
    resize_for_features: int,
    canny_low: int,
    canny_high: int,
) -> np.ndarray:
    features = []
    for patch in patches:
        small = cv2.resize(patch, (resize_for_features, resize_for_features), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        patch_u8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
        edges = cv2.Canny(patch_u8, canny_low, canny_high)
        stats = np.array(
            [
                patch.mean(),
                patch.std(),
                np.percentile(patch, 10),
                np.percentile(patch, 90),
                grad.mean(),
                grad.std(),
                edges.mean() / 255.0,
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([small.flatten(), stats]))
    return np.stack(features).astype(np.float32)


def colorize_labels(label_image: np.ndarray, record: ModelRecord) -> np.ndarray:
    color = np.full((*label_image.shape, 3), 232, dtype=np.uint8)
    for class_id, class_name in record.class_names.items():
        rgb = record.display_class_colors.get(class_id, (200, 200, 200))
        color[label_image == class_id] = rgb
    color[label_image == NO_DATA_LABEL] = (220, 220, 220)
    return color


def resize_label_preview(patch_grid: np.ndarray, width: int, height: int) -> np.ndarray:
    preview = Image.fromarray(patch_grid.astype(np.uint8), mode="L")
    preview = preview.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(preview, dtype=np.uint8)


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = np.log(probabilities.shape[1]) if probabilities.shape[1] > 1 else 1.0
    return entropy / max(max_entropy, 1e-8)


def cluster_separation_values(reduced_features: np.ndarray, kmeans: Any) -> np.ndarray:
    distances = kmeans.transform(reduced_features)
    if distances.shape[1] < 2:
        return np.full((reduced_features.shape[0],), 0.5, dtype=np.float32)
    sorted_distances = np.sort(distances, axis=1)
    first = sorted_distances[:, 0]
    second = sorted_distances[:, 1]
    separation = np.clip((second - first) / np.maximum(second, 1e-6), 0.0, 1.0)
    return separation.astype(np.float32)


def cluster_separation_score(reduced_features: np.ndarray, kmeans: Any) -> float:
    return float(np.mean(cluster_separation_values(reduced_features, kmeans)))


def class_percentage_table(
    labels: np.ndarray,
    coverages: np.ndarray,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    total_pixels = int(coverages.sum())
    rows = []
    for class_id, class_name in class_names.items():
        class_pixels = int(coverages[labels == class_id].sum())
        percent = (100.0 * class_pixels / total_pixels) if total_pixels else 0.0
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "pixel_count": class_pixels,
                "percentage": round(percent, 2),
            }
        )
    rows.sort(key=lambda item: item["percentage"], reverse=True)
    return rows


def class_percentage_table_from_counts(
    class_pixel_counts: np.ndarray,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    total_pixels = int(class_pixel_counts.sum())
    rows = []
    for class_id, class_name in class_names.items():
        class_pixels = int(class_pixel_counts[class_id]) if class_id < len(class_pixel_counts) else 0
        percent = (100.0 * class_pixels / total_pixels) if total_pixels else 0.0
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "pixel_count": class_pixels,
                "percentage": round(percent, 2),
            }
        )
    rows.sort(key=lambda item: item["percentage"], reverse=True)
    return rows


def normalized_entropy_map(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = np.log(probabilities.shape[1]) if probabilities.shape[1] > 1 else 1.0
    return entropy / max(max_entropy, 1e-8)


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class InferenceService:
    def __init__(self, settings: Settings, registry: ModelRegistry):
        self.settings = settings
        self.registry = registry
        self._domain_profiles: dict[str, dict[str, Any]] = {}
        self._torch_models: dict[str, tuple[torch.nn.Module, torch.device]] = {}
        self._dense_auto_selector: dict[str, Any] | None = None
        self._prediction_lock = threading.Lock()
        self._road_head_fingerprint: str | None = None

    def _raster_inference_geometry(self, loaded_image: LoadedImage, patch_size: int) -> tuple[int, float | None, float]:
        units = loaded_image.source_metadata.get("pixel_size_units")
        pixel_size_x = loaded_image.source_metadata.get("pixel_size_x")
        pixel_size_y = loaded_image.source_metadata.get("pixel_size_y")
        if units != "metre" or pixel_size_x is None or pixel_size_y is None:
            return patch_size, None, 1.0

        native_gsd = float((abs(float(pixel_size_x)) + abs(float(pixel_size_y))) / 2.0)
        if not np.isfinite(native_gsd) or native_gsd <= 0:
            return patch_size, None, 1.0

        requested_scale = float(self.settings.target_inference_gsd_m) / native_gsd
        scale = float(np.clip(requested_scale, 0.5, 4.0))
        source_patch_size = max(32, int(round(patch_size * scale)))
        effective_gsd = native_gsd * source_patch_size / float(patch_size)
        return source_patch_size, effective_gsd, scale

    def _use_local_contrast(self, loaded_image: LoadedImage, scale: float) -> bool:
        if not self.settings.enable_local_contrast:
            return False
        return bool(loaded_image.source_metadata.get("radiometric_tail_detected")) or abs(scale - 1.0) >= 0.20

    def _cloud_threshold(self, loaded_image: LoadedImage) -> float | None:
        if loaded_image.cloud_mask is not None:
            return None
        value = loaded_image.source_metadata.get("cloud_threshold")
        return float(value) if value is not None else None

    def _road_cloud_threshold(self, loaded_image: LoadedImage) -> float | None:
        value = loaded_image.source_metadata.get("cloud_threshold")
        threshold = float(value) if value is not None else None
        if threshold is None or loaded_image.normalization_high is None:
            return threshold
        high = float(loaded_image.normalization_high)
        # Saturated PAN responses are usually cloud/glare rather than road surface.
        guard_fraction = float(np.clip(self.settings.road_head_bright_guard_fraction, 0.0, 1.0))
        return min(threshold, high + guard_fraction * max(0.0, threshold - high))

    def _should_override_dark_patch(self, record: ModelRecord, patch_mean: float, patch_std: float) -> bool:
        return bool(
            self.settings.enable_dark_water_override
            and 0 in record.class_names
            and patch_std < record.std_threshold_value
            and patch_mean < record.dark_water_mean_threshold_value
        )

    def _refresh_prediction_percentages(self, prediction: ModelPrediction) -> None:
        if prediction.full_label_map is None:
            return
        labels = prediction.full_label_map
        class_pixel_counts = np.zeros(len(prediction.record.class_names), dtype=np.int64)
        # Bounded strips avoid multi-GB int64 copies for large TIFFs.
        for y in range(0, labels.shape[0], 256):
            block = labels[y:y+256]
            valid = block[block != NO_DATA_LABEL]
            class_pixel_counts += np.bincount(valid, minlength=len(class_pixel_counts))
        prediction.class_percentages = class_percentage_table_from_counts(
            class_pixel_counts,
            prediction.record.class_names,
        )
        prediction.dominant_class_name = prediction.class_percentages[0]["class_name"] if class_pixel_counts.sum() else "No clear classified pixels"

    def _apply_cloud_exclusions(self, loaded_image: LoadedImage, prediction: ModelPrediction) -> None:
        cloud_mask = loaded_image.cloud_mask
        if cloud_mask is None or prediction.full_label_map is None:
            return
        if not cloud_mask.has_clouds:
            cloud_mask.metadata.update({"excluded_pixels": 0, "excluded_percent_of_full_grid": 0.0,
                                        "class_percentage_denominator": "Clear classified pixels only; no-data excluded"})
            return
        counts = np.zeros(3, dtype=np.int64)
        for y in range(0, loaded_image.height, 512):
            height = min(512, loaded_image.height - y)
            codes = cloud_mask.window(0, y, loaded_image.width, height)
            counts += np.bincount(codes.ravel(), minlength=3)
            prediction.full_label_map[y:y+height][codes > 0] = NO_DATA_LABEL
        cloud_mask.metadata.update({
            "likely_cloud_pixels": int(counts[1]),
            "cloud_fringe_uncertain_pixels": int(counts[2]),
            "excluded_pixels": int(counts[1] + counts[2]),
            "excluded_percent_of_full_grid": round(float(100 * (counts[1] + counts[2]) / counts.sum()), 3),
            "class_percentage_denominator": "Clear classified pixels only; cloud, fringe and no-data excluded",
        })
        self._refresh_prediction_percentages(prediction)

    def _write_cloud_quality_raster(self, destination: Path, loaded_image: LoadedImage) -> None:
        profile = loaded_image.raster_profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255, compress="lzw", tiled=True,
                       blockxsize=256, blockysize=256)
        with rasterio.open(loaded_image.source_path) as source, rasterio.open(destination, "w", **profile) as dst:
            for y in range(0, loaded_image.height, 512):
                height = min(512, loaded_image.height - y)
                codes = loaded_image.cloud_mask.window(0, y, loaded_image.width, height).copy()
                valid = read_raster_valid_mask(source, x0=0, y0=y, width=loaded_image.width, height=height)
                codes[~valid] = 255
                dst.write(codes, 1, window=rasterio.windows.Window(0, y, loaded_image.width, height))
            dst.update_tags(quality_codes="0=not screened out;1=likely cloud;2=cloud fringe/uncertain;255=source no-data",
                            method="single_band_spatial_cloud_screen_v1", shadows="Not automatically masked",
                            limitation="Not validated cloud truth; no hidden ground reconstruction")

    def _generalization_warning(self, prediction: ModelPrediction) -> str | None:
        breakdown = prediction.score_breakdown
        issues = []
        if float(breakdown.get("domain_similarity", 1.0)) < 0.70:
            issues.append("training-domain similarity is low")
        if float(breakdown.get("confidence_mean", 1.0)) < 0.65:
            issues.append("mean model confidence is low")
        if float(breakdown.get("final_score", 1.0)) < 0.60:
            issues.append("overall model-fit score is low")
        if not issues:
            return None
        return "Review recommended: " + "; ".join(issues) + "."

    def _domain_profile_cache_path(self, record: ModelRecord) -> Path:
        return self.settings.runtime_dir / "cache" / "domain_profiles" / f"{record.id}.json"

    def _get_domain_profile(self, record: ModelRecord) -> dict[str, Any]:
        if record.id in self._domain_profiles:
            return self._domain_profiles[record.id]

        cache_path = self._domain_profile_cache_path(record)
        ensure_dir(cache_path.parent)
        if cache_path.exists():
            profile = json.loads(cache_path.read_text())
            self._domain_profiles[record.id] = profile
            return profile

        project_root = self.settings.root_dir.parent
        dataset = record.summary["dataset"]
        image_rel_path = dataset.get("image_path") or dataset.get("spec", {}).get("image_path")
        if not image_rel_path:
            raise ValueError(f"Training image path is missing for model {record.id}")
        image_path = (project_root / image_rel_path).resolve()
        with rasterio.open(image_path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            valid_mask = np.ones_like(band, dtype=bool)
            if nodata is not None:
                valid_mask &= np.not_equal(band, nodata)
        normalized, _ = robust_normalize(band, valid_mask)
        tiled = tile_image(normalized, valid_mask, patch_size=record.patch_size)

        required_coverage = int(np.ceil(record.valid_fraction_threshold_value * (record.patch_size ** 2)))
        keep_mask = (tiled.patch_coverages >= required_coverage) & (tiled.patch_stds >= record.std_threshold_value)
        training_patches = tiled.patches[keep_mask]
        if len(training_patches) == 0:
            training_patches = tiled.patches

        if len(training_patches) > 256:
            sample_indices = np.linspace(0, len(training_patches) - 1, 256, dtype=int)
            training_patches = training_patches[sample_indices]

        config = record.model_config
        feature_matrix = compute_patch_features(
            training_patches,
            resize_for_features=int(config.get("resize_for_features", 32)),
            canny_low=int(config.get("canny_low", 60)),
            canny_high=int(config.get("canny_high", 160)),
        )
        handcrafted = feature_matrix[:, -7:]
        profile = {
            "image_path": str(image_path),
            "sample_count": int(len(training_patches)),
            "handcrafted_mean": np.mean(handcrafted, axis=0).round(6).tolist(),
            "handcrafted_std": np.maximum(np.std(handcrafted, axis=0), 1e-4).round(6).tolist(),
        }
        cache_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        self._domain_profiles[record.id] = profile
        return profile

    def _domain_similarity(self, record: ModelRecord, handcrafted_features: np.ndarray) -> float:
        profile = self._get_domain_profile(record)
        reference_mean = np.asarray(profile["handcrafted_mean"], dtype=np.float32)
        reference_std = np.asarray(profile["handcrafted_std"], dtype=np.float32)
        upload_mean = np.mean(handcrafted_features, axis=0)
        upload_std = np.std(handcrafted_features, axis=0)
        mean_distance = np.mean(np.abs(upload_mean - reference_mean) / np.maximum(reference_std, 1e-4))
        std_distance = np.mean(np.abs(upload_std - reference_std) / np.maximum(reference_std, 1e-4))
        combined_distance = 0.65 * mean_distance + 0.35 * std_distance
        return float(1.0 / (1.0 + combined_distance))

    def _dense_auto_selector_cache_path(self, records: list[ModelRecord]) -> Path:
        selector_key = "_".join(sorted(record.id for record in records))
        return self.settings.runtime_dir / "cache" / "dense_auto_selector" / f"{selector_key}_v2.joblib"

    def _get_dense_auto_selector(self, records: list[ModelRecord]) -> dict[str, Any]:
        if self._dense_auto_selector is not None:
            return self._dense_auto_selector

        cache_path = self._dense_auto_selector_cache_path(records)
        ensure_dir(cache_path.parent)
        if cache_path.exists():
            self._dense_auto_selector = json.loads(cache_path.read_text())
            self._dense_auto_selector["classifier"] = joblib.load(str(cache_path.with_suffix(".clf.joblib")))
            return self._dense_auto_selector

        patch_size = records[0].patch_size
        feature_blocks = []
        label_blocks: list[str] = []
        for record in records:
            sample_patches: np.ndarray
            dense_metadata_path = record.summary.get("artifacts", {}).get("dense_metadata_path")
            if dense_metadata_path and Path(dense_metadata_path).exists():
                dense_metadata = pd.read_csv(dense_metadata_path)
                sample_frames = []
                for _, group in dense_metadata.groupby("dominant_class"):
                    if len(group) > 320:
                        sample_frames.append(group.sample(n=320, random_state=42))
                    else:
                        sample_frames.append(group)
                sampled = pd.concat(sample_frames, ignore_index=True)
                image_patches = []
                for _, row in sampled.iterrows():
                    patch = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32) / 255.0
                    image_patches.append(patch)
                sample_patches = np.stack(image_patches).astype(np.float32)
            else:
                project_root = self.settings.root_dir.parent
                dataset = record.summary["dataset"]
                image_rel_path = dataset.get("image_path") or dataset.get("spec", {}).get("image_path")
                image_path = (project_root / image_rel_path).resolve()
                with rasterio.open(image_path) as src:
                    band = src.read(1).astype(np.float32)
                    nodata = src.nodata
                    valid_mask = np.ones_like(band, dtype=bool)
                    if nodata is not None:
                        valid_mask &= np.not_equal(band, nodata)
                normalized, _ = robust_normalize(band, valid_mask)
                tiled = tile_image(normalized, valid_mask, patch_size=patch_size)
                keep_mask = tiled.patch_coverages >= int(np.ceil(record.valid_fraction_threshold_value * (patch_size ** 2)))
                sample_patches = tiled.patches[keep_mask]
                if len(sample_patches) == 0:
                    sample_patches = tiled.patches
                if len(sample_patches) > 256:
                    sample_indices = np.linspace(0, len(sample_patches) - 1, 256, dtype=int)
                    sample_patches = sample_patches[sample_indices]
            feature_matrix = compute_patch_features(
                sample_patches,
                resize_for_features=int(record.model_config.get("resize_for_features", 32)),
                canny_low=int(record.model_config.get("canny_low", 60)),
                canny_high=int(record.model_config.get("canny_high", 160)),
            )
            feature_blocks.append(feature_matrix)
            label_blocks.extend([record.id] * len(feature_matrix))

        x = np.vstack(feature_blocks).astype(np.float32)
        y = np.asarray(label_blocks)
        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        classifier.fit(x, y)

        meta = {"model_ids": sorted(record.id for record in records), "patch_size": patch_size}
        cache_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        joblib.dump(classifier, str(cache_path.with_suffix(".clf.joblib")))
        meta["classifier"] = classifier
        self._dense_auto_selector = meta
        return meta

    def _sample_selector_patches(self, loaded_image: LoadedImage, patch_size: int) -> np.ndarray:
        target_count = 96
        if loaded_image.processing_mode != "streaming_raster":
            assert loaded_image.grayscale is not None
            assert loaded_image.valid_mask is not None
            tiled = tile_image(loaded_image.grayscale, loaded_image.valid_mask, patch_size=patch_size)
            sample_patches = tiled.patches
            if len(sample_patches) > target_count:
                sample_indices = np.linspace(0, len(sample_patches) - 1, target_count, dtype=int)
                sample_patches = sample_patches[sample_indices]
            return sample_patches.astype(np.float32)

        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)
        source_patch_size, _, scale = self._raster_inference_geometry(loaded_image, patch_size)
        row_positions = np.linspace(0, max(loaded_image.height - source_patch_size, 0), 8, dtype=int)
        col_positions = np.linspace(0, max(loaded_image.width - source_patch_size, 0), 8, dtype=int)
        cloud_threshold = self._cloud_threshold(loaded_image)
        enhance_local_contrast = self._use_local_contrast(loaded_image, scale)
        patches = []
        with rasterio.open(loaded_image.source_path) as src:
            for y0 in row_positions:
                for x0 in col_positions:
                    patch, _ = read_normalized_raster_patch(
                        src,
                        x0=int(x0),
                        y0=int(y0),
                        patch_size=patch_size,
                        normalization_low=normalization_low,
                        normalization_high=normalization_high,
                        source_window_size=source_patch_size,
                        cloud_threshold=cloud_threshold,
                        cloud_mask=loaded_image.cloud_mask,
                        enhance_local_contrast=enhance_local_contrast,
                    )
                    patches.append(patch)
        return np.stack(patches).astype(np.float32)

    def _choose_record_from_filename_hint(self, loaded_image: LoadedImage, records: list[ModelRecord]) -> ModelRecord | None:
        filename = loaded_image.source_path.name.lower()
        dataset_tokens = {
            "jnpa": ("jnpa",),
            "cartosat": ("cartosat",),
        }

        for dataset_key, tokens in dataset_tokens.items():
            if not any(token in filename for token in tokens):
                continue
            matched = [
                record
                for record in records
                if dataset_key in record.id.lower()
                or dataset_key in record.display_name.lower()
                or dataset_key in record.dataset_name.lower()
            ]
            if matched:
                return sorted(matched, key=lambda record: (-record.priority, record.display_name))[0]
        return None

    def _choose_dense_auto_record(self, loaded_image: LoadedImage, records: list[ModelRecord]) -> ModelRecord:
        if not records:
            raise ValueError("At least one dense model record is required.")

        hinted_record = self._choose_record_from_filename_hint(loaded_image, records)
        if hinted_record is not None:
            return hinted_record

        sampled_by_patch_size: dict[int, np.ndarray] = {}
        best_record = records[0]
        best_score = float("-inf")

        for record in records:
            sample_patches = sampled_by_patch_size.get(record.patch_size)
            if sample_patches is None:
                sample_patches = self._sample_selector_patches(loaded_image, record.patch_size)
                sampled_by_patch_size[record.patch_size] = sample_patches
            if len(sample_patches) == 0:
                continue

            config = record.model_config
            resize_for_features = int(config.get("resize_for_features", 32))
            canny_low = int(config.get("canny_low", 60))
            canny_high = int(config.get("canny_high", 160))
            feature_matrix = compute_patch_features(
                sample_patches,
                resize_for_features=resize_for_features,
                canny_low=canny_low,
                canny_high=canny_high,
            )
            handcrafted = feature_matrix[:, -7:]

            model, device = self._get_unet_model(record)
            confidence_batches: list[np.ndarray] = []
            entropy_batches: list[np.ndarray] = []
            batch_size = min(max(1, int(self.settings.prediction_batch_patches)), 48)

            with torch.no_grad():
                for start in range(0, len(sample_patches), batch_size):
                    end = min(start + batch_size, len(sample_patches))
                    batch = torch.from_numpy(sample_patches[start:end]).unsqueeze(1).float().to(device)
                    logits = model(batch)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                    confidence_maps = probabilities.max(axis=1)
                    entropy_maps = normalized_entropy_map(probabilities)
                    confidence_batches.append(confidence_maps.reshape(confidence_maps.shape[0], -1).mean(axis=1).astype(np.float32))
                    entropy_batches.append(entropy_maps.reshape(entropy_maps.shape[0], -1).mean(axis=1).astype(np.float32))

            score, _ = self._score_unet_fit_from_metrics(
                record,
                max_confidences=np.concatenate(confidence_batches),
                normalized_entropies=np.concatenate(entropy_batches),
                handcrafted_features=handcrafted,
                class_count=len(record.class_names),
            )
            if score > best_score:
                best_score = score
                best_record = record

        return best_record

    def _score_model_fit_from_metrics(
        self,
        record: ModelRecord,
        *,
        max_confidences: np.ndarray,
        normalized_entropies: np.ndarray,
        separations: np.ndarray,
        handcrafted_features: np.ndarray,
        class_count: int,
    ) -> tuple[float, dict[str, float]]:
        class_count = max(1, class_count)
        chance_level = 1.0 / class_count
        confidence_mean = float(np.mean(max_confidences))
        confidence_p90 = float(np.percentile(max_confidences, 90))
        adjusted_confidence_mean = float((confidence_mean - chance_level) / max(1.0 - chance_level, 1e-6))
        adjusted_confidence_p90 = float((confidence_p90 - chance_level) / max(1.0 - chance_level, 1e-6))
        adjusted_confidence_mean = float(np.clip(adjusted_confidence_mean, 0.0, 1.0))
        adjusted_confidence_p90 = float(np.clip(adjusted_confidence_p90, 0.0, 1.0))
        certainty = float(1.0 - np.mean(normalized_entropies))
        separation = float(np.mean(separations))
        validation_accuracy = float(record.model_metrics.get("val_accuracy", 0.0) or 0.0)
        domain_similarity = self._domain_similarity(record, handcrafted_features)
        score = (
            0.18 * adjusted_confidence_mean
            + 0.14 * certainty
            + 0.13 * adjusted_confidence_p90
            + 0.10 * separation
            + 0.10 * validation_accuracy
            + 0.35 * domain_similarity
        )
        breakdown = {
            "confidence_mean": round(confidence_mean, 4),
            "adjusted_confidence_mean": round(adjusted_confidence_mean, 4),
            "certainty": round(certainty, 4),
            "confidence_p90": round(confidence_p90, 4),
            "adjusted_confidence_p90": round(adjusted_confidence_p90, 4),
            "cluster_separation": round(separation, 4),
            "validation_accuracy": round(validation_accuracy, 4),
            "domain_similarity": round(domain_similarity, 4),
            "final_score": round(float(score), 4),
        }
        return float(score), breakdown

    def _score_model_fit(
        self,
        record: ModelRecord,
        probabilities: np.ndarray,
        reduced_features: np.ndarray,
        kmeans: Any,
        handcrafted_features: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        return self._score_model_fit_from_metrics(
            record,
            max_confidences=probabilities.max(axis=1),
            normalized_entropies=normalized_entropy(probabilities),
            separations=cluster_separation_values(reduced_features, kmeans),
            handcrafted_features=handcrafted_features,
            class_count=probabilities.shape[1],
        )

    def _score_unet_fit_from_metrics(
        self,
        record: ModelRecord,
        *,
        max_confidences: np.ndarray,
        normalized_entropies: np.ndarray,
        handcrafted_features: np.ndarray,
        class_count: int,
    ) -> tuple[float, dict[str, float]]:
        class_count = max(1, class_count)
        chance_level = 1.0 / class_count
        confidence_mean = float(np.mean(max_confidences))
        confidence_p90 = float(np.percentile(max_confidences, 90))
        adjusted_confidence_mean = float((confidence_mean - chance_level) / max(1.0 - chance_level, 1e-6))
        adjusted_confidence_p90 = float((confidence_p90 - chance_level) / max(1.0 - chance_level, 1e-6))
        adjusted_confidence_mean = float(np.clip(adjusted_confidence_mean, 0.0, 1.0))
        adjusted_confidence_p90 = float(np.clip(adjusted_confidence_p90, 0.0, 1.0))
        certainty = float(1.0 - np.mean(normalized_entropies))
        validation_accuracy = float(record.model_metrics.get("val_accuracy", 0.0) or 0.0)
        domain_similarity = self._domain_similarity(record, handcrafted_features)
        score = (
            0.22 * adjusted_confidence_mean
            + 0.18 * certainty
            + 0.15 * adjusted_confidence_p90
            + 0.15 * validation_accuracy
            + 0.30 * domain_similarity
        )
        breakdown = {
            "confidence_mean": round(confidence_mean, 4),
            "adjusted_confidence_mean": round(adjusted_confidence_mean, 4),
            "certainty": round(certainty, 4),
            "confidence_p90": round(confidence_p90, 4),
            "adjusted_confidence_p90": round(adjusted_confidence_p90, 4),
            "cluster_separation": 0.0,
            "validation_accuracy": round(validation_accuracy, 4),
            "domain_similarity": round(domain_similarity, 4),
            "final_score": round(float(score), 4),
        }
        return float(score), breakdown

    def _create_smp_model(self, model_name: str, classes: int) -> torch.nn.Module:
        if model_name == "deeplabv3plus":
            return smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=classes)
        if model_name == "segformer":
            return smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=1, classes=classes)
        if model_name == "unetplusplus":
            return smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=classes)
        raise ValueError(f"Unsupported SMP model architecture: {model_name}")

    def _get_unet_model(self, record: ModelRecord) -> tuple[torch.nn.Module, torch.device]:
        cached = self._torch_models.get(record.id)
        if cached is not None:
            return cached

        checkpoint = record.load_bundle()
        if record.model_type == "smp_segmentation":
            model_name = checkpoint.get("model_name") or record.summary.get("model_name")
            if not model_name:
                raise ValueError(f"SMP checkpoint for {record.id} is missing model_name.")
            model = self._create_smp_model(str(model_name), len(record.class_names))
            model.load_state_dict(checkpoint["model"])
            water_branch = record.summary.get("water_branch")
            if water_branch:
                path = (self.settings.root_dir / water_branch["checkpoint"]).resolve()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != water_branch["checkpoint_sha256"]:
                    raise ValueError("Water-branch fingerprint mismatch; refusing incompatible PAN weights")
                water_model = self._create_smp_model(str(model_name), len(record.class_names))
                water_model.load_state_dict(torch.load(path, map_location="cpu")["model"])
                model = PanWaterLand(water_model, model)
            device = resolve_torch_device()
            model = model.to(device)
            model.eval()
            self._torch_models[record.id] = (model, device)
            return model, device

        state_dict = checkpoint.get("model", checkpoint)
        base_channels = int(state_dict["down1.block.0.weight"].shape[0])
        model = UNet(in_channels=1, num_classes=len(record.class_names), base_channels=base_channels)
        model.load_state_dict(state_dict)
        device = resolve_torch_device()
        model = model.to(device)
        model.eval()
        self._torch_models[record.id] = (model, device)
        return model, device

    def _get_learned_road_head(self, classes: int, expected_fingerprint: str | None = None) -> tuple[torch.nn.Module, torch.device]:
        cache_key = "__learned_road_head__"
        cached = self._torch_models.get(cache_key)
        if cached is not None:
            if expected_fingerprint and self._road_head_fingerprint != expected_fingerprint:
                raise ValueError("Cached road specialist does not match the registered checkpoint fingerprint")
            return cached

        checkpoint_path = self.settings.road_head_model_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Road-head checkpoint is missing: {checkpoint_path}")
        checkpoint_bytes = checkpoint_path.read_bytes()
        fingerprint = hashlib.sha256(checkpoint_bytes).hexdigest()
        if expected_fingerprint and fingerprint != expected_fingerprint:
            raise ValueError("Road specialist checkpoint fingerprint mismatch")
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
        model_name = str(checkpoint.get("model_name", "segformer"))
        model = self._create_smp_model(model_name, classes)
        model.load_state_dict(checkpoint["model"])
        device = resolve_torch_device()
        model = model.to(device)
        model.eval()
        self._torch_models[cache_key] = (model, device)
        self._road_head_fingerprint = fingerprint
        return model, device

    def _predict_with_patch_classifier(
        self,
        loaded_image: LoadedImage,
        tiled: TiledImage,
        record: ModelRecord,
    ) -> ModelPrediction:
        bundle = record.load_bundle()
        config = bundle.get("config", {})
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        feature_matrix = compute_patch_features(
            tiled.patches,
            resize_for_features=resize_for_features,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        handcrafted = feature_matrix[:, -7:]

        reduced = bundle["pca"].transform(feature_matrix)
        classifier = bundle["classifier"]
        probabilities = classifier.predict_proba(reduced)
        predicted_indices = probabilities.argmax(axis=1)
        predicted_labels = classifier.classes_[predicted_indices].astype(np.uint8)
        patch_confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices].astype(np.float32)

        overrides = 0
        for idx in range(len(predicted_labels)):
            if self._should_override_dark_patch(record, float(tiled.patch_means[idx]), float(tiled.patch_stds[idx])):
                if predicted_labels[idx] != 0:
                    overrides += 1
                predicted_labels[idx] = 0
                patch_confidences[idx] = max(float(patch_confidences[idx]), 0.85)

        score, score_breakdown = self._score_model_fit(
            record,
            probabilities,
            reduced,
            bundle["kmeans"],
            handcrafted,
        )
        class_percentages = class_percentage_table(predicted_labels, tiled.patch_coverages, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=predicted_labels.reshape(tiled.rows, tiled.cols),
            patch_confidences=patch_confidences,
            patch_labels=predicted_labels,
            patch_coverages=tiled.patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
        )

    def _predict_with_unet(
        self,
        loaded_image: LoadedImage,
        tiled: TiledImage,
        record: ModelRecord,
    ) -> ModelPrediction:
        model, device = self._get_unet_model(record)
        config = record.model_config
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        feature_matrix = compute_patch_features(
            tiled.patches,
            resize_for_features=resize_for_features,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        handcrafted = feature_matrix[:, -7:]

        batch_size = max(1, int(self.settings.prediction_batch_patches))
        patch_labels = np.zeros(len(tiled.patches), dtype=np.uint8)
        patch_confidences = np.zeros(len(tiled.patches), dtype=np.float32)
        max_confidences: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        overrides = 0

        with torch.no_grad():
            for start in range(0, len(tiled.patches), batch_size):
                end = min(start + batch_size, len(tiled.patches))
                batch = torch.from_numpy(tiled.patches[start:end]).unsqueeze(1).float().to(device)
                logits = model(batch)
                patch_logits = logits.mean(dim=(2, 3))
                probabilities = torch.softmax(patch_logits, dim=1).cpu().numpy()
                predicted_indices = probabilities.argmax(axis=1)
                predicted = predicted_indices.astype(np.uint8)
                confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices].astype(np.float32)

                for local_idx, global_idx in enumerate(range(start, end)):
                    if self._should_override_dark_patch(
                        record,
                        float(tiled.patch_means[global_idx]),
                        float(tiled.patch_stds[global_idx]),
                    ):
                        if predicted[local_idx] != 0:
                            overrides += 1
                        predicted[local_idx] = 0
                        confidences[local_idx] = max(float(confidences[local_idx]), 0.85)

                patch_labels[start:end] = predicted
                patch_confidences[start:end] = confidences
                max_confidences.append(confidences)
                entropies.append(normalized_entropy(probabilities).astype(np.float32))

        score, score_breakdown = self._score_unet_fit_from_metrics(
            record,
            max_confidences=np.concatenate(max_confidences),
            normalized_entropies=np.concatenate(entropies),
            handcrafted_features=handcrafted,
            class_count=len(record.class_names),
        )
        class_percentages = class_percentage_table(patch_labels, tiled.patch_coverages, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(tiled.rows, tiled.cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=tiled.patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
        )

    def _predict_with_dense_unet(
        self,
        loaded_image: LoadedImage,
        tiled: TiledImage,
        record: ModelRecord,
    ) -> ModelPrediction:
        model, device = self._get_unet_model(record)
        config = record.model_config
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        feature_matrix = compute_patch_features(
            tiled.patches,
            resize_for_features=resize_for_features,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        handcrafted = feature_matrix[:, -7:]

        patch_size = record.patch_size
        full_label_map = np.full((tiled.rows * patch_size, tiled.cols * patch_size), NO_DATA_LABEL, dtype=np.uint8)
        valid_crop = np.zeros(full_label_map.shape, dtype=bool)
        assert loaded_image.valid_mask is not None
        valid_h, valid_w = loaded_image.valid_mask.shape
        valid_crop[:valid_h, :valid_w] = loaded_image.valid_mask
        patch_labels = np.zeros(len(tiled.patches), dtype=np.uint8)
        patch_confidences = np.zeros(len(tiled.patches), dtype=np.float32)
        patch_coverages = tiled.patch_coverages.copy()
        patch_entropies = np.zeros(len(tiled.patches), dtype=np.float32)
        class_pixel_counts = np.zeros(len(record.class_names), dtype=np.int64)
        overrides = 0

        batch_size = max(1, int(self.settings.prediction_batch_patches))
        with torch.no_grad():
            for start in range(0, len(tiled.patches), batch_size):
                end = min(start + batch_size, len(tiled.patches))
                batch = torch.from_numpy(tiled.patches[start:end]).unsqueeze(1).float().to(device)
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predicted_masks = probabilities.argmax(axis=1).astype(np.uint8)
                confidence_maps = probabilities.max(axis=1).astype(np.float32)
                entropy_maps = normalized_entropy_map(probabilities).astype(np.float32)

                for local_idx, global_idx in enumerate(range(start, end)):
                    row = global_idx // tiled.cols
                    col = global_idx % tiled.cols
                    y0 = row * patch_size
                    x0 = col * patch_size
                    y1 = y0 + patch_size
                    x1 = x0 + patch_size
                    pred_mask = predicted_masks[local_idx]
                    conf_map = confidence_maps[local_idx]
                    entropy_map = entropy_maps[local_idx]

                    if self._should_override_dark_patch(
                        record,
                        float(tiled.patch_means[global_idx]),
                        float(tiled.patch_stds[global_idx]),
                    ):
                        if np.any(pred_mask != 0):
                            overrides += 1
                        pred_mask = np.zeros_like(pred_mask, dtype=np.uint8)
                        conf_map = np.maximum(conf_map, 0.85)
                        entropy_map = np.minimum(entropy_map, 0.15)

                    valid_slice = valid_crop[y0:y1, x0:x1]
                    patch_valid_pixels = pred_mask[valid_slice]
                    if patch_valid_pixels.size:
                        patch_labels[global_idx] = int(np.bincount(patch_valid_pixels.reshape(-1), minlength=len(record.class_names)).argmax())
                        patch_confidences[global_idx] = float(conf_map[valid_slice].mean())
                        patch_entropies[global_idx] = float(entropy_map[valid_slice].mean())
                        class_pixel_counts += np.bincount(
                            patch_valid_pixels.reshape(-1),
                            minlength=len(record.class_names),
                        ).astype(np.int64)
                    else:
                        patch_labels[global_idx] = 0
                        patch_confidences[global_idx] = 0.0
                        patch_entropies[global_idx] = 1.0

                    target = full_label_map[y0:y1, x0:x1]
                    target[:] = pred_mask
                    target[~valid_slice] = NO_DATA_LABEL

        score, score_breakdown = self._score_unet_fit_from_metrics(
            record,
            max_confidences=patch_confidences,
            normalized_entropies=patch_entropies,
            handcrafted_features=handcrafted,
            class_count=len(record.class_names),
        )
        class_percentages = class_percentage_table_from_counts(class_pixel_counts, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(tiled.rows, tiled.cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
            full_label_map=full_label_map,
        )

    def _predict_with_patch_classifier_streaming(self, loaded_image: LoadedImage, record: ModelRecord) -> ModelPrediction:
        bundle = record.load_bundle()
        config = bundle.get("config", {})
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))
        classifier = bundle["classifier"]
        pca = bundle["pca"]
        kmeans = bundle["kmeans"]

        patch_size = record.patch_size
        rows = int(np.ceil(loaded_image.height / patch_size))
        cols = int(np.ceil(loaded_image.width / patch_size))
        total_patches = rows * cols

        patch_labels = np.zeros(total_patches, dtype=np.uint8)
        patch_confidences = np.zeros(total_patches, dtype=np.float32)
        patch_coverages = np.zeros(total_patches, dtype=np.int64)
        max_confidences: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        separations: list[np.ndarray] = []
        handcrafted_batches: list[np.ndarray] = []
        overrides = 0

        batch_size = max(1, int(self.settings.prediction_batch_patches))
        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)

        with rasterio.open(loaded_image.source_path) as src:
            batch_patches: list[np.ndarray] = []
            batch_means: list[float] = []
            batch_stds: list[float] = []
            batch_coverages: list[int] = []
            batch_indices: list[int] = []

            def flush_batch() -> None:
                nonlocal overrides
                if not batch_patches:
                    return

                patches = np.stack(batch_patches).astype(np.float32)
                feature_matrix = compute_patch_features(
                    patches,
                    resize_for_features=resize_for_features,
                    canny_low=canny_low,
                    canny_high=canny_high,
                )
                handcrafted = feature_matrix[:, -7:]
                reduced = pca.transform(feature_matrix)
                probabilities = classifier.predict_proba(reduced)
                predicted_indices = probabilities.argmax(axis=1)
                predicted = classifier.classes_[predicted_indices].astype(np.uint8)
                confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices].astype(np.float32)

                for local_idx, flat_idx in enumerate(batch_indices):
                    if self._should_override_dark_patch(
                        record,
                        float(batch_means[local_idx]),
                        float(batch_stds[local_idx]),
                    ):
                        if predicted[local_idx] != 0:
                            overrides += 1
                        predicted[local_idx] = 0
                        confidences[local_idx] = max(float(confidences[local_idx]), 0.85)

                    patch_labels[flat_idx] = predicted[local_idx]
                    patch_confidences[flat_idx] = confidences[local_idx]
                    patch_coverages[flat_idx] = batch_coverages[local_idx]

                max_confidences.append(confidences)
                entropies.append(normalized_entropy(probabilities).astype(np.float32))
                separations.append(cluster_separation_values(reduced, kmeans).astype(np.float32))
                handcrafted_batches.append(handcrafted.astype(np.float32))

                batch_patches.clear()
                batch_means.clear()
                batch_stds.clear()
                batch_coverages.clear()
                batch_indices.clear()

            for row in range(rows):
                for col in range(cols):
                    y0 = row * patch_size
                    x0 = col * patch_size
                    patch, valid_mask = read_normalized_raster_patch(
                        src,
                        x0=x0,
                        y0=y0,
                        patch_size=patch_size,
                        normalization_low=normalization_low,
                        normalization_high=normalization_high,
                        cloud_mask=loaded_image.cloud_mask,
                    )
                    batch_patches.append(patch)
                    batch_means.append(float(patch.mean()))
                    batch_stds.append(float(patch.std()))
                    batch_coverages.append(int(valid_mask.sum()))
                    batch_indices.append(row * cols + col)
                    if len(batch_patches) >= batch_size:
                        flush_batch()

            flush_batch()

        handcrafted_features = np.vstack(handcrafted_batches)
        score, score_breakdown = self._score_model_fit_from_metrics(
            record,
            max_confidences=np.concatenate(max_confidences),
            normalized_entropies=np.concatenate(entropies),
            separations=np.concatenate(separations),
            handcrafted_features=handcrafted_features,
            class_count=len(classifier.classes_),
        )

        class_percentages = class_percentage_table(patch_labels, patch_coverages, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(rows, cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
        )

    def _predict_with_dense_unet_streaming(self, loaded_image: LoadedImage, record: ModelRecord) -> ModelPrediction:
        model, device = self._get_unet_model(record)
        config = record.model_config
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        patch_size = record.patch_size
        source_patch_size, effective_gsd, scale = self._raster_inference_geometry(loaded_image, patch_size)
        rows = int(np.ceil(loaded_image.height / source_patch_size))
        cols = int(np.ceil(loaded_image.width / source_patch_size))
        total_patches = rows * cols

        patch_labels = np.zeros(total_patches, dtype=np.uint8)
        patch_confidences = np.zeros(total_patches, dtype=np.float32)
        patch_coverages = np.zeros(total_patches, dtype=np.int64)
        patch_entropies = np.zeros(total_patches, dtype=np.float32)
        class_pixel_counts = np.zeros(len(record.class_names), dtype=np.int64)
        handcrafted_batches: list[np.ndarray] = []
        overrides = 0

        full_label_map = np.full((loaded_image.height, loaded_image.width), NO_DATA_LABEL, dtype=np.uint8)
        batch_size = max(1, int(self.settings.prediction_batch_patches))
        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)
        cloud_threshold = self._cloud_threshold(loaded_image)
        enhance_local_contrast = self._use_local_contrast(loaded_image, scale)
        loaded_image.source_metadata["inference_source_patch_size"] = source_patch_size
        loaded_image.source_metadata["inference_target_gsd_m"] = round(effective_gsd, 4) if effective_gsd is not None else None
        loaded_image.source_metadata["inference_scale_factor"] = round(scale, 4)
        loaded_image.source_metadata["local_contrast_applied"] = enhance_local_contrast
        loaded_image.source_metadata["cloud_masking_applied"] = (
            loaded_image.cloud_mask.has_clouds if loaded_image.cloud_mask is not None else cloud_threshold is not None
        )

        with rasterio.open(loaded_image.source_path) as src:
            batch_patches: list[np.ndarray] = []
            batch_means: list[float] = []
            batch_stds: list[float] = []
            batch_coverages: list[int] = []
            batch_indices: list[int] = []
            batch_valid_masks: list[np.ndarray] = []

            def flush_batch() -> None:
                nonlocal overrides, class_pixel_counts
                if not batch_patches:
                    return

                patches = np.stack(batch_patches).astype(np.float32)
                feature_matrix = compute_patch_features(
                    patches,
                    resize_for_features=resize_for_features,
                    canny_low=canny_low,
                    canny_high=canny_high,
                )
                handcrafted_batches.append(feature_matrix[:, -7:].astype(np.float32))

                with torch.no_grad():
                    batch = torch.from_numpy(patches).unsqueeze(1).float().to(device)
                    logits = model(batch)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()

                predicted_masks = probabilities.argmax(axis=1).astype(np.uint8)
                confidence_maps = probabilities.max(axis=1).astype(np.float32)
                entropy_maps = normalized_entropy_map(probabilities).astype(np.float32)

                for local_idx, flat_idx in enumerate(batch_indices):
                    row = flat_idx // cols
                    col = flat_idx % cols
                    y0 = row * source_patch_size
                    x0 = col * source_patch_size
                    y1 = min(y0 + source_patch_size, loaded_image.height)
                    x1 = min(x0 + source_patch_size, loaded_image.width)
                    output_height = y1 - y0
                    output_width = x1 - x0
                    model_valid = batch_valid_masks[local_idx]
                    pred_mask = predicted_masks[local_idx]
                    conf_map = confidence_maps[local_idx]
                    entropy_map = entropy_maps[local_idx]

                    if self._should_override_dark_patch(
                        record,
                        float(batch_means[local_idx]),
                        float(batch_stds[local_idx]),
                    ):
                        if np.any(pred_mask != 0):
                            overrides += 1
                        pred_mask = np.zeros_like(pred_mask, dtype=np.uint8)
                        conf_map = np.maximum(conf_map, 0.85)
                        entropy_map = np.minimum(entropy_map, 0.15)

                    pred_source = cv2.resize(
                        pred_mask,
                        (output_width, output_height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
                    valid_source = cv2.resize(
                        model_valid.astype(np.uint8),
                        (output_width, output_height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    full_label_map[y0:y1, x0:x1] = pred_source
                    full_label_map[y0:y1, x0:x1][~valid_source] = NO_DATA_LABEL

                    valid_pixels = pred_source[valid_source]
                    patch_coverages[flat_idx] = int(valid_source.sum())
                    if valid_pixels.size:
                        patch_labels[flat_idx] = int(np.bincount(valid_pixels.reshape(-1), minlength=len(record.class_names)).argmax())
                        patch_confidences[flat_idx] = float(conf_map[model_valid].mean())
                        patch_entropies[flat_idx] = float(entropy_map[model_valid].mean())
                        class_pixel_counts += np.bincount(valid_pixels.reshape(-1), minlength=len(record.class_names)).astype(np.int64)
                    else:
                        patch_labels[flat_idx] = 0
                        patch_confidences[flat_idx] = 0.0
                        patch_entropies[flat_idx] = 1.0

                batch_patches.clear()
                batch_means.clear()
                batch_stds.clear()
                batch_coverages.clear()
                batch_indices.clear()
                batch_valid_masks.clear()

            for row in range(rows):
                for col in range(cols):
                    y0 = row * source_patch_size
                    x0 = col * source_patch_size
                    patch, patch_valid_mask = read_normalized_raster_patch(
                        src,
                        x0=x0,
                        y0=y0,
                        patch_size=patch_size,
                        normalization_low=normalization_low,
                        normalization_high=normalization_high,
                        source_window_size=source_patch_size,
                        cloud_threshold=cloud_threshold,
                        cloud_mask=loaded_image.cloud_mask,
                        enhance_local_contrast=enhance_local_contrast,
                    )
                    batch_patches.append(patch)
                    batch_means.append(float(patch.mean()))
                    batch_stds.append(float(patch.std()))
                    batch_coverages.append(int(patch_valid_mask.sum()))
                    batch_indices.append(row * cols + col)
                    batch_valid_masks.append(patch_valid_mask)
                    if len(batch_patches) >= batch_size:
                        flush_batch()

            flush_batch()

        handcrafted_features = np.vstack(handcrafted_batches)
        score, score_breakdown = self._score_unet_fit_from_metrics(
            record,
            max_confidences=patch_confidences,
            normalized_entropies=patch_entropies,
            handcrafted_features=handcrafted_features,
            class_count=len(record.class_names),
        )
        class_percentages = class_percentage_table_from_counts(class_pixel_counts, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(rows, cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
            full_label_map=full_label_map,
        )

    def _predict_pan_overlap(self, loaded_image: LoadedImage, record: ModelRecord, progress=None) -> ModelPrediction:
        model, device = self._get_unet_model(record)
        patch_size = record.patch_size
        source_size, effective_gsd, _ = self._raster_inference_geometry(loaded_image, patch_size)
        scale = source_size/float(patch_size)
        width, height = int(np.ceil(loaded_image.width/scale)), int(np.ceil(loaded_image.height/scale))
        rows, cols = int(np.ceil(height/patch_size)), int(np.ceil(width/patch_size))
        classes = len(record.class_names)
        labels = np.full((loaded_image.height, loaded_image.width), NO_DATA_LABEL, np.uint8)
        sums = np.zeros((rows,cols),np.float64)
        entropies = np.zeros_like(sums)
        counts = np.zeros_like(sums)
        features = []
        config = record.summary.get("inference", {})
        minimum = float(config.get("minimum_probability", .40))
        margin_minimum = float(config.get("minimum_margin", .05))
        uncertain_pixels = 0
        stride = int(config.get("stride", 192))
        loaded_image.source_metadata.update(
            inference_pipeline="pan_probability_overlap_v8", inference_source_patch_size=source_size,
            inference_target_gsd_m=effective_gsd, inference_scale_factor=scale,
            inference_patch_stride=stride, base_probability_blending="positive_hann_global_lattice",
            local_contrast_applied=False, edge_padding="native_extent_then_edge_pad",
            uncertainty_minimum_probability=minimum, uncertainty_minimum_margin=margin_minimum)
        with rasterio.open(loaded_image.source_path) as source:
            def read_patch(x,y):
                return read_normalized_raster_patch(
                    source, x0=int(round(x*scale)), y0=int(round(y*scale)), patch_size=patch_size,
                    source_window_size=source_size, normalization_low=float(loaded_image.normalization_low),
                    normalization_high=float(loaded_image.normalization_high),
                    cloud_mask=loaded_image.cloud_mask, cloud_threshold=self._cloud_threshold(loaded_image),
                    enhance_local_contrast=False)

            def predict(patches):
                if sum(len(part) for part in features) < 128:
                    features.append(compute_patch_features(patches,32,60,160)[:,-7:])
                with torch.inference_mode():
                    return model(torch.from_numpy(patches[:,None]).float().to(device)).softmax(1).cpu().numpy()

            def emit(y, probabilities, valid):
                nonlocal uncertain_pixels
                confidence = probabilities.max(0)
                sorted_probabilities = np.sort(probabilities,axis=0)
                margin = sorted_probabilities[-1]-sorted_probabilities[-2]
                result = probabilities.argmax(0).astype(np.uint8)
                supported = valid & (confidence >= minimum) & (margin >= margin_minimum)
                uncertain_pixels += int(np.count_nonzero(valid & (~supported)))
                result[~supported] = NO_DATA_LABEL
                y0,y1 = int(round(y*scale)), min(loaded_image.height,int(round((y+result.shape[0])*scale)))
                if y1>y0:
                    output = cv2.resize(result,(loaded_image.width,y1-y0),interpolation=cv2.INTER_NEAREST)
                    source_valid = read_raster_valid_mask(source,x0=0,y0=y0,width=loaded_image.width,height=y1-y0)
                    output[~source_valid] = NO_DATA_LABEL
                    labels[y0:y1] = output
                entropy = -np.sum(probabilities*np.log(np.maximum(probabilities,1e-8)),axis=0)/np.log(classes)
                for row in range(y//patch_size, (y+result.shape[0]-1)//patch_size+1):
                    local_y0,local_y1 = max(0,row*patch_size-y), min(result.shape[0],(row+1)*patch_size-y)
                    for col in range(cols):
                        region = np.s_[local_y0:local_y1,col*patch_size:min(width,(col+1)*patch_size)]
                        mask = valid[region]
                        counts[row,col] += mask.sum()
                        sums[row,col] += confidence[region][mask].sum()
                        entropies[row,col] += entropy[region][mask].sum()

            blend_probabilities(width,height,classes,read_patch,predict,emit,window=patch_size,
                                stride=stride,batch_size=min(32,self.settings.prediction_batch_patches),
                                progress=(lambda value: progress(.10+.65*value,"Blending overlapping PAN probabilities")) if progress else None)
        loaded_image.source_metadata["uncertain_model_pixels"] = uncertain_pixels
        confidences = np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0).astype(np.float32).ravel()
        entropy_values = np.divide(entropies,counts,out=np.ones_like(entropies),where=counts>0).astype(np.float32).ravel()
        score, breakdown = self._score_unet_fit_from_metrics(
            record,max_confidences=confidences,normalized_entropies=entropy_values,
            handcrafted_features=np.vstack(features),class_count=classes)
        patch_labels = np.zeros(rows*cols,np.uint8)
        prediction = ModelPrediction(record=record,patch_grid=patch_labels.reshape(rows,cols),
                                     patch_confidences=confidences,patch_labels=patch_labels,
                                     patch_coverages=np.zeros(rows*cols,np.int64),class_percentages=[],
                                     selected_score=score,score_breakdown=breakdown,heuristic_water_overrides=0,
                                     dominant_class_name="Unknown",full_label_map=labels)
        self._refresh_prediction_percentages(prediction)
        return prediction

    def _predict_pan_texture(self, loaded_image: LoadedImage, record: ModelRecord, progress=None) -> ModelPrediction:
        from threadpoolctl import threadpool_limits
        bundle=record.load_bundle()
        if bundle["features"] != PAN_TEXTURE_FEATURES or list(bundle["classifier"].classes_) != [0,2,3,4]:
            raise ValueError("Incompatible PAN texture model feature or class schema")
        texture_predict=bundle["classifier"].predict_proba
        teacher_fallback_pixels=0
        if "student_state" in bundle:
            cached=self._torch_models.get(record.id)
            if cached is None:
                state={key:torch.from_numpy(value) for key,value in bundle["student_state"].items()}
                student=PanTextureNet(state["mean"],state["scale"],
                                      hidden=(state["layers.0.weight"].shape[0],state["layers.3.weight"].shape[0]))
                student.load_state_dict(state)
                target=resolve_torch_device()
                student.to(target).eval()
                self._torch_models[record.id]=(student,target)
            student,target=self._torch_models[record.id]
            def texture_predict(matrix):
                nonlocal teacher_fallback_pixels
                parts=[]
                with torch.inference_mode():
                    for start in range(0,len(matrix),65536):
                        batch=torch.from_numpy(np.ascontiguousarray(matrix[start:start+65536])).float().to(target)
                        parts.append(student(batch).softmax(1).cpu().numpy())
                probability=np.concatenate(parts)
                fallback=probability.max(1)<float(bundle.get("teacher_fallback_threshold",0))
                if fallback.any():
                    probability[fallback]=bundle["classifier"].predict_proba(matrix[fallback])
                    teacher_fallback_pixels+=int(fallback.sum())
                return probability
        patch_size=record.patch_size
        source_size,effective_gsd,_=self._raster_inference_geometry(loaded_image,patch_size)
        scale=source_size/float(patch_size)
        width,height=int(np.ceil(loaded_image.width/scale)),int(np.ceil(loaded_image.height/scale))
        rows,cols=int(np.ceil(height/patch_size)),int(np.ceil(width/patch_size))
        labels=np.full((loaded_image.height,loaded_image.width),255,np.uint8)
        sums=np.zeros((rows,cols),np.float64)
        amounts=np.zeros_like(sums)
        entropy_sums=np.zeros_like(sums)
        class_ids=np.array([0,2,3,4],np.uint8)
        config=record.summary.get("inference",{})
        minimum=float(config.get("minimum_probability",.55))
        margin_minimum=float(config.get("minimum_margin",.12))
        sampling_stride=int(config.get("classifier_sampling_stride",2))
        if sampling_stride not in (1,2):
            raise ValueError("PAN classifier sampling stride must be 1 or 2")
        uncertain=0
        loaded_image.source_metadata.update(
            inference_pipeline=config.get("pipeline","pan_texture_context_v8"),inference_source_patch_size=source_size,
            inference_target_gsd_m=effective_gsd,inference_scale_factor=scale,
            contextual_feature_halo_pixels=PAN_TEXTURE_HALO,model_features_use_coordinates=False,
            local_contrast_applied=False,uncertainty_minimum_probability=minimum,
            uncertainty_minimum_margin=margin_minimum,
            land_classifier_evaluation_spacing_m=(effective_gsd*sampling_stride) if effective_gsd else None,
            land_feature_grid_gsd_m=effective_gsd,
            inference_backend=bundle.get("inference_backend","cpu_hist_gradient_boosting"))
        block_size=512
        with rasterio.open(loaded_image.source_path) as source,threadpool_limits(limits=6,user_api="openmp"):
            if source.count!=1:
                raise ValueError("The PAN branch needs a single-band TIFF; four-band 0.6 m products use their separate model.")
            for y in range(0,height,block_size):
                h=min(block_size,height-y)
                for x in range(0,width,block_size):
                    w=min(block_size,width-x)
                    left,top=max(0,x-PAN_TEXTURE_HALO),max(0,y-PAN_TEXTURE_HALO)
                    right,bottom=min(width,x+w+PAN_TEXTURE_HALO),min(height,y+h+PAN_TEXTURE_HALO)
                    window=rasterio.windows.Window(left*scale,top*scale,
                                                   min(source.width-left*scale,(right-left)*scale),
                                                   min(source.height-top*scale,(bottom-top)*scale))
                    data=source.read(1,window=window,out_shape=(bottom-top,right-left),
                                     resampling=rasterio.enums.Resampling.bilinear,masked=True)
                    raw=np.asarray(data.filled(0),dtype=np.float32)
                    valid=(~np.ma.getmaskarray(data))&np.isfinite(raw)
                    if loaded_image.cloud_mask is not None:
                        valid &= loaded_image.cloud_mask.window(int(round(left*scale)),int(round(top*scale)),
                                  int(round(window.width)),int(round(window.height)),raw.shape)==0
                    core_valid=valid[y-top:y-top+h,x-left:x-left+w]
                    if not core_valid.any():
                        continue
                    image=np.clip((raw-float(loaded_image.normalization_low))/
                                  max(float(loaded_image.normalization_high-loaded_image.normalization_low),1e-6),0,1)
                    image[~valid]=float(np.median(image[valid])) if valid.any() else .5
                    padding=((PAN_TEXTURE_HALO-(y-top),PAN_TEXTURE_HALO-(bottom-y-h)),
                             (PAN_TEXTURE_HALO-(x-left),PAN_TEXTURE_HALO-(right-x-w)))
                    image=np.pad(image,padding,mode="reflect" if min(image.shape)>1 else "edge")
                    all_features=texture_features(image)
                    if sampling_stride==1:
                        feature=all_features[PAN_TEXTURE_HALO:PAN_TEXTURE_HALO+h,PAN_TEXTURE_HALO:PAN_TEXTURE_HALO+w]
                        matrix=feature.reshape(-1,len(PAN_TEXTURE_FEATURES))
                        probability=np.zeros((h*w,4),np.float32)
                        probability[core_valid.ravel()]=texture_predict(np.ascontiguousarray(matrix[core_valid.ravel()]))
                        probability=probability.reshape(h,w,4)
                    else:
                        # Features retain the model's 2.5 m support. Evaluate on a
                        # global 5 m lattice, interpolate probabilities (not labels),
                        # and include context samples across every block boundary.
                        sampled=all_features[::sampling_stride,::sampling_stride]
                        coarse=texture_predict(np.ascontiguousarray(sampled.reshape(-1,len(PAN_TEXTURE_FEATURES))))
                        coarse=coarse.reshape(*sampled.shape[:2],4).astype(np.float32)
                        yy,xx=np.indices((h,w),dtype=np.float32)
                        probability=cv2.remap(coarse,(xx+PAN_TEXTURE_HALO)/sampling_stride,
                                              (yy+PAN_TEXTURE_HALO)/sampling_stride,
                                              interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                    confidence=probability.max(-1)
                    ordered=np.sort(probability,axis=-1)
                    supported=core_valid&(confidence>=minimum)&((ordered[:,:,-1]-ordered[:,:,-2])>=margin_minimum)
                    output=class_ids[probability.argmax(-1)]
                    output[~supported]=255
                    uncertain+=int((core_valid&(~supported)).sum())
                    x0,y0=int(round(x*scale)),int(round(y*scale))
                    x1,y1=min(source.width,int(round((x+w)*scale))),min(source.height,int(round((y+h)*scale)))
                    native=cv2.resize(output,(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
                    native_valid=read_raster_valid_mask(source,x0=x0,y0=y0,width=x1-x0,height=y1-y0)
                    native[~native_valid]=255
                    labels[y0:y1,x0:x1]=native
                    entropy=-np.sum(probability*np.log(np.maximum(probability,1e-8)),axis=-1)/np.log(4)
                    for row in range(y//patch_size,(y+h-1)//patch_size+1):
                        for col in range(x//patch_size,(x+w-1)//patch_size+1):
                            region=np.s_[max(0,row*patch_size-y):min(h,(row+1)*patch_size-y),
                                         max(0,col*patch_size-x):min(w,(col+1)*patch_size-x)]
                            mask=core_valid[region]
                            sums[row,col]+=confidence[region][mask].sum()
                            entropy_sums[row,col]+=entropy[region][mask].sum()
                            amounts[row,col]+=mask.sum()
                if progress:
                    progress(.10+.65*min(1,(y+h)/height),"Classifying PAN multi-scale texture with halo-safe blocks")
        confidence=np.divide(sums,amounts,out=np.zeros_like(sums),where=amounts>0).astype(np.float32).ravel()
        entropy=np.divide(entropy_sums,amounts,out=np.ones_like(sums),where=amounts>0).astype(np.float32).ravel()
        loaded_image.source_metadata["uncertain_model_pixels"]=uncertain
        loaded_image.source_metadata["teacher_fallback_pixels"]=teacher_fallback_pixels
        loaded_image.source_metadata["teacher_fallback_threshold"]=bundle.get("teacher_fallback_threshold")
        score=float(np.average(confidence,weights=amounts.ravel())) if amounts.sum() else 0.
        breakdown={"confidence_mean":score,"certainty":float(1-entropy.mean()),"final_score":score,
                   "scope":"mean model confidence, not calibrated accuracy or geographic fit"}
        patch_labels=np.zeros(rows*cols,np.uint8)
        prediction=ModelPrediction(record=record,patch_grid=patch_labels.reshape(rows,cols),patch_labels=patch_labels,
                                   patch_confidences=confidence,patch_coverages=np.zeros(rows*cols,np.int64),
                                   class_percentages=[],selected_score=score,score_breakdown=breakdown,
                                   heuristic_water_overrides=0,dominant_class_name="Unknown",full_label_map=labels)
        self._refresh_prediction_percentages(prediction)
        return prediction

    def _predict_with_unet_streaming(self, loaded_image: LoadedImage, record: ModelRecord) -> ModelPrediction:
        model, device = self._get_unet_model(record)
        config = record.model_config
        resize_for_features = int(config.get("resize_for_features", 32))
        canny_low = int(config.get("canny_low", 60))
        canny_high = int(config.get("canny_high", 160))

        patch_size = record.patch_size
        rows = int(np.ceil(loaded_image.height / patch_size))
        cols = int(np.ceil(loaded_image.width / patch_size))
        total_patches = rows * cols

        patch_labels = np.zeros(total_patches, dtype=np.uint8)
        patch_confidences = np.zeros(total_patches, dtype=np.float32)
        patch_coverages = np.zeros(total_patches, dtype=np.int64)
        max_confidences: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        handcrafted_batches: list[np.ndarray] = []
        overrides = 0

        batch_size = max(1, int(self.settings.prediction_batch_patches))
        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)

        with rasterio.open(loaded_image.source_path) as src:
            batch_patches: list[np.ndarray] = []
            batch_means: list[float] = []
            batch_stds: list[float] = []
            batch_coverages: list[int] = []
            batch_indices: list[int] = []

            def flush_batch() -> None:
                nonlocal overrides
                if not batch_patches:
                    return

                patches = np.stack(batch_patches).astype(np.float32)
                feature_matrix = compute_patch_features(
                    patches,
                    resize_for_features=resize_for_features,
                    canny_low=canny_low,
                    canny_high=canny_high,
                )
                handcrafted = feature_matrix[:, -7:]
                handcrafted_batches.append(handcrafted.astype(np.float32))

                with torch.no_grad():
                    batch = torch.from_numpy(patches).unsqueeze(1).float().to(device)
                    logits = model(batch)
                    patch_logits = logits.mean(dim=(2, 3))
                    probabilities = torch.softmax(patch_logits, dim=1).cpu().numpy()

                predicted_indices = probabilities.argmax(axis=1)
                predicted = predicted_indices.astype(np.uint8)
                confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices].astype(np.float32)

                for local_idx, flat_idx in enumerate(batch_indices):
                    if self._should_override_dark_patch(
                        record,
                        float(batch_means[local_idx]),
                        float(batch_stds[local_idx]),
                    ):
                        if predicted[local_idx] != 0:
                            overrides += 1
                        predicted[local_idx] = 0
                        confidences[local_idx] = max(float(confidences[local_idx]), 0.85)

                    patch_labels[flat_idx] = predicted[local_idx]
                    patch_confidences[flat_idx] = confidences[local_idx]
                    patch_coverages[flat_idx] = batch_coverages[local_idx]

                max_confidences.append(confidences)
                entropies.append(normalized_entropy(probabilities).astype(np.float32))

                batch_patches.clear()
                batch_means.clear()
                batch_stds.clear()
                batch_coverages.clear()
                batch_indices.clear()

            for row in range(rows):
                for col in range(cols):
                    y0 = row * patch_size
                    x0 = col * patch_size
                    patch, valid_mask = read_normalized_raster_patch(
                        src,
                        x0=x0,
                        y0=y0,
                        patch_size=patch_size,
                        normalization_low=normalization_low,
                        normalization_high=normalization_high,
                        cloud_mask=loaded_image.cloud_mask,
                    )
                    batch_patches.append(patch)
                    batch_means.append(float(patch.mean()))
                    batch_stds.append(float(patch.std()))
                    batch_coverages.append(int(valid_mask.sum()))
                    batch_indices.append(row * cols + col)
                    if len(batch_patches) >= batch_size:
                        flush_batch()

            flush_batch()

        handcrafted_features = np.vstack(handcrafted_batches)
        score, score_breakdown = self._score_unet_fit_from_metrics(
            record,
            max_confidences=np.concatenate(max_confidences),
            normalized_entropies=np.concatenate(entropies),
            handcrafted_features=handcrafted_features,
            class_count=len(record.class_names),
        )
        class_percentages = class_percentage_table(patch_labels, patch_coverages, record.class_names)
        dominant_class_name = class_percentages[0]["class_name"] if class_percentages else "Unknown"

        return ModelPrediction(
            record=record,
            patch_grid=patch_labels.reshape(rows, cols),
            patch_confidences=patch_confidences,
            patch_labels=patch_labels,
            patch_coverages=patch_coverages,
            class_percentages=class_percentages,
            selected_score=score,
            score_breakdown=score_breakdown,
            heuristic_water_overrides=overrides,
            dominant_class_name=dominant_class_name,
        )

    def _write_geotiff_labels(
        self,
        destination: Path,
        patch_grid: np.ndarray,
        patch_size: int,
        loaded_image: LoadedImage,
        full_label_map: np.ndarray | None = None,
    ) -> None:
        if loaded_image.raster_profile is None:
            return

        profile = loaded_image.raster_profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=NO_DATA_LABEL,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        height = loaded_image.height
        width = loaded_image.width

        with rasterio.open(destination, "w", **profile) as dst:
            if full_label_map is not None:
                label_crop = full_label_map[:height, :width].astype(np.uint8)
                dst.write(label_crop, 1)
                return
            src = rasterio.open(loaded_image.source_path)
            try:
                for row_index in range(patch_grid.shape[0]):
                    y0 = row_index * patch_size
                    y1 = min((row_index + 1) * patch_size, height)
                    row_block = np.repeat(patch_grid[row_index : row_index + 1, :], patch_size, axis=0)
                    row_block = np.repeat(row_block, patch_size, axis=1)
                    row_block = row_block[: y1 - y0, :width].astype(np.uint8)
                    if loaded_image.valid_mask is not None:
                        mask_slice = loaded_image.valid_mask[y0:y1, :width]
                    else:
                        mask_slice = read_raster_valid_mask(
                            src,
                            x0=0,
                            y0=y0,
                            width=width,
                            height=y1 - y0,
                        )
                    row_block = row_block.copy()
                    row_block[~mask_slice] = NO_DATA_LABEL
                    window = rasterio.windows.Window(col_off=0, row_off=y0, width=width, height=y1 - y0)
                    dst.write(row_block, 1, window=window)
            finally:
                src.close()

    def _write_class_shapefiles(
        self,
        geotiff_path: Path,
        destination_dir: Path,
        record: ModelRecord,
    ) -> tuple[Path | None, list[dict[str, Any]]]:
        ensure_dir(destination_dir)

        with rasterio.open(geotiff_path) as src:
            if src.crs is None:
                return None, []

            label_image = src.read(1)
            valid_mask = label_image != NO_DATA_LABEL
            if not np.any(valid_mask):
                return None, []

            transform = src.transform
            crs = src.crs
            pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
            vector_label_image = label_image
            sieve_pixels = max(0, int(self.settings.vector_sieve_pixels))
            if sieve_pixels > 0:
                vector_label_image = rasterio_features.sieve(
                    label_image,
                    size=sieve_pixels,
                    mask=valid_mask & (label_image != 1),
                    connectivity=8,
                ).astype(np.uint8)
                vector_label_image[label_image == 1] = 1
                vector_label_image[~valid_mask] = NO_DATA_LABEL

            class_features: dict[int, list[dict[str, Any]]] = {
                int(class_id): [] for class_id in record.class_names
            }
            for geometry, value in rasterio_features.shapes(
                vector_label_image,
                mask=valid_mask,
                transform=transform,
            ):
                class_id = int(value)
                if class_id == NO_DATA_LABEL or class_id not in record.class_names:
                    continue
                polygon = shapely_shape(geometry)
                if polygon.is_empty:
                    continue
                class_features[class_id].append(
                    {
                        "cls_id": class_id,
                        "cls_name": record.class_names[class_id],
                        "geometry": polygon,
                    }
                )

        written_layers: list[dict[str, Any]] = []
        class_summary_path = destination_dir / "vector_class_summary.csv"
        with class_summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_id", "class_name", "feature_count", "pixel_count", "area_estimate", "sieve_pixels"])

            for class_id, rows in class_features.items():
                class_name = record.class_names[class_id]
                pixel_count = int(np.count_nonzero(vector_label_image == class_id))
                area_estimate = float(pixel_count * pixel_area)
                class_sieve_pixels = 0 if class_id == 1 else sieve_pixels
                writer.writerow([class_id, class_name, len(rows), pixel_count, round(area_estimate, 4), class_sieve_pixels])

                if not rows:
                    continue

                class_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in class_name).strip("_")
                while "__" in class_slug:
                    class_slug = class_slug.replace("__", "_")
                layer_dir = ensure_dir(destination_dir / f"class_{class_id:02d}_{class_slug}")
                shapefile_path = layer_dir / f"class_{class_id:02d}_{class_slug}.shp"

                gdf = gpd.GeoDataFrame(rows, crs=crs)
                gdf["cls_pix"] = pixel_count
                gdf["area_ha"] = (gdf.geometry.area / 10_000.0).round(6)
                gdf = gdf[["cls_id", "cls_name", "cls_pix", "area_ha", "geometry"]]
                gdf.to_file(shapefile_path, driver="ESRI Shapefile")

                written_layers.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "feature_count": len(rows),
                        "pixel_count": pixel_count,
                        "area_estimate": round(area_estimate, 4),
                        "sieve_pixels": class_sieve_pixels,
                        "shapefile": str(shapefile_path.relative_to(destination_dir)),
                    }
                )

        zip_path = destination_dir.parent / "class_shapefiles_georef.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(destination_dir.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(destination_dir.parent))

        return zip_path, written_layers

    def _validate_vector_export(
        self,
        zip_path: Path | None,
        destination_dir: Path,
        vector_layers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        validation = {
            "ok": False,
            "zip_exists": False,
            "zip_integrity_ok": False,
            "layer_count": len(vector_layers),
            "missing_sidecars": [],
            "missing_prj": [],
            "schema_fields": ["cls_id", "cls_name", "cls_pix", "area_ha"],
            "message": "Vector export was not created.",
        }
        if zip_path is None:
            return validation
        validation["zip_exists"] = zip_path.exists()
        if not zip_path.exists():
            validation["message"] = "Vector zip path was returned but the file is missing."
            return validation

        required_exts = [".shp", ".shx", ".dbf", ".prj"]
        try:
            with zipfile.ZipFile(zip_path) as archive:
                if self.settings.validate_zip_integrity:
                    bad_file = archive.testzip()
                    validation["zip_integrity_ok"] = bad_file is None
                else:
                    validation["zip_integrity_ok"] = True
                    validation["zip_integrity_check"] = "skipped_fast_mode"
                names = set(archive.namelist())
                for layer in vector_layers:
                    shapefile_rel = Path("class_shapefiles") / str(layer["shapefile"])
                    base = shapefile_rel.with_suffix("").as_posix()
                    missing = [ext for ext in required_exts if f"{base}{ext}" not in names]
                    if missing:
                        validation["missing_sidecars"].append(
                            {
                                "layer": layer["class_name"],
                                "missing": missing,
                            }
                        )
                    if f"{base}.prj" in names:
                        prj_text = archive.read(f"{base}.prj").decode("utf-8", errors="ignore").strip()
                        if not prj_text:
                            validation["missing_prj"].append(layer["class_name"])
                validation["ok"] = (
                    validation["zip_exists"]
                    and validation["zip_integrity_ok"]
                    and not validation["missing_sidecars"]
                    and not validation["missing_prj"]
                    and validation["layer_count"] > 0
                )
                validation["message"] = "Vector shapefile package passed export checks." if validation["ok"] else "Vector shapefile package failed export checks."
        except Exception as exc:
                validation["message"] = f"Vector validation failed: {exc}"
        return validation

    def generate_shapefiles_for_result(self, result_id: str) -> dict[str, Any]:
        result_dir = (self.settings.result_dir / result_id).resolve()
        if self.settings.result_dir.resolve() not in result_dir.parents and result_dir != self.settings.result_dir.resolve():
            raise ValueError("Invalid result id.")
        if not result_dir.exists():
            raise ValueError(f"Result not found: {result_id}")

        summary_path = result_dir / "prediction_summary.json"
        geotiff_path = result_dir / "segmentation_labels_georef.tif"
        if not summary_path.exists():
            raise ValueError("Prediction summary is missing for this result.")
        if not geotiff_path.exists():
            raise ValueError("Georeferenced label GeoTIFF is missing for this result.")

        summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
        model_id = summary_json.get("selected_model", {}).get("id")
        if not model_id:
            raise ValueError("Selected model id is missing from this result summary.")
        record = self.registry.get(str(model_id))

        destination_dir = result_dir / "class_shapefiles"
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        existing_zip = result_dir / "class_shapefiles_georef.zip"
        if existing_zip.exists():
            existing_zip.unlink()

        vector_zip_path, vector_layers = self._write_class_shapefiles(
            geotiff_path=geotiff_path,
            destination_dir=destination_dir,
            record=record,
        )
        vector_validation = self._validate_vector_export(
            zip_path=vector_zip_path,
            destination_dir=destination_dir,
            vector_layers=vector_layers,
        )

        summary_json["vector_shapefiles"] = vector_layers
        summary_json["vector_export_validation"] = vector_validation
        summary_json["class_shapefiles_generated_at"] = datetime.now().isoformat(timespec="seconds")
        summary_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

        summary_text_path = result_dir / "RESULTS_SIMPLE.txt"
        if summary_text_path.exists():
            text = summary_text_path.read_text(encoding="utf-8")
            block = "\n".join(
                [
                    "",
                    "On-request vector export:",
                    "- class_shapefiles_georef.zip",
                    "- class_shapefiles/ folder with one georeferenced shapefile per detected class",
                    f"- Vector export validation: {'PASS' if vector_validation.get('ok') else 'CHECK'} ({vector_validation.get('message')})",
                ]
            )
            if "On-request vector export:" not in text:
                summary_text_path.write_text(text.rstrip() + "\n" + block + "\n", encoding="utf-8")

        return {
            "result_id": result_id,
            "class_shapefiles_zip_url": f"/results/{result_id}/{vector_zip_path.name}" if vector_zip_path else None,
            "vector_shapefiles": vector_layers,
            "vector_export_validation": vector_validation,
        }

    def _reference_road_vector_path(self, record: ModelRecord) -> Path | None:
        project_root = self.settings.root_dir.parent
        for source in record.summary.get("sources", []):
            road_path = source.get("road_vector_path")
            if not road_path:
                continue
            path = Path(road_path)
            if not path.is_absolute():
                path = (project_root / path).resolve()
            if path.exists():
                return path
        return None

    def _apply_reference_road_overlay(
        self,
        loaded_image: LoadedImage,
        prediction: ModelPrediction,
        road_buffer_m: float | None = None,
    ) -> int:
        if loaded_image.raster_profile is None or prediction.full_label_map is None:
            return 0
        transform = loaded_image.raster_profile.get("transform")
        crs = loaded_image.raster_profile.get("crs")
        if transform is None or crs is None or 1 not in prediction.record.class_names:
            return 0

        road_path = self._reference_road_vector_path(prediction.record)
        if road_path is None:
            return 0

        roads = gpd.read_file(road_path)
        roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
        if roads.empty:
            return 0
        if "L_TYPE" in roads.columns:
            existing_road_mask = roads["L_TYPE"].fillna("").astype(str).str.contains("existing road", case=False, regex=False)
            if existing_road_mask.any():
                roads = roads[existing_road_mask].copy()
        if roads.empty:
            return 0
        if roads.crs is None:
            roads = roads.set_crs(crs)
        elif str(roads.crs) != str(crs):
            roads = roads.to_crs(crs)

        bounds = rasterio.transform.array_bounds(loaded_image.height, loaded_image.width, transform)
        raster_box = shapely_box(bounds[0], bounds[1], bounds[2], bounds[3])
        try:
            indices = list(roads.sindex.query(raster_box, predicate="intersects"))
            roads = roads.iloc[indices].copy()
        except Exception:
            roads = roads[roads.intersects(raster_box)].copy()
        if roads.empty:
            return 0

        labels = prediction.full_label_map[: loaded_image.height, : loaded_image.width]
        valid_mask = labels != NO_DATA_LABEL
        non_water_mask = valid_mask & (labels != 0)
        urban_pixels = int(np.count_nonzero(labels == 2)) if 2 in prediction.record.class_names else 0
        valid_pixels = int(np.count_nonzero(valid_mask))
        target_share = 0.06
        # Mumbai DP-derived prior: roads are roughly 8-12% of the urban + road fabric,
        # but PAN-only overlays are capped to avoid drawing non-visible planning links.
        target_road_pixels = int((target_share / max(1.0 - target_share, 1e-6)) * urban_pixels)
        max_road_pixels = int(0.018 * valid_pixels)
        target_road_pixels = max(0, min(target_road_pixels, max_road_pixels))

        def buffer_in_raster_units(buffer_m: float) -> float:
            try:
                if getattr(crs, "is_geographic", False):
                    return buffer_m / 111_320.0
            except Exception:
                pass
            return buffer_m

        if road_buffer_m is not None:
            candidate_buffers = [float(road_buffer_m)]
        else:
            candidate_buffers = [2.0, 3.0, 4.0, 5.0, 6.0]

        selected_mask: np.ndarray | None = None
        selected_buffer = candidate_buffers[0]
        selected_road_pixels = -1
        best_distance = float("inf")

        for candidate_buffer_m in candidate_buffers:
            buffer_distance = buffer_in_raster_units(candidate_buffer_m)
            geoms = roads.geometry.buffer(buffer_distance) if buffer_distance > 0 else roads.geometry
            features = [(geom, 1) for geom in geoms if geom is not None and not geom.is_empty]
            if not features:
                continue
            road_mask = rasterio_features.rasterize(
                features,
                out_shape=(loaded_image.height, loaded_image.width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
            overlay_mask = road_mask & non_water_mask
            road_pixels = int(np.count_nonzero(overlay_mask))
            if road_pixels <= 0:
                continue
            distance = abs(road_pixels - target_road_pixels) if target_road_pixels else road_pixels
            if road_pixels <= max_road_pixels and distance < best_distance:
                selected_mask = overlay_mask
                selected_buffer = candidate_buffer_m
                selected_road_pixels = road_pixels
                best_distance = distance
            if target_road_pixels and road_pixels >= int(0.9 * target_road_pixels) and road_pixels <= max_road_pixels:
                selected_mask = overlay_mask
                selected_buffer = candidate_buffer_m
                selected_road_pixels = road_pixels
                break

        if selected_mask is None:
            return 0

        overlay_mask = selected_mask
        changed = int(np.count_nonzero(overlay_mask & (labels != 1)))
        labels[overlay_mask] = 1
        prediction.reference_road_overlay_pixels = changed
        prediction.road_buffer_meters = selected_buffer
        prediction.road_target_share_of_urban = target_share
        road_plus_urban = int(np.count_nonzero((labels == 1) | (labels == 2)))
        if road_plus_urban:
            achieved_share = float(np.count_nonzero(labels == 1) / road_plus_urban)
            prediction.road_ratio_calibration_applied = achieved_share >= 0.06 or selected_road_pixels > 0
        if changed:
            self._refresh_prediction_percentages(prediction)
        return changed

    def _apply_pan_statistical_calibration(
        self,
        loaded_image: LoadedImage,
        prediction: ModelPrediction,
        tile_size: int = 1024,
    ) -> bool:
        if loaded_image.raster_profile is None or prediction.full_label_map is None:
            return False
        if not {0, 2, 3, 4}.issubset(set(prediction.record.class_names)):
            return False

        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)
        labels = prediction.full_label_map[: loaded_image.height, : loaded_image.width]
        class_pixel_counts = np.zeros(len(prediction.record.class_names), dtype=np.int64)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        with rasterio.open(loaded_image.source_path) as src:
            for y0 in range(0, loaded_image.height, tile_size):
                for x0 in range(0, loaded_image.width, tile_size):
                    height = min(tile_size, loaded_image.height - y0)
                    width = min(tile_size, loaded_image.width - x0)
                    window = rasterio.windows.Window(x0, y0, width, height)
                    raw = src.read(1, window=window).astype(np.float32)
                    valid = np.isfinite(raw)
                    if src.nodata is not None:
                        valid &= raw != src.nodata
                    cloud_threshold = self._cloud_threshold(loaded_image)
                    if cloud_threshold is not None:
                        valid &= raw < cloud_threshold
                    model_road = labels[y0 : y0 + height, x0 : x0 + width] == 1

                    image = np.clip(
                        (raw - normalization_low) / max(normalization_high - normalization_low, 1e-6),
                        0,
                        1,
                    ).astype(np.float32)
                    contrast_image = clahe.apply(to_uint8(image)).astype(np.float32) / 255.0
                    contrast_image[~valid] = 0.0
                    texture_image = (0.45 * image + 0.55 * contrast_image).astype(np.float32)

                    raw_mean31 = cv2.blur(image, (31, 31))
                    raw_sq31 = cv2.blur(image * image, (31, 31))
                    raw_std31 = np.sqrt(np.clip(raw_sq31 - raw_mean31 * raw_mean31, 0, None))
                    raw_gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
                    raw_gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
                    raw_gradient = np.sqrt(raw_gx * raw_gx + raw_gy * raw_gy)

                    mean11 = cv2.blur(texture_image, (11, 11))
                    sq11 = cv2.blur(texture_image * texture_image, (11, 11))
                    std11 = np.sqrt(np.clip(sq11 - mean11 * mean11, 0, None))
                    mean31 = cv2.blur(texture_image, (31, 31))
                    sq31 = cv2.blur(texture_image * texture_image, (31, 31))
                    std31 = np.sqrt(np.clip(sq31 - mean31 * mean31, 0, None))
                    gx = cv2.Sobel(texture_image, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(texture_image, cv2.CV_32F, 0, 1, ksize=3)
                    gradient = np.sqrt(gx * gx + gy * gy)

                    valid_values = image[valid]
                    if valid_values.size:
                        valid_std11 = std11[valid]
                        valid_std31 = std31[valid]
                        valid_gradient = gradient[valid]
                        valid_raw_std31 = raw_std31[valid]
                        valid_raw_gradient = raw_gradient[valid]
                        p20, p38, p55 = np.percentile(valid_values, [20, 38, 55])
                        water_intensity_cut = float(min(0.46, max(0.24, p38)))
                        water_std_cut = float(min(0.095, max(0.040, np.percentile(valid_raw_std31, 48))))
                        water_gradient_cut = float(min(0.14, max(0.050, np.percentile(valid_raw_gradient, 48))))
                        urban_std11_cut = float(max(0.095, np.percentile(valid_std11, 72)))
                        urban_std31_cut = float(max(0.085, np.percentile(valid_std31, 70)))
                        urban_gradient_cut = float(max(0.16, np.percentile(valid_gradient, 76)))
                        barren_brightness_cut = float(max(0.36, p55))
                        barren_smooth_cut = float(min(0.13, max(0.07, np.percentile(valid_std31, 62))))
                        vegetation_upper_cut = float(min(0.58, max(0.42, np.percentile(valid_values, 64))))
                    else:
                        water_intensity_cut = 0.42
                        water_std_cut = 0.085
                        water_gradient_cut = 0.13
                        urban_std11_cut = 0.105
                        urban_std31_cut = 0.095
                        urban_gradient_cut = 0.18
                        barren_brightness_cut = 0.38
                        barren_smooth_cut = 0.105
                        vegetation_upper_cut = 0.55
                        p20 = 0.18

                    calibrated = np.full((height, width), 3, dtype=np.uint8)
                    water = (
                        (
                            ((image < water_intensity_cut) & (raw_std31 < water_std_cut) & (raw_gradient < water_gradient_cut))
                            | (
                                (image < min(0.50, water_intensity_cut + 0.08))
                                & (raw_std31 < max(0.11, water_std_cut * 1.35))
                                & (raw_gradient < max(0.17, water_gradient_cut * 1.30))
                            )
                        )
                        & valid
                    )
                    urban = (
                        ((std11 > urban_std11_cut) | (std31 > urban_std31_cut) | (gradient > urban_gradient_cut))
                        & (image > max(0.18, float(p20)))
                        & valid
                        & (~water)
                    )
                    vegetation = (
                        (image < vegetation_upper_cut)
                        & (std11 > 0.045)
                        & (std11 < max(0.13, urban_std11_cut * 1.10))
                        & (std31 > 0.035)
                        & valid
                        & (~water)
                        & (~urban)
                    )
                    barren = (
                        (image > barren_brightness_cut)
                        & (std31 < barren_smooth_cut)
                        & (gradient < max(0.17, urban_gradient_cut * 0.85))
                        & valid
                        & (~water)
                        & (~urban)
                        & (~vegetation)
                    )

                    calibrated[water] = 0
                    calibrated[barren] = 3
                    calibrated[vegetation] = 4
                    calibrated[urban] = 2
                    calibrated[model_road & valid & (~water)] = 1
                    calibrated[~valid] = NO_DATA_LABEL
                    if valid_values.size:
                        smooth_source = calibrated.copy()
                        smooth_source[~valid] = 3
                        smooth_labels = cv2.medianBlur(smooth_source, 3)
                        calibrated[valid] = smooth_labels[valid]
                        calibrated[model_road & valid & (~water)] = 1
                        calibrated[~valid] = NO_DATA_LABEL
                    labels[y0 : y0 + height, x0 : x0 + width] = calibrated

                    valid_labels = calibrated[calibrated != NO_DATA_LABEL]
                    if valid_labels.size:
                        class_pixel_counts += np.bincount(
                            valid_labels.reshape(-1),
                            minlength=len(prediction.record.class_names),
                        ).astype(np.int64)

        prediction.class_percentages = class_percentage_table_from_counts(
            class_pixel_counts,
            prediction.record.class_names,
        )
        prediction.dominant_class_name = prediction.class_percentages[0]["class_name"] if prediction.class_percentages else "Unknown"
        prediction.statistical_calibration_applied = True
        prediction.contrast_enhancement_applied = True
        return True

    @staticmethod
    def _road_window_positions(length: int, window_size: int, stride: int) -> list[int]:
        if length <= window_size:
            return [0]
        positions = list(range(0, length - window_size + 1, stride))
        final_position = length - window_size
        if positions[-1] != final_position:
            positions.append(final_position)
        return positions

    def _filter_learned_road_patch(
        self,
        road_probability: np.ndarray,
        base_labels: np.ndarray,
        valid_mask: np.ndarray,
        candidate_valid_mask: np.ndarray | None = None,
        gsd_m: float | None = None,
        audit_stats: dict[str, int] | None = None,
    ) -> np.ndarray:
        threshold = float(self.settings.road_head_probability_threshold)
        strong_threshold = max(threshold, float(self.settings.road_head_strong_threshold))
        halo_size = max(1, int(self.settings.road_head_invalid_halo_pixels))
        if halo_size % 2 == 0:
            halo_size += 1
        invalid_halo = cv2.dilate(
            (~valid_mask).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_size, halo_size)),
            iterations=1,
        ).astype(bool)
        candidate_valid = valid_mask if candidate_valid_mask is None else candidate_valid_mask
        safe_valid = valid_mask & candidate_valid & (~invalid_halo)
        strong = road_probability >= strong_threshold
        allowed = np.isin(base_labels, (1, 2, 3))
        candidate = (road_probability >= threshold) & allowed & safe_valid

        # Prevent a shifted specialist from painting an entire tile as road.
        allowed_count = int(np.count_nonzero(allowed & safe_valid))
        if allowed_count and np.count_nonzero(candidate) / allowed_count > 0.45:
            local_values = road_probability[allowed & safe_valid]
            adaptive_threshold = max(threshold, float(np.quantile(local_values, 0.65)))
            candidate = (road_probability >= adaptive_threshold) & allowed & safe_valid

        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ).astype(bool)
        candidate &= allowed & safe_valid
        if audit_stats is not None:
            audit_stats["candidate_cells_including_tile_halos"] = audit_stats.get("candidate_cells_including_tile_halos", 0) + int(candidate.sum())
        if not np.any(candidate):
            return candidate

        num_labels, components, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
        cleaned = np.zeros_like(candidate, dtype=bool)
        distance = cv2.distanceTransform(candidate.astype(np.uint8), cv2.DIST_L2, 5) if gsd_m is not None else None
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            x, y = int(stats[idx, cv2.CC_STAT_LEFT]), int(stats[idx, cv2.CC_STAT_TOP])
            width = int(stats[idx, cv2.CC_STAT_WIDTH])
            height = int(stats[idx, cv2.CC_STAT_HEIGHT])
            region = np.s_[y:y+height, x:x+width]
            component = components[region] == idx
            longest_side = max(width, height)
            elongation = longest_side / max(1, min(width, height))
            fill_ratio = area / max(1, width * height)
            has_strong_seed = bool(np.any(strong[region] & component))
            road_shaped = elongation >= 1.55 or fill_ratio <= 0.50
            if distance is not None:
                typical_width = 2 * float(np.quantile(distance[region][component], 0.90)) * gsd_m
                road_shaped &= typical_width <= 22.0 and longest_side * gsd_m >= 18.0
            if area >= 7 and has_strong_seed and road_shaped:
                cleaned[region] |= component
        if audit_stats is not None:
            audit_stats["accepted_cells_including_tile_halos"] = audit_stats.get("accepted_cells_including_tile_halos", 0) + int(cleaned.sum())
        return cleaned

    def _apply_learned_road_head(
        self,
        loaded_image: LoadedImage,
        prediction: ModelPrediction,
    ) -> int:
        if prediction.full_label_map is None or loaded_image.raster_profile is None:
            return 0
        if prediction.record.model_type == "pan_texture":
            return apply_connected_pan_roads(self, loaded_image, prediction)
        labels = prediction.full_label_map[: loaded_image.height, : loaded_image.width]
        patch_size = prediction.record.patch_size
        source_size, effective_gsd, scale = self._raster_inference_geometry(loaded_image, patch_size)
        step = max(1, int(round(source_size / patch_size)))
        step = max(step, int(np.ceil(np.sqrt(labels.size / self.settings.road_water_max_analysis_pixels))))
        frozen_labels = labels[::step, ::step].copy()
        shape = frozen_labels.shape
        with rasterio.open(loaded_image.source_path) as source:
            source_valid = source.read_masks(1,out_shape=shape,resampling=rasterio.enums.Resampling.nearest)>0
        if loaded_image.cloud_mask is not None:
            source_valid &= loaded_image.cloud_mask.window(0,0,loaded_image.width,loaded_image.height,shape)==0
        road_sum = np.zeros(shape, np.float32)
        weight_sum = np.zeros(shape, np.float32)
        observations = np.zeros(shape, np.uint8)
        votes = np.zeros(shape, np.uint8)
        model, device = self._get_learned_road_head(len(prediction.record.class_names),
                           prediction.record.summary.get("road_specialist",{}).get("checkpoint_sha256"))
        stride = max(1, int(round(source_size * np.clip(self.settings.road_head_stride_ratio, 0.5, 1.0))))
        model_weights = np.maximum(np.outer(np.hanning(patch_size), np.hanning(patch_size)), 0.05).astype(np.float32)
        batch_size = max(1, min(32, self.settings.prediction_batch_patches))
        cloud_threshold = self._cloud_threshold(loaded_image)
        bright_threshold = self._road_cloud_threshold(loaded_image)
        with rasterio.open(loaded_image.source_path) as src:
            patches: list[np.ndarray] = []
            valid_masks: list[np.ndarray] = []
            windows: list[tuple[int, int, int, int, int, int, int, int]] = []

            def flush() -> None:
                if not patches:
                    return
                with torch.inference_mode():
                    tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).float().to(device)
                    probabilities = torch.softmax(model(tensor), 1)[:, 1].cpu().numpy()
                weight = max(0.1, self.settings.road_head_weight)
                probabilities = weight * probabilities / (1 - probabilities + weight * probabilities + 1e-7)
                for idx, (x, y, width, height, ax0, ay0, ax1, ay1) in enumerate(windows):
                    map_x, map_y = np.meshgrid(
                        ((np.arange(ax0, ax1)*step-x+0.5)*patch_size/width-0.5).astype(np.float32),
                        ((np.arange(ay0, ay1)*step-y+0.5)*patch_size/height-0.5).astype(np.float32),
                    )
                    def sample(array, interpolation):
                        return cv2.remap(array.astype(np.float32), map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)
                    region = (slice(ay0, ay1), slice(ax0, ax1))
                    valid = sample(valid_masks[idx], cv2.INTER_NEAREST) > 0.5
                    probability = sample(probabilities[idx], cv2.INTER_LINEAR)
                    weighted = sample(model_weights, cv2.INTER_LINEAR) * valid
                    road_sum[region] += probability * weighted
                    weight_sum[region] += weighted
                    observations[region] += valid.astype(np.uint8)
                    vote_threshold = max(0.50, self.settings.road_head_probability_threshold - 0.10)
                    votes[region] += ((probability >= vote_threshold) & valid).astype(np.uint8)
                patches.clear()
                valid_masks.clear()
                windows.clear()

            for y in self._road_window_positions(loaded_image.height, source_size, stride):
                for x in self._road_window_positions(loaded_image.width, source_size, stride):
                    width, height = min(source_size, loaded_image.width-x), min(source_size, loaded_image.height-y)
                    ax0, ay0 = int(np.ceil(x/step)), int(np.ceil(y/step))
                    ax1, ay1 = min(shape[1], int(np.ceil((x+width)/step))), min(shape[0], int(np.ceil((y+height)/step)))
                    if not np.any(np.isin(frozen_labels[ay0:ay1, ax0:ax1], (1, 2, 3))):
                        continue
                    patch, valid = read_normalized_raster_patch(
                        src, x0=x, y0=y, patch_size=patch_size,
                        normalization_low=float(loaded_image.normalization_low or 0),
                        normalization_high=float(loaded_image.normalization_high or 1),
                        source_window_size=source_size, cloud_threshold=cloud_threshold,
                        cloud_mask=loaded_image.cloud_mask,
                        enhance_local_contrast=self._use_local_contrast(loaded_image, scale),
                    )
                    halo_size = max(1, self.settings.road_head_invalid_halo_pixels) | 1
                    halo = cv2.dilate((~valid).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(halo_size, halo_size))) > 0
                    valid &= ~halo
                    if bright_threshold is not None:
                        valid &= read_raster_below_threshold_mask(src, x0=x, y0=y, patch_size=patch_size,
                                                                  source_window_size=source_size, threshold=bright_threshold)
                    patches.append(patch)
                    valid_masks.append(valid)
                    windows.append((x, y, width, height, ax0, ay0, ax1, ay1))
                    if len(patches) >= batch_size:
                        flush()
            flush()

        road_sum /= np.maximum(weight_sum, 1e-7)
        consensus_valid = (weight_sum > 0) & (votes / np.maximum(observations, 1) >= 0.67)
        # Reject a single optimistic edge view unless it is exceptionally strong.
        consensus_valid &= (observations >= 2) | (road_sum >= 0.97)
        gsd = None
        if loaded_image.source_metadata.get("pixel_size_units") == "metre":
            gsd = step * max(float(loaded_image.source_metadata["pixel_size_x"]), float(loaded_image.source_metadata["pixel_size_y"]))
        accepted = np.zeros(shape, bool)
        filter_audit: dict[str, int] = {}
        for y in range(0, shape[0], 256):
            for x in range(0, shape[1], 256):
                y0, x0 = max(0, y-32), max(0, x-32)
                y1, x1 = min(shape[0], y+288), min(shape[1], x+288)
                region = np.s_[y0:y1, x0:x1]
                local = self._filter_learned_road_patch(
                    road_sum[region], frozen_labels[region], source_valid[region],
                    consensus_valid[region], gsd_m=gsd, audit_stats=filter_audit,
                )
                height, width = min(256, shape[0]-y), min(256, shape[1]-x)
                accepted[y:y+height, x:x+width] = local[y-y0:y-y0+height, x-x0:x-x0+width]
        total_changed = 0
        for y in range(0, shape[0], 256):
            for x in range(0, shape[1], 256):
                coarse = accepted[y:y+256, x:x+256]
                if not coarse.any():
                    continue
                y0, x0 = y*step, x*step
                y1, x1 = min((y+coarse.shape[0])*step, labels.shape[0]), min((x+coarse.shape[1])*step, labels.shape[1])
                candidate = np.repeat(np.repeat(coarse,step,0),step,1)[:y1-y0,:x1-x0]
                local = labels[y0:y1, x0:x1]
                candidate &= np.isin(local, (1, 2, 3))
                total_changed += int(np.count_nonzero(candidate & (local != 1)))
                local[candidate] = 1
        loaded_image.source_metadata["road_head_fusion"] = "weighted_overlap_consensus_with_immutable_seeds"
        loaded_image.source_metadata["road_head_filter_audit"] = filter_audit
        prediction.learned_road_head_applied = True
        prediction.learned_road_head_pixels_added = total_changed
        prediction.topology_postprocessing_applied = True
        self._refresh_prediction_percentages(prediction)
        return total_changed

    def _apply_learned_road_head_legacy(
        self,
        loaded_image: LoadedImage,
        prediction: ModelPrediction,
    ) -> int:
        if loaded_image.raster_profile is None or prediction.full_label_map is None:
            return 0
        if 1 not in prediction.record.class_names:
            return 0

        model, device = self._get_learned_road_head(len(prediction.record.class_names))
        patch_size = prediction.record.patch_size
        source_patch_size, _, scale = self._raster_inference_geometry(loaded_image, patch_size)
        stride_ratio = float(np.clip(self.settings.road_head_stride_ratio, 0.5, 1.0))
        stride = max(1, int(round(source_patch_size * stride_ratio)))
        x_positions = self._road_window_positions(loaded_image.width, source_patch_size, stride)
        y_positions = self._road_window_positions(loaded_image.height, source_patch_size, stride)

        labels = prediction.full_label_map[: loaded_image.height, : loaded_image.width]
        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)
        cloud_threshold = self._cloud_threshold(loaded_image)
        road_bright_threshold = self._road_cloud_threshold(loaded_image)
        enhance_local_contrast = self._use_local_contrast(loaded_image, scale)
        batch_size = max(1, int(self.settings.prediction_batch_patches))
        total_changed = 0

        with rasterio.open(loaded_image.source_path) as src:
            batch_patches: list[np.ndarray] = []
            batch_valid_masks: list[np.ndarray] = []
            batch_road_valid_masks: list[np.ndarray] = []
            batch_windows: list[tuple[int, int, int, int]] = []

            def flush_batch() -> None:
                nonlocal total_changed
                if not batch_patches:
                    return
                patches = np.stack(batch_patches).astype(np.float32)
                with torch.no_grad():
                    tensor = torch.from_numpy(patches).unsqueeze(1).float().to(device)
                    probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()

                road_probability = probabilities[:, 1]
                weight = max(0.1, float(self.settings.road_head_weight))
                road_probability = (weight * road_probability) / (
                    1.0 - road_probability + weight * road_probability + 1e-7
                )

                for local_idx, (x0, y0, output_width, output_height) in enumerate(batch_windows):
                    local_labels = labels[y0 : y0 + output_height, x0 : x0 + output_width]
                    base_model_labels = cv2.resize(
                        local_labels,
                        (patch_size, patch_size),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
                    model_candidate = self._filter_learned_road_patch(
                        road_probability[local_idx],
                        base_model_labels,
                        batch_valid_masks[local_idx],
                        batch_road_valid_masks[local_idx],
                    )
                    if not np.any(model_candidate):
                        continue

                    source_candidate = cv2.resize(
                        model_candidate.astype(np.uint8),
                        (output_width, output_height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    source_valid = cv2.resize(
                        batch_valid_masks[local_idx].astype(np.uint8),
                        (output_width, output_height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    source_candidate &= source_valid
                    source_candidate &= (local_labels != 0) & (local_labels != NO_DATA_LABEL)
                    changed = source_candidate & (local_labels != 1)
                    total_changed += int(np.count_nonzero(changed))
                    local_labels[source_candidate] = 1

                batch_patches.clear()
                batch_valid_masks.clear()
                batch_road_valid_masks.clear()
                batch_windows.clear()

            for y0 in y_positions:
                for x0 in x_positions:
                    output_width = min(source_patch_size, loaded_image.width - x0)
                    output_height = min(source_patch_size, loaded_image.height - y0)
                    patch, valid_mask = read_normalized_raster_patch(
                        src,
                        x0=x0,
                        y0=y0,
                        patch_size=patch_size,
                        normalization_low=normalization_low,
                        normalization_high=normalization_high,
                        source_window_size=source_patch_size,
                        cloud_threshold=cloud_threshold,
                        cloud_mask=loaded_image.cloud_mask,
                        enhance_local_contrast=enhance_local_contrast,
                    )
                    batch_patches.append(patch)
                    batch_valid_masks.append(valid_mask)
                    if road_bright_threshold is None:
                        road_valid_mask = valid_mask
                    else:
                        road_valid_mask = read_raster_below_threshold_mask(
                            src,
                            x0=x0,
                            y0=y0,
                            patch_size=patch_size,
                            source_window_size=source_patch_size,
                            threshold=road_bright_threshold,
                        )
                    batch_road_valid_masks.append(road_valid_mask)
                    batch_windows.append((x0, y0, output_width, output_height))
                    if len(batch_patches) >= batch_size:
                        flush_batch()
            flush_batch()

        if total_changed:
            self._refresh_prediction_percentages(prediction)
            prediction.learned_road_head_applied = True
            prediction.learned_road_head_pixels_added = total_changed
            prediction.topology_postprocessing_applied = True
        return total_changed

    def _apply_road_water_recovery(self, loaded_image: LoadedImage, prediction: ModelPrediction) -> int:
        labels = prediction.full_label_map
        metadata = loaded_image.source_metadata
        stats: dict[str, Any] = {"status": "skipped", "pixels_changed": 0, "method_version": "road_water_v1"}
        prediction.road_water_recovery = stats
        if labels is None or loaded_image.raster_profile is None or metadata.get("pixel_size_units") != "metre":
            stats["reason"] = "Projected metre-based raster labels are required for physical width checks."
            return 0

        pixel_x = float(metadata["pixel_size_x"])
        pixel_y = float(metadata["pixel_size_y"])
        native_gsd = max(pixel_x, pixel_y)
        if native_gsd > 5.0 or min(pixel_x, pixel_y) <= 0 or max(pixel_x, pixel_y) / min(pixel_x, pixel_y) > 1.1:
            stats["reason"] = "Recovery requires approximately square pixels with native GSD at most 5 m."
            return 0

        patch_size = prediction.record.patch_size
        source_size, _, scale = self._raster_inference_geometry(loaded_image, patch_size)
        step = max(1, int(round(source_size / patch_size)))
        step = max(step, int(np.ceil(np.sqrt(labels.size / self.settings.road_water_max_analysis_pixels))))
        gsd = native_gsd * step
        if gsd > 5.0:
            stats["reason"] = "Scene exceeds the analysis memory budget at a road-resolving scale."
            return 0

        # Freeze both seeds and protected water before any corrections are applied.
        analysis_labels = labels[: loaded_image.height : step, : loaded_image.width : step].copy()
        geometry = water_recovery_geometry(analysis_labels, gsd)
        stats.update(geometry.stats)
        stats["min_road_probability"] = self.settings.road_water_min_probability
        stats["max_base_water_probability"] = self.settings.road_water_max_base_water_probability
        stats["minimum_overlapping_observations"] = 2
        stats["rotation_consistency_required"] = True
        if not np.any(geometry.eligible):
            stats["reason"] = "No isolated thin urban water components meet the geometry gate."
            return 0

        consensus = ProbabilityConsensus(analysis_labels.shape)
        base_model, device = self._get_unet_model(prediction.record)
        road_model, road_device = self._get_learned_road_head(len(prediction.record.class_names))
        if device != road_device:
            raise RuntimeError("Base and road models must use the same device for recovery.")
        stride = max(1, int(round(source_size * 0.75)))
        x_positions = self._road_window_positions(loaded_image.width, source_size, stride)
        y_positions = self._road_window_positions(loaded_image.height, source_size, stride)
        weights = np.maximum(np.outer(np.hanning(patch_size), np.hanning(patch_size)), 0.05).astype(np.float32)
        cloud_threshold = self._cloud_threshold(loaded_image)
        road_bright_threshold = self._road_cloud_threshold(loaded_image)
        local_contrast = self._use_local_contrast(loaded_image, scale)
        batch_size = min(16, max(1, int(self.settings.prediction_batch_patches)))
        evaluated_windows = 0

        with rasterio.open(loaded_image.source_path) as src:
            patches: list[np.ndarray] = []
            validity: list[np.ndarray] = []
            windows: list[tuple[int, int, int, int, int, int, int, int]] = []

            def flush() -> None:
                nonlocal evaluated_windows
                if not patches:
                    return
                batch = torch.from_numpy(np.stack(patches)).unsqueeze(1).float().to(device)
                with torch.inference_mode():
                    base_water = torch.softmax(base_model(batch), dim=1)[:, 0].cpu().numpy()
                    expert = torch.softmax(road_model(batch), dim=1)
                    rotated = torch.flip(torch.softmax(road_model(torch.flip(batch, (-2, -1))), dim=1), (-2, -1))
                    road_a = expert[:, 1].cpu().numpy()
                    road_b = rotated[:, 1].cpu().numpy()
                    expert_road = 0.5 * (road_a + road_b)
                    expert_water = 0.5 * (expert[:, 0] + rotated[:, 0]).cpu().numpy()
                for idx, (x, y, width, height, ax0, ay0, ax1, ay1) in enumerate(windows):
                    map_x, map_y = np.meshgrid(
                        ((np.arange(ax0, ax1) * step - x + 0.5) * patch_size / width - 0.5).astype(np.float32),
                        ((np.arange(ay0, ay1) * step - y + 0.5) * patch_size / height - 0.5).astype(np.float32),
                    )

                    def sample(array: np.ndarray, nearest: bool = False) -> np.ndarray:
                        return cv2.remap(
                            array.astype(np.float32), map_x, map_y,
                            cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE,
                        )

                    stable = (road_a[idx] >= 0.80) & (road_b[idx] >= 0.80)
                    valid = sample(validity[idx], nearest=True) > 0.5
                    region = (slice(ay0, ay1), slice(ax0, ax1))
                    valid &= geometry.eligible[region]
                    consensus.add(
                        region, sample(expert_road[idx]), sample(expert_water[idx]),
                        sample(base_water[idx]), valid, sample(weights),
                        sample(stable, nearest=True) > 0.5,
                    )
                evaluated_windows += len(patches)
                patches.clear()
                validity.clear()
                windows.clear()

            for y in y_positions:
                for x in x_positions:
                    width = min(source_size, loaded_image.width - x)
                    height = min(source_size, loaded_image.height - y)
                    ax0, ay0 = int(np.ceil(x / step)), int(np.ceil(y / step))
                    ax1 = min(analysis_labels.shape[1], int(np.ceil((x + width) / step)))
                    ay1 = min(analysis_labels.shape[0], int(np.ceil((y + height) / step)))
                    if not np.any(geometry.eligible[ay0:ay1, ax0:ax1]):
                        continue
                    patch, valid = read_normalized_raster_patch(
                        src, x0=x, y0=y, patch_size=patch_size,
                        normalization_low=float(loaded_image.normalization_low or 0.0),
                        normalization_high=float(loaded_image.normalization_high or 1.0),
                        source_window_size=source_size, cloud_threshold=cloud_threshold,
                        cloud_mask=loaded_image.cloud_mask,
                        enhance_local_contrast=local_contrast,
                    )
                    invalid_halo = cv2.dilate(
                        (~valid).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
                    ).astype(bool)
                    valid &= ~invalid_halo
                    if road_bright_threshold is not None:
                        valid &= read_raster_below_threshold_mask(
                            src, x0=x, y0=y, patch_size=patch_size,
                            source_window_size=source_size, threshold=road_bright_threshold,
                        )
                    patches.append(patch)
                    validity.append(valid)
                    windows.append((x, y, width, height, ax0, ay0, ax1, ay1))
                    if len(patches) >= batch_size:
                        flush()
            flush()

        road, water, base_water, agreement = consensus.finish()
        stats["evidence_counts"] = {
            "strong_road": int(np.count_nonzero(geometry.eligible & (road >= self.settings.road_water_min_probability))),
            "base_water_uncertain": int(np.count_nonzero(geometry.eligible & (base_water <= self.settings.road_water_max_base_water_probability) & (consensus.observations > 0))),
            "multiple_contexts": int(np.count_nonzero(geometry.eligible & (consensus.observations >= 2))),
            "rotation_agreement": int(np.count_nonzero(geometry.eligible & (agreement >= 0.67))),
        }
        corrections = select_water_road_corrections(
            analysis_labels, geometry, road, water, base_water, agreement,
            consensus.observations, gsd,
            min_road_probability=self.settings.road_water_min_probability,
            max_base_water_probability=self.settings.road_water_max_base_water_probability,
        )
        total_changed = 0
        changed_cells = np.zeros(corrections.shape, dtype=bool)
        for ay0 in range(0, corrections.shape[0], 256):
            ay1 = min(ay0 + 256, corrections.shape[0])
            y0, y1 = ay0 * step, min(ay1 * step, loaded_image.height)
            for ax0 in range(0, corrections.shape[1], 256):
                ax1 = min(ax0 + 256, corrections.shape[1])
                coarse = corrections[ay0:ay1, ax0:ax1]
                if not np.any(coarse):
                    continue
                x0, x1 = ax0 * step, min(ax1 * step, loaded_image.width)
                source_candidate = np.repeat(np.repeat(coarse, step, axis=0), step, axis=1)[: y1-y0, : x1-x0]
                local = labels[y0:y1, x0:x1]
                change = source_candidate & (local == 0)
                total_changed += int(change.sum())
                local[change] = 1
                changed_cells[ay0:ay1, ax0:ax1] = coarse

        stats.update({
            "status": "applied" if total_changed else "no_confident_corrections",
            "pixels_changed": total_changed,
            "analysis_cells_changed": int(changed_cells.sum()),
            "evaluated_windows": evaluated_windows,
            "protected_water_cells_changed": 0,
            "thresholds_are_calibrated": False,
        })
        preview_h, preview_w = loaded_image.preview_rgb.shape[:2]
        diagnostic = np.zeros(analysis_labels.shape, dtype=np.uint8)
        diagnostic[analysis_labels == 0] = 1
        diagnostic[geometry.protected] = 2
        diagnostic[geometry.eligible] = 3
        diagnostic[changed_cells] = 4
        prediction.road_water_correction_preview = cv2.resize(diagnostic, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)
        if total_changed:
            self._refresh_prediction_percentages(prediction)
        return total_changed

    def _apply_road_specialist_pass(
        self,
        loaded_image: LoadedImage,
        prediction: ModelPrediction,
        tile_size: int = 1024,
    ) -> int:
        if loaded_image.raster_profile is None or prediction.full_label_map is None:
            return 0
        if 1 not in prediction.record.class_names:
            return 0

        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)
        labels = prediction.full_label_map[: loaded_image.height, : loaded_image.width]
        total_changed = 0
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))

        with rasterio.open(loaded_image.source_path) as src:
            for y0 in range(0, loaded_image.height, tile_size):
                for x0 in range(0, loaded_image.width, tile_size):
                    height = min(tile_size, loaded_image.height - y0)
                    width = min(tile_size, loaded_image.width - x0)
                    window = rasterio.windows.Window(x0, y0, width, height)
                    raw = src.read(1, window=window).astype(np.float32)
                    valid = np.isfinite(raw)
                    if src.nodata is not None:
                        valid &= raw != src.nodata
                    cloud_threshold = self._road_cloud_threshold(loaded_image)
                    if cloud_threshold is not None:
                        valid &= raw < cloud_threshold

                    image = np.clip(
                        (raw - normalization_low) / max(normalization_high - normalization_low, 1e-6),
                        0,
                        1,
                    ).astype(np.float32)
                    enhanced = clahe.apply(to_uint8(image)).astype(np.float32) / 255.0
                    enhanced[~valid] = 0.0

                    local_labels = labels[y0 : y0 + height, x0 : x0 + width]
                    protected_water = (local_labels == 0) | (~valid)
                    valid_values = enhanced[valid]
                    if valid_values.size < 64:
                        continue

                    p40, p55, p75 = np.percentile(valid_values, [40, 55, 75])
                    mean15 = cv2.blur(enhanced, (15, 15))
                    sq15 = cv2.blur(enhanced * enhanced, (15, 15))
                    std15 = np.sqrt(np.clip(sq15 - mean15 * mean15, 0, None))
                    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
                    grad = np.sqrt(gx * gx + gy * gy)

                    edges = cv2.Canny(to_uint8(enhanced), 40, 130) > 0
                    bright_linear = (enhanced > p55) & (std15 < 0.16) & (grad > np.percentile(grad[valid], 55))
                    dark_linear = (enhanced > p40) & (enhanced < p75) & edges & (std15 < 0.18)
                    base = (bright_linear | dark_linear) & valid & (~protected_water)

                    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
                    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17))
                    d_kernel = np.eye(13, dtype=np.uint8)
                    ad_kernel = np.fliplr(d_kernel)
                    road_h = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_OPEN, h_kernel)
                    road_v = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_OPEN, v_kernel)
                    road_d = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_OPEN, d_kernel)
                    road_ad = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_OPEN, ad_kernel)
                    road_candidate = (road_h | road_v | road_d | road_ad).astype(bool)
                    road_candidate = cv2.dilate(
                        road_candidate.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                        iterations=1,
                    ).astype(bool)

                    allowed_source = (local_labels == 2) | (local_labels == 3) | (local_labels == 1)
                    road_candidate &= allowed_source & (~protected_water)
                    if not np.any(road_candidate):
                        continue

                    num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(road_candidate.astype(np.uint8), 8)
                    cleaned = np.zeros_like(road_candidate, dtype=bool)
                    for idx in range(1, num_labels):
                        area = int(stats[idx, cv2.CC_STAT_AREA])
                        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
                        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
                        elongation = max(bw, bh) / max(1, min(bw, bh))
                        if area >= 24 and (elongation >= 2.2 or max(bw, bh) >= 24):
                            cleaned[cc == idx] = True

                    if not np.any(cleaned):
                        continue

                    changed_mask = cleaned & (local_labels != 1)
                    total_changed += int(np.count_nonzero(changed_mask))
                    local_labels[cleaned] = 1

        if total_changed:
            self._refresh_prediction_percentages(prediction)
            prediction.road_specialist_pass_applied = True
            prediction.topology_postprocessing_applied = True
        return total_changed

    def _apply_water_coherence_cleanup(
        self,
        loaded_image: LoadedImage,
        prediction: ModelPrediction,
        tile_size: int = 2048,
    ) -> int:
        if loaded_image.raster_profile is None or prediction.full_label_map is None:
            return 0
        if 0 not in prediction.record.class_names:
            return 0

        normalization_low = float(loaded_image.normalization_low or 0.0)
        normalization_high = float(loaded_image.normalization_high or 1.0)
        labels = prediction.full_label_map[: loaded_image.height, : loaded_image.width]
        total_restored = 0

        with rasterio.open(loaded_image.source_path) as src:
            for y0 in range(0, loaded_image.height, tile_size):
                for x0 in range(0, loaded_image.width, tile_size):
                    height = min(tile_size, loaded_image.height - y0)
                    width = min(tile_size, loaded_image.width - x0)
                    window = rasterio.windows.Window(x0, y0, width, height)
                    raw = src.read(1, window=window).astype(np.float32)
                    valid = np.isfinite(raw)
                    if src.nodata is not None:
                        valid &= raw != src.nodata

                    image = np.clip(
                        (raw - normalization_low) / max(normalization_high - normalization_low, 1e-6),
                        0,
                        1,
                    ).astype(np.float32)
                    mean31 = cv2.blur(image, (31, 31))
                    sq31 = cv2.blur(image * image, (31, 31))
                    std31 = np.sqrt(np.clip(sq31 - mean31 * mean31, 0, None))
                    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
                    gradient = np.sqrt(gx * gx + gy * gy)

                    local_labels = labels[y0 : y0 + height, x0 : x0 + width]
                    valid_values = image[valid]
                    if valid_values.size < 64:
                        continue

                    p42, p58 = np.percentile(valid_values, [42, 58])
                    water_brightness_cut = float(min(0.52, max(0.34, p42 + 0.08)))
                    water_like = (
                        valid
                        & (image < water_brightness_cut)
                        & (std31 < 0.16)
                        & (gradient < 0.23)
                        & (local_labels != NO_DATA_LABEL)
                    )
                    existing_water = (local_labels == 0) & valid
                    if not np.any(water_like):
                        continue

                    existing_water_fraction = float(np.mean(existing_water))
                    water_like_fraction = float(np.mean(water_like & valid))
                    tile_mean = float(np.mean(valid_values))
                    has_water_context = existing_water_fraction > 0.02
                    large_dark_water_tile = water_like_fraction > 0.42 and tile_mean < 0.52 and p58 < 0.58
                    if not has_water_context and not large_dark_water_tile:
                        continue

                    water_seed = cv2.dilate(
                        existing_water.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                        iterations=1,
                    ).astype(bool)
                    if existing_water_fraction > 0.12 or large_dark_water_tile:
                        coherent_water = water_like & (water_seed | existing_water | (mean31 < water_brightness_cut))
                    else:
                        coherent_water = water_like & water_seed

                    # Keep roads and very bright built-up cores from being swallowed by the water cleanup.
                    bright_land = (image > max(0.62, water_brightness_cut + 0.12)) & (std31 > 0.08)
                    restore_mask = coherent_water & (~bright_land) & (local_labels != 1) & (local_labels != 0)
                    if not np.any(restore_mask):
                        continue

                    if large_dark_water_tile or water_like_fraction > 0.28:
                        cleaned = restore_mask
                    else:
                        num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(restore_mask.astype(np.uint8), 8)
                        cleaned = np.zeros_like(restore_mask, dtype=bool)
                        for idx in range(1, num_labels):
                            area = int(stats[idx, cv2.CC_STAT_AREA])
                            if area >= 32:
                                cleaned[cc == idx] = True

                    if not np.any(cleaned):
                        continue
                    total_restored += int(np.count_nonzero(cleaned))
                    local_labels[cleaned] = 0

        if total_restored:
            prediction.water_coherence_cleanup_applied = True
            prediction.water_coherence_pixels_restored = total_restored
            self._refresh_prediction_percentages(prediction)
        return total_restored

    def _save_outputs(
        self,
        result_dir: Path,
        loaded_image: LoadedImage,
        prediction: ModelPrediction,
        mode: str,
        candidate_rankings: list[dict[str, Any]],
        requested_model_id: str | None,
    ) -> dict[str, Any]:
        ensure_dir(result_dir)
        self._apply_cloud_exclusions(loaded_image, prediction)

        calibration_applied = bool(
            self.settings.enable_statistical_calibration
            and self._apply_pan_statistical_calibration(loaded_image, prediction)
        )
        prediction.contrast_enhancement_applied = bool(
            prediction.contrast_enhancement_applied
            or loaded_image.source_metadata.get("local_contrast_applied")
        )
        learned_road_head_pixels = (
            self._apply_learned_road_head(loaded_image, prediction)
            if self.settings.enable_learned_road_head
            else 0
        )
        road_specialist_pixels = (
            self._apply_road_specialist_pass(loaded_image, prediction)
            if self.settings.enable_road_specialist
            else 0
        )
        water_restored_pixels = (
            self._apply_water_coherence_cleanup(loaded_image, prediction)
            if self.settings.enable_water_coherence_cleanup
            else 0
        )
        road_overlay_pixels = self._apply_reference_road_overlay(loaded_image, prediction)
        water_to_road_pixels = (
            self._apply_road_water_recovery(loaded_image, prediction)
            if self.settings.enable_road_water_recovery and prediction.record.model_type != "pan_texture"
            else 0
        )
        # No fusion or cleanup pass may put a land-cover label back under clouds.
        self._apply_cloud_exclusions(loaded_image, prediction)
        if prediction.full_label_map is not None:
            patch_step = int(loaded_image.source_metadata.get("inference_source_patch_size", prediction.record.patch_size))
            for row in range(prediction.patch_grid.shape[0]):
                for col in range(prediction.patch_grid.shape[1]):
                    local = prediction.full_label_map[
                        row * patch_step : min((row + 1) * patch_step, loaded_image.height),
                        col * patch_step : min((col + 1) * patch_step, loaded_image.width),
                    ]
                    valid = local[local != NO_DATA_LABEL]
                    if valid.size:
                        flat_idx = row * prediction.patch_grid.shape[1] + col
                        prediction.patch_labels[flat_idx] = np.bincount(valid, minlength=len(prediction.record.class_names)).argmax()
                        prediction.patch_grid[row, col] = prediction.patch_labels[flat_idx]
                        prediction.patch_coverages[flat_idx] = valid.size
                    else:
                        flat_idx = row * prediction.patch_grid.shape[1] + col
                        prediction.patch_coverages[flat_idx] = 0
                        prediction.patch_confidences[flat_idx] = 0
        generalization_warning = self._generalization_warning(prediction)

        input_preview_path = result_dir / "input_preview.png"
        Image.fromarray(loaded_image.preview_rgb, mode="RGB").save(input_preview_path)

        preview_height, preview_width = loaded_image.preview_rgb.shape[:2]
        if prediction.full_label_map is not None:
            label_preview = np.asarray(
                Image.fromarray(prediction.full_label_map.astype(np.uint8), mode="L").resize(
                    (preview_width, preview_height),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
        else:
            label_preview = resize_label_preview(prediction.patch_grid, preview_width, preview_height)
        color_preview = colorize_labels(label_preview, prediction.record)
        overlay_preview = cv2.addWeighted(loaded_image.preview_rgb, 0.48, color_preview, 0.52, 0)
        cloud_preview_path = None
        cloud_quality_path = None
        cloud_metadata = loaded_image.source_metadata.get("cloud_screening")
        if loaded_image.cloud_mask is not None and loaded_image.cloud_mask.has_clouds:
            cloud_codes = loaded_image.cloud_mask.window(0, 0, loaded_image.width, loaded_image.height,
                                                        (preview_height, preview_width))
            color_preview = color_cloud_mask(cloud_codes, color_preview)
            overlay_preview = color_cloud_mask(cloud_codes, overlay_preview)
            cloud_preview_path = result_dir / "cloud_screening_preview.png"
            Image.fromarray(color_cloud_mask(cloud_codes, loaded_image.preview_rgb), mode="RGB").save(cloud_preview_path)
            cloud_quality_path = result_dir / "cloud_quality_georef.tif"
            self._write_cloud_quality_raster(cloud_quality_path, loaded_image)

        segmentation_preview_path = result_dir / "segmentation_preview.png"
        marked_preview_path = result_dir / "marked_preview.png"
        Image.fromarray(color_preview, mode="RGB").save(segmentation_preview_path)
        Image.fromarray(overlay_preview, mode="RGB").save(marked_preview_path)

        road_water_diagnostic_path = None
        if prediction.road_water_correction_preview is not None:
            diagnostic_palette = np.array(
                [(232, 232, 232), (31, 119, 180), (9, 47, 87), (240, 155, 38), (215, 28, 137)],
                dtype=np.uint8,
            )
            road_water_diagnostic_path = result_dir / "road_water_diagnostic.png"
            Image.fromarray(diagnostic_palette[prediction.road_water_correction_preview]).save(road_water_diagnostic_path)

        patch_prediction_csv = result_dir / "patch_predictions.csv"
        with patch_prediction_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "col", "class_id", "class_name", "confidence", "coverage_pixels"])
            for row_idx in range(prediction.patch_grid.shape[0]):
                for col_idx in range(prediction.patch_grid.shape[1]):
                    flat_idx = row_idx * prediction.patch_grid.shape[1] + col_idx
                    class_id = int(prediction.patch_labels[flat_idx])
                    writer.writerow(
                        [
                            row_idx,
                            col_idx,
                            class_id,
                            prediction.record.class_names[class_id],
                            round(float(prediction.patch_confidences[flat_idx]), 4),
                            int(prediction.patch_coverages[flat_idx]),
                        ]
                    )

        percentages_json_path = result_dir / "class_percentages.json"
        percentages_json_path.write_text(json.dumps(prediction.class_percentages, indent=2), encoding="utf-8")

        labels_npy_path = result_dir / "segmentation_labels.npy"
        np.save(labels_npy_path, (prediction.full_label_map if prediction.full_label_map is not None else prediction.patch_grid).astype(np.uint8))

        geotiff_path = None
        vector_zip_path = None
        vector_layers: list[dict[str, Any]] = []
        vector_validation: dict[str, Any] = {
            "ok": False,
            "message": "Vector export was not attempted.",
        }
        if loaded_image.raster_profile is not None:
            geotiff_path = result_dir / "segmentation_labels_georef.tif"
            self._write_geotiff_labels(
                geotiff_path,
                patch_grid=prediction.patch_grid,
                patch_size=prediction.record.patch_size,
                loaded_image=loaded_image,
                full_label_map=prediction.full_label_map,
            )
            if self.settings.export_shapefiles_on_predict and not self.settings.skip_vector_export:
                vector_zip_path, vector_layers = self._write_class_shapefiles(
                    geotiff_path=geotiff_path,
                    destination_dir=result_dir / "class_shapefiles",
                    record=prediction.record,
                )
                vector_validation = self._validate_vector_export(
                    zip_path=vector_zip_path,
                    destination_dir=result_dir / "class_shapefiles",
                    vector_layers=vector_layers,
                )

        full_outputs_created = False
        segmentation_full_path = None
        marked_full_path = None
        if prediction.full_label_map is not None:
            full_labels = prediction.full_label_map[: loaded_image.height, : loaded_image.width]
            if (
                loaded_image.grayscale is not None
                and loaded_image.width * loaded_image.height <= self.settings.full_visual_pixel_limit
            ):
                full_color = colorize_labels(full_labels, prediction.record)
                gray_u8 = to_uint8(loaded_image.grayscale[: full_labels.shape[0], : full_labels.shape[1]])
                base_rgb = np.dstack([gray_u8] * 3)
                full_overlay = cv2.addWeighted(base_rgb, 0.48, full_color, 0.52, 0)
                if loaded_image.cloud_mask is not None and loaded_image.cloud_mask.has_clouds:
                    codes = loaded_image.cloud_mask.window(0, 0, loaded_image.width, loaded_image.height)
                    full_color = color_cloud_mask(codes, full_color)
                    full_overlay = color_cloud_mask(codes, full_overlay)

                segmentation_full_path = result_dir / "segmentation_full.png"
                marked_full_path = result_dir / "marked_full.png"
                Image.fromarray(full_color, mode="RGB").save(segmentation_full_path)
                Image.fromarray(full_overlay, mode="RGB").save(marked_full_path)
                full_outputs_created = True
        elif (
            loaded_image.grayscale is not None
            and loaded_image.valid_mask is not None
            and loaded_image.width * loaded_image.height <= self.settings.full_visual_pixel_limit
        ):
            full_labels = build_full_label_map(
                prediction.patch_grid,
                patch_size=prediction.record.patch_size,
                height=loaded_image.height,
                width=loaded_image.width,
            )
            full_labels[~loaded_image.valid_mask] = NO_DATA_LABEL
            full_color = colorize_labels(full_labels, prediction.record)
            gray_u8 = to_uint8(loaded_image.grayscale)
            base_rgb = np.dstack([gray_u8] * 3)
            full_overlay = cv2.addWeighted(base_rgb, 0.48, full_color, 0.52, 0)
            if loaded_image.cloud_mask is not None and loaded_image.cloud_mask.has_clouds:
                codes = loaded_image.cloud_mask.window(0, 0, loaded_image.width, loaded_image.height)
                full_color = color_cloud_mask(codes, full_color)
                full_overlay = color_cloud_mask(codes, full_overlay)

            segmentation_full_path = result_dir / "segmentation_full.png"
            marked_full_path = result_dir / "marked_full.png"
            Image.fromarray(full_color, mode="RGB").save(segmentation_full_path)
            Image.fromarray(full_overlay, mode="RGB").save(marked_full_path)
            full_outputs_created = True

        summary_text_path = result_dir / "RESULTS_SIMPLE.txt"
        road_continuity = loaded_image.source_metadata.get("road_continuity", {})
        summary_lines = [
            "Urban Feature Platform Result",
            "",
            f"Selected model: {prediction.record.display_name}",
            f"Dataset family: {prediction.record.dataset_name}",
            f"Model type: {prediction.record.model_type}",
            f"Prediction mode: {mode}",
            f"Image processing mode: {loaded_image.processing_mode}",
            f"Requested model id: {requested_model_id or 'Auto'}",
            f"Top detected class: {prediction.dominant_class_name}",
            f"Model fit score: {prediction.selected_score:.4f}",
            f"{prediction.record.primary_metric_label} stored for this model: {float(prediction.record.primary_metric_value or 0.0):.4f}",
            f"PAN statistical calibration applied: {'Yes' if calibration_applied else 'No'}",
            f"Local contrast normalization applied: {'Yes' if prediction.contrast_enhancement_applied else 'No'}",
            f"Cloud-aware masking applied: {'Yes' if loaded_image.source_metadata.get('cloud_masking_applied') else 'No'}",
            f"Cloud/uncertain area excluded: {(cloud_metadata or {}).get('excluded_percent_of_full_grid', 0):.3f}% of full image grid",
            "Class percentages use clear classified pixels only; cloud/uncertain and no-data are excluded.",
            "Cloud screening does not reconstruct hidden ground. Thin clouds and cloud shadows may remain.",
            f"Inference target GSD: {loaded_image.source_metadata.get('inference_target_gsd_m') or 'native/unknown'} m",
            f"Learned road head applied: {'Yes' if prediction.learned_road_head_applied else 'No'}",
            f"Learned road head pixels added: {learned_road_head_pixels}",
            f"Learned road probability weight: {self.settings.road_head_weight:.2f}",
            f"Learned road support threshold: {road_continuity.get('support_probability', self.settings.road_head_probability_threshold):.2f}",
            f"Road processing method: {loaded_image.source_metadata.get('road_head_fusion', 'legacy')}",
            f"Road analysis GSD: {loaded_image.source_metadata.get('road_analysis_gsd_m', 'not available')} m",
            f"Road-water recovery status: {prediction.road_water_recovery.get('status', 'disabled')}",
            f"Water-to-road pixels recovered: {water_to_road_pixels}",
            "Confidence and historical validation metrics describe the base model, before fusion.",
            f"Road specialist pass applied: {'Yes' if prediction.road_specialist_pass_applied else 'No'}",
            f"Topology-aware post-processing applied: {'Yes' if prediction.topology_postprocessing_applied else 'No'}",
            f"Road specialist pixels added: {road_specialist_pixels}",
            f"Water coherence cleanup applied: {'Yes' if prediction.water_coherence_cleanup_applied else 'No'}",
            f"Water pixels restored: {water_restored_pixels}",
            f"Reference road overlay pixels: {road_overlay_pixels}",
            f"Road ratio calibration applied: {'Yes' if prediction.road_ratio_calibration_applied else 'No'}",
            f"Road buffer selected: {prediction.road_buffer_meters if prediction.road_buffer_meters is not None else 'N/A'} m",
            f"Road target share of urban fabric: {100.0 * prediction.road_target_share_of_urban:.1f}%" if prediction.road_target_share_of_urban else "Road target share of urban fabric: N/A",
            f"Generalization warning: {generalization_warning or 'None'}",
            "",
            "Class percentages:",
        ]
        for item in prediction.class_percentages:
            summary_lines.append(f"- {item['class_name']}: {item['percentage']:.2f}%")
        summary_lines.extend(
            [
                "",
                "Files generated:",
                f"- {input_preview_path.name}",
                f"- {marked_preview_path.name}",
                f"- {segmentation_preview_path.name}",
                f"- {patch_prediction_csv.name}",
                f"- {percentages_json_path.name}",
                f"- {labels_npy_path.name}",
            ]
        )
        if geotiff_path is not None:
            summary_lines.append(f"- {geotiff_path.name}")
        if cloud_quality_path is not None:
            summary_lines.extend([f"- {cloud_quality_path.name} (0=not screened out; 1=likely cloud; 2=fringe; 255=no-data)",
                                  f"- {cloud_preview_path.name}"])
        if vector_zip_path is not None:
            summary_lines.append(f"- {vector_zip_path.name}")
            summary_lines.append("- class_shapefiles/ folder with one georeferenced shapefile per detected class")
            summary_lines.append(f"- Vector export validation: {'PASS' if vector_validation.get('ok') else 'CHECK'} ({vector_validation.get('message')})")
        elif self.settings.skip_vector_export:
            summary_lines.append("- Vector shapefile export was skipped for this validation run.")
        elif geotiff_path is not None:
            summary_lines.append("- Class shapefile export is available on request after prediction.")
        if full_outputs_created:
            summary_lines.append(f"- {marked_full_path.name}")
            summary_lines.append(f"- {segmentation_full_path.name}")
        else:
            summary_lines.append("- Full-size PNG visuals were skipped to keep memory usage safe for this image size.")
        summary_text_path.write_text("\n".join(summary_lines), encoding="utf-8")

        summary_json = {
            "deployment_status": prediction.record.summary.get("deployment_status", "active"),
            "result_generated_at": datetime.now().isoformat(timespec="seconds"),
            "image": loaded_image.source_metadata,
            "selected_model": {
                "id": prediction.record.id,
                "display_name": prediction.record.display_name,
                "dataset_name": prediction.record.dataset_name,
                "model_type": prediction.record.model_type,
                "primary_metric_label": prediction.record.primary_metric_label,
                "secondary_metric_label": prediction.record.secondary_metric_label,
                "validation_accuracy": prediction.record.model_metrics.get("val_accuracy"),
                "validation_macro_f1": prediction.record.secondary_metric_value,
                "checkpoint_file": prediction.record.model_path.name,
                "checkpoint_sha256": prediction.record.loaded_checkpoint_sha256 or prediction.record.checkpoint_fingerprint(),
                "pipeline_version": loaded_image.source_metadata.get("inference_pipeline", "pan_legacy_v7"),
                "water_branch_sha256": prediction.record.summary.get("water_branch", {}).get("checkpoint_sha256"),
            },
            "selection_mode": mode,
            "requested_model_id": requested_model_id,
            "model_fit_score": round(prediction.selected_score, 4),
            "score_breakdown": prediction.score_breakdown,
            "dominant_class": prediction.dominant_class_name,
            "heuristic_water_overrides": prediction.heuristic_water_overrides,
            "pan_statistical_calibration_applied": calibration_applied,
            "contrast_enhancement_applied": prediction.contrast_enhancement_applied,
            "learned_road_head_applied": prediction.learned_road_head_applied,
            "road_head_checkpoint": str(self.settings.road_head_model_path.resolve()),
            "road_head_checkpoint_sha256": self._road_head_fingerprint,
            "road_head_reference_validation": prediction.record.summary.get("road_specialist"),
            "learned_road_head_pixels_added": learned_road_head_pixels,
            "learned_road_head_weight": self.settings.road_head_weight,
            "learned_road_head_probability_threshold": loaded_image.source_metadata.get("road_continuity", {}).get("support_probability", self.settings.road_head_probability_threshold),
            "learned_road_head_strong_threshold": loaded_image.source_metadata.get("road_continuity", {}).get("seed_probability", self.settings.road_head_strong_threshold),
            "learned_road_head_stride_ratio": 0.5 if loaded_image.source_metadata.get("road_continuity") else self.settings.road_head_stride_ratio,
            "road_continuity_processing": {key: loaded_image.source_metadata.get(key) for key in (
                "road_head_fusion", "road_analysis_gsd_m", "road_analysis_shape", "road_continuity",
                "road_size_dependent_downsampling", "road_consensus_policy", "road_head_filter_audit")},
            "learned_road_head_bright_guard_fraction": self.settings.road_head_bright_guard_fraction,
            "learned_road_head_invalid_halo_pixels": self.settings.road_head_invalid_halo_pixels,
            "road_water_recovery": prediction.road_water_recovery,
            "road_water_diagnostic_legend": {
                "blue": "Other retained water", "dark_blue": "Protected broad water or water-context channels",
                "orange": "Geometry candidate retained as water", "magenta": "Water changed to road",
                "grey": "Other classes or no-data",
            },
            "confidence_scope": "base_model_before_fusion",
            "validation_scope": prediction.record.summary.get("validation_scope", "historical_base_model_pseudo_label_validation_not_composite_or_independent_test"),
            "road_specialist_pass_applied": prediction.road_specialist_pass_applied,
            "topology_postprocessing_applied": prediction.topology_postprocessing_applied,
            "road_specialist_pixels_added": road_specialist_pixels,
            "water_coherence_cleanup_applied": prediction.water_coherence_cleanup_applied,
            "water_coherence_pixels_restored": water_restored_pixels,
            "reference_road_overlay_pixels": road_overlay_pixels,
            "road_ratio_calibration_applied": prediction.road_ratio_calibration_applied,
            "road_buffer_meters": prediction.road_buffer_meters,
            "road_target_share_of_urban_fabric": prediction.road_target_share_of_urban,
            "generalization_warning": generalization_warning,
            "production_postprocessing": {
                "model_output_preserved": not calibration_applied,
                "dark_water_override_enabled": self.settings.enable_dark_water_override,
                "statistical_calibration_enabled": self.settings.enable_statistical_calibration,
                "water_cleanup_enabled": self.settings.enable_water_coherence_cleanup,
                "road_specialist_enabled": self.settings.enable_road_specialist,
                "learned_road_head_enabled": self.settings.enable_learned_road_head,
                "road_water_recovery_enabled": self.settings.enable_road_water_recovery and prediction.record.model_type != "pan_texture",
                "spatial_cloud_screening_enabled": self.settings.enable_spatial_cloud_mask,
            },
            "class_percentages": prediction.class_percentages,
            "cloud_screening": cloud_metadata,
            "class_percentage_scope": "Clear classified pixels; cloud/uncertain and no-data excluded",
            "vector_shapefiles": vector_layers,
            "vector_export_validation": vector_validation,
            "candidate_rankings": candidate_rankings,
            "full_outputs_created": full_outputs_created,
            "note": " ".join(
                item
                for item in [
                    generalization_warning,
                    (f"Cloud/uncertain screening excludes {cloud_metadata.get('excluded_percent_of_full_grid', 0):.2f}% of the full grid. "
                     "Class percentages refer only to clear classified pixels. Thin clouds/shadows may remain; hidden ground is not reconstructed."
                     if cloud_metadata else None),
                    prediction.record.summary.get("validation_note", "Validation scores describe fit to cluster-derived pseudo-labels rather than fully manual ground-truth annotation."),
                ]
                if item
            ),
        }

        summary_json_path = result_dir / "prediction_summary.json"
        summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

        def file_url(path: Path | None) -> str | None:
            if path is None:
                return None
            return f"/results/{result_dir.name}/{path.name}"

        legend = [
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "color_rgb": list(prediction.record.display_class_colors.get(class_id, (200, 200, 200))),
            }
            for class_id, class_name in sorted(prediction.record.class_names.items())
        ]
        if loaded_image.cloud_mask is not None and loaded_image.cloud_mask.has_clouds:
            legend.append({"class_id": 254, "class_name": "Likely cloud (excluded)", "color_rgb": list(CLOUD_COLOR)})
            legend.append({"class_id": 253, "class_name": "Cloud fringe / uncertain (excluded)", "color_rgb": [233, 225, 201]})

        return {
            "result_id": result_dir.name,
            "image": loaded_image.source_metadata,
            "selection_mode": mode,
            "selected_model": summary_json["selected_model"],
            "score_breakdown": prediction.score_breakdown,
            "class_percentages": prediction.class_percentages,
            "cloud_screening": cloud_metadata,
            "dominant_class": prediction.dominant_class_name,
            "candidate_rankings": candidate_rankings,
            "legend": legend,
            "artifacts": {
                "input_preview_url": file_url(input_preview_path),
                "marked_preview_url": file_url(marked_preview_path),
                "segmentation_preview_url": file_url(segmentation_preview_path),
                "road_water_diagnostic_url": file_url(road_water_diagnostic_path),
                "cloud_screening_preview_url": file_url(cloud_preview_path),
                "cloud_quality_geotiff_url": file_url(cloud_quality_path),
                "marked_full_url": file_url(marked_full_path),
                "segmentation_full_url": file_url(segmentation_full_path),
                "labels_npy_url": file_url(labels_npy_path),
                "geotiff_url": file_url(geotiff_path),
                "class_shapefiles_zip_url": file_url(vector_zip_path),
                "summary_json_url": file_url(summary_json_path),
                "summary_text_url": file_url(summary_text_path),
                "patch_predictions_csv_url": file_url(patch_prediction_csv),
                "class_percentages_url": file_url(percentages_json_path),
            },
            "note": summary_json["note"],
        }

    def predict_file(self, file_path: Path, mode: str = "auto", model_id: str | None = None,
                     band_order: str | None = None, progress=None, native_refinement: bool = True) -> dict[str, Any]:
        # Background and direct API calls share one GPU and the same model cache.
        requested = time.perf_counter()
        with self._prediction_lock:
            started = time.perf_counter()
            result = self._predict_file(file_path,mode,model_id,band_order,progress,native_refinement)
            if isinstance(result, dict):
                result["runtime"] = {"processing_seconds": time.perf_counter() - started,
                                     "service_lock_wait_seconds": started - requested,
                                     "scope": "Input verification, model selection/loading, inference and raster/preview exports; upload, job queue and optional addons excluded"}
                identifier = result.get("result_id")
                if identifier:
                    from .saved_results import RESULT_ID
                    if RESULT_ID.fullmatch(identifier):
                        path = self.settings.result_dir / identifier / "prediction_summary.json"
                        if path.is_file():
                            summary = json.loads(path.read_text())
                            summary["runtime"] = result["runtime"]
                            temporary = path.with_suffix(".json.tmp")
                            temporary.write_text(json.dumps(summary, indent=2))
                            temporary.replace(path)
            return result

    def predict_model_roads(self, result_id, progress):
        with self._prediction_lock:
            if not hasattr(self, "_model_road_service"):
                from .model_roads import ModelRoadService
                self._model_road_service = ModelRoadService(self.settings, self.registry)
            return self._model_road_service.build(result_id, progress)

    def export_model_roads(self, result_id):
        with self._prediction_lock:
            from .model_roads import export_shapefile
            return export_shapefile(self.settings, result_id)

    def _predict_file(self, file_path: Path, mode: str = "auto", model_id: str | None = None,
                      band_order: str | None = None, progress=None, native_refinement: bool = True) -> dict[str, Any]:
        from .native06m_inference import Native06mInference, is_native_grid
        native_records = [record for record in self.registry.get_records().values() if record.model_type == "native_06m"]
        native = None
        if mode == "manual" and model_id:
            selected = self.registry.get(model_id)
            if selected.model_type == "native_06m":
                native = selected
        elif mode == "auto" and native_records:
            with rasterio.open(file_path) as source:
                if is_native_grid(source):
                    native = max((record for record in native_records if record.auto_select),
                                 key=lambda record: record.priority, default=None)
        if native is not None:
            if not hasattr(self, "_native_service"):
                self._native_service = Native06mInference(self.settings)
            return self._native_service.predict(file_path, native, mode, band_order, progress, refinement=native_refinement)
        loaded_image = load_uploaded_image(
            file_path,
            preview_max_dimension=self.settings.preview_max_dimension,
            raster_stream_pixel_limit=self.settings.raster_stream_pixel_limit,
            raster_stream_file_size_mb=self.settings.raster_stream_file_size_mb,
            raster_stats_sample_max_dimension=self.settings.raster_stats_sample_max_dimension,
            enable_spatial_cloud_mask=self.settings.enable_spatial_cloud_mask,
        )
        records: list[ModelRecord]
        if mode == "manual":
            if not model_id:
                raise ValueError("A model must be selected in manual mode.")
            records = [self.registry.get(model_id)]
        else:
            records = [self.registry.get(model["id"]) for model in self.registry.list_models(auto_only=True)
                       if model["model_type"] != "native_06m"]
            dense_auto_records = [record for record in records if record.model_type in {"dense_unet", "smp_segmentation"}]
            if dense_auto_records and len(dense_auto_records) == len(records):
                records = [self._choose_dense_auto_record(loaded_image, dense_auto_records)]

        if not records:
            raise RuntimeError("No enabled models are available.")

        best_prediction: ModelPrediction | None = None
        candidate_rankings: list[dict[str, Any]] = []
        predictions_by_id: dict[str, ModelPrediction] = {}
        tile_cache: dict[int, TiledImage] = {}

        for record in records:
            if record.model_type == "pan_texture":
                inherit_parent_normalization(loaded_image,[(self.settings.root_dir/path).resolve()
                                             for path in record.summary.get("normalization_sources",[])])
                prediction = self._predict_pan_texture(loaded_image, record, progress)
            elif record.summary.get("inference", {}).get("pipeline") == "pan_probability_overlap_v8":
                prediction = self._predict_pan_overlap(loaded_image, record, progress)
            elif loaded_image.processing_mode == "streaming_raster":
                if record.model_type == "patch_classifier":
                    prediction = self._predict_with_patch_classifier_streaming(loaded_image, record)
                elif record.model_type == "unet":
                    prediction = self._predict_with_unet_streaming(loaded_image, record)
                elif record.model_type in {"dense_unet", "smp_segmentation"}:
                    prediction = self._predict_with_dense_unet_streaming(loaded_image, record)
                else:
                    raise ValueError(f"Unsupported model type: {record.model_type}")
            else:
                tiled = tile_cache.get(record.patch_size)
                if tiled is None:
                    assert loaded_image.grayscale is not None
                    assert loaded_image.valid_mask is not None
                    tiled = tile_image(loaded_image.grayscale, loaded_image.valid_mask, patch_size=record.patch_size)
                    tile_cache[record.patch_size] = tiled
                if record.model_type == "patch_classifier":
                    prediction = self._predict_with_patch_classifier(loaded_image, tiled, record)
                elif record.model_type == "unet":
                    prediction = self._predict_with_unet(loaded_image, tiled, record)
                elif record.model_type in {"dense_unet", "smp_segmentation"}:
                    prediction = self._predict_with_dense_unet(loaded_image, tiled, record)
                else:
                    raise ValueError(f"Unsupported model type: {record.model_type}")
            candidate_entry = {
                "model_id": record.id,
                "display_name": record.display_name,
                "dataset_name": record.dataset_name,
                "model_type": record.model_type,
                "score": round(prediction.selected_score, 4),
                "dominant_class": prediction.dominant_class_name,
                "primary_metric_label": record.primary_metric_label,
                "validation_accuracy": round(float(record.model_metrics.get("val_accuracy", 0.0) or 0.0), 4),
                "secondary_metric_label": record.secondary_metric_label,
                "secondary_metric_value": round(float(record.secondary_metric_value or 0.0), 4)
                if record.secondary_metric_value is not None
                else None,
                "score_breakdown": prediction.score_breakdown,
            }
            candidate_rankings.append(candidate_entry)
            predictions_by_id[record.id] = prediction

            if best_prediction is None or prediction.selected_score > best_prediction.selected_score:
                best_prediction = prediction

        assert best_prediction is not None
        if mode == "auto" and len(candidate_rankings) > 1:
            ranked_by_domain = sorted(
                candidate_rankings,
                key=lambda item: item["score_breakdown"]["domain_similarity"],
                reverse=True,
            )
            domain_gap = ranked_by_domain[0]["score_breakdown"]["domain_similarity"] - ranked_by_domain[1]["score_breakdown"]["domain_similarity"]
            if domain_gap >= 0.04:
                allowed_ids = {
                    item["model_id"]
                    for item in candidate_rankings
                    if item["score_breakdown"]["domain_similarity"] >= ranked_by_domain[0]["score_breakdown"]["domain_similarity"] - 0.02
                }
                filtered_rankings = [item for item in candidate_rankings if item["model_id"] in allowed_ids]
                if filtered_rankings:
                    candidate_rankings = filtered_rankings
                    best_prediction = max(
                        (predictions_by_id[item["model_id"]] for item in candidate_rankings),
                        key=lambda pred: pred.selected_score,
                    )

        candidate_rankings.sort(key=lambda item: item["score"], reverse=True)
        if mode == "auto" and candidate_rankings:
            best_prediction = predictions_by_id[candidate_rankings[0]["model_id"]]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        result_dir = ensure_dir(self.settings.result_dir / result_id)

        return self._save_outputs(
            result_dir=result_dir,
            loaded_image=loaded_image,
            prediction=best_prediction,
            mode=mode,
            candidate_rankings=candidate_rankings,
            requested_model_id=model_id,
        )
