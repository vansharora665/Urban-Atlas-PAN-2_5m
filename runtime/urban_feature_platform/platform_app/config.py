from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path = Path(__file__).resolve().parents[1]
    registry_path: Path = root_dir / "config" / "model_registry.json"
    runtime_dir: Path = root_dir / "runtime"
    upload_dir: Path = runtime_dir / "uploads"
    result_dir: Path = runtime_dir / "results"
    max_content_length: int = int(os.environ.get("URBAN_MAX_UPLOAD_MB", "1000")) * 1024 * 1024
    preview_max_dimension: int = 1800
    full_visual_pixel_limit: int = 25_000_000
    raster_stream_pixel_limit: int = 40_000_000
    raster_stream_file_size_mb: int = 150
    prediction_batch_patches: int = 64
    raster_stats_sample_max_dimension: int = 2048
    target_inference_gsd_m: float = float(os.environ.get("URBAN_TARGET_GSD_M", "2.5"))
    enable_local_contrast: bool = os.environ.get("URBAN_ENABLE_LOCAL_CONTRAST", "1") == "1"
    enable_spatial_cloud_mask: bool = os.environ.get("URBAN_ENABLE_SPATIAL_CLOUD_MASK", "1") == "1"
    enable_dark_water_override: bool = os.environ.get("URBAN_ENABLE_DARK_WATER_OVERRIDE", "0") == "1"
    enable_statistical_calibration: bool = os.environ.get("URBAN_ENABLE_STATISTICAL_CALIBRATION", "0") == "1"
    enable_water_coherence_cleanup: bool = os.environ.get("URBAN_ENABLE_WATER_CLEANUP", "0") == "1"
    enable_road_specialist: bool = os.environ.get("URBAN_ENABLE_ROAD_SPECIALIST", "0") == "1"
    enable_learned_road_head: bool = os.environ.get("URBAN_ENABLE_LEARNED_ROAD_HEAD", "1") == "1"
    road_head_model_path: Path = Path(os.environ.get(
        "URBAN_ROAD_HEAD_MODEL_PATH",
        str(root_dir.parent / "model_rebuild_v2" / "outputs" / "pan_road_continuity_v9" / "best_model.pt"),
    ))
    road_head_weight: float = float(os.environ.get("URBAN_ROAD_HEAD_WEIGHT", "1.0"))
    road_head_probability_threshold: float = float(os.environ.get("URBAN_ROAD_HEAD_THRESHOLD", "0.75"))
    road_head_strong_threshold: float = float(os.environ.get("URBAN_ROAD_HEAD_STRONG_THRESHOLD", "0.90"))
    road_head_stride_ratio: float = float(os.environ.get("URBAN_ROAD_HEAD_STRIDE_RATIO", "0.75"))
    road_head_bright_guard_fraction: float = float(os.environ.get("URBAN_ROAD_BRIGHT_GUARD_FRACTION", "0.10"))
    road_head_invalid_halo_pixels: int = int(os.environ.get("URBAN_ROAD_INVALID_HALO_PIXELS", "41"))
    enable_road_water_recovery: bool = os.environ.get("URBAN_ENABLE_ROAD_WATER_RECOVERY", "1") == "1"
    road_water_min_probability: float = float(os.environ.get("URBAN_ROAD_WATER_MIN_PROBABILITY", "0.90"))
    road_water_max_base_water_probability: float = float(os.environ.get("URBAN_ROAD_WATER_MAX_BASE_WATER", "0.80"))
    road_water_max_analysis_pixels: int = 80_000_000
    skip_vector_export: bool = os.environ.get("URBAN_SKIP_VECTOR_EXPORT", "0") == "1"
    export_shapefiles_on_predict: bool = os.environ.get("URBAN_EXPORT_SHAPEFILES_ON_PREDICT", "0") == "1"
    vector_sieve_pixels: int = int(os.environ.get("URBAN_VECTOR_SIEVE_PIXELS", "96"))
    validate_zip_integrity: bool = os.environ.get("URBAN_VALIDATE_ZIP_INTEGRITY", "0") == "1"
    secret_key: str = os.environ.get("URBAN_PLATFORM_SECRET", "urban-feature-platform-dev")
    allowed_extensions: tuple[str, ...] = (
        ".tif",
    )
