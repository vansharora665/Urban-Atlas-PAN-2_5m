"""Isolated worker using the original frozen implementation and components."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def materialize(config):
    import torch
    weights = HERE / "model_weights.pth"
    if digest(weights) != config["weights_sha256"]:
        raise ValueError("Bundle checksum mismatch")
    bundle = torch.load(weights, map_location="cpu", weights_only=True)
    if bundle.get("format") != "urban_atlas_original_components_v1":
        raise ValueError("Not the expected frozen-component bundle")
    home = HERE / ".component_cache" / config["weights_sha256"][:16]
    project = home / "project"
    # Each worker has its own import namespace; no dependence on the BTP checkout.
    shutil.copytree(HERE / "runtime", project, dirs_exist_ok=True)
    verified = []
    for row in bundle["components"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe component path")
        value = row["data"]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("Expected an ordinary byte tensor, not executable checkpoint objects")
        raw = value.numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Component content checksum mismatch")
        target = project / relative
        if not target.is_file() or digest(target) != row["sha256"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".writing")
            temporary.write_bytes(raw)
            temporary.replace(target)
        verified.append({"path": row["path"], "sha256": row["sha256"]})
    for path in [project, project / "urban_feature_platform", project / "native_06m_specialist/src"]:
        sys.path.insert(0, str(path))
    return project, verified


def settings_for(project, output):
    from platform_app.config import Settings
    platform = project / "urban_feature_platform"
    runtime = output / "runtime"
    for path in [runtime, output / "results", output / "uploads"]:
        path.mkdir(parents=True, exist_ok=True)
    return replace(Settings(), root_dir=platform, registry_path=platform / "config/model_registry.json",
                   runtime_dir=runtime, upload_dir=output / "uploads", result_dir=output / "results",
                   road_head_model_path=project / "model_rebuild_v2/outputs/pan_road_continuity_v9/best_model.pt",
                   export_shapefiles_on_predict=False)


def verify_models(project, config, settings, registry):
    import numpy as np
    import torch
    import segmentation_models_pytorch as smp
    record = registry.get(config["model_id"])
    details = []
    if config["family"] == "pan_2_5m":
        from platform_app.pan_texture_net import PanTextureNet
        bundle = record.load_bundle()
        state = {key: torch.from_numpy(value) for key, value in bundle["student_state"].items()}
        model = PanTextureNet(state["mean"], state["scale"],
                              hidden=(state["layers.0.weight"].shape[0], state["layers.3.weight"].shape[0]))
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.inference_mode():
            assert torch.isfinite(model(torch.zeros(2, 16))).all()
        assert np.isfinite(bundle["classifier"].predict_proba(np.zeros((2, 16), np.float32))).all()
        details.append("Texture teacher and student: strict restore / finite output")
        checkpoint = torch.load(settings.road_head_model_path, map_location="cpu", weights_only=False)
        model = smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=1, classes=5)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        with torch.inference_mode():
            assert torch.isfinite(model(torch.zeros(1, 1, 64, 64))).all()
        details.append("SegFormer V9 roads: strict restore / finite output")
    else:
        from platform_app.native06m_inference import Native06mInference
        model, device = Native06mInference(settings).load_model(record)
        with torch.inference_mode():
            out = model(torch.zeros(1, 4, 64, 64, device=device))
            assert out["semantic"].shape == (1, 5, 64, 64)
            assert all(torch.isfinite(value).all() for value in out.values())
        details.append("Four-band attention U-Net++: strict restore / finite output")
        from road_transfer_v3_coanet.model import load_checkpoint
        from road_transfer_v5_mainroad_precision.full_scene import surface_only
        model, _ = load_checkpoint(project / "road_transfer_v5_mainroad_precision/experiments/precision_surface_512_v1/BEST_DEVELOPMENT_WEIGHTS.pt")
        model.eval()
        with torch.inference_mode():
            assert torch.isfinite(surface_only(model, torch.zeros(1, 3, 64, 64))).all()
        details.append("CoANet roads: strict restore / finite output")
    return {"status": "PASS", "model": config["display_name"], "checks": details,
            "accuracy_test": False, "scope": "Checkpoint restoration and synthetic finite-output smoke checks only"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--roads", action="store_true")
    p.add_argument("--vectors", action="store_true")
    p.add_argument("--zones", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = p.parse_args()
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    if args.device == "cpu":
        # The isolated worker contains this override; the host platform is unaffected.
        torch.cuda.is_available = lambda: False
        torch.backends.mps.is_available = lambda: False
    config = json.loads((HERE / "model_config.json").read_text())
    project, verified = materialize(config)
    output = args.output.resolve() if args.output else HERE / ".component_cache/verification"
    if args.output and output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty output directory; results are never overwritten")
    settings = settings_for(project, output)
    from platform_app.model_registry import ModelRegistry
    registry = ModelRegistry(settings)
    if args.verify:
        result = verify_models(project, config, settings, registry)
        result["original_components_verified"] = verified
    else:
        import numpy as np
        import rasterio
        if not args.input or not args.input.is_file():
            raise FileNotFoundError("Provide an existing GeoTIFF")
        with rasterio.open(args.input) as src:
            if src.driver != "GTiff" or src.count != config["input_channels"]:
                raise ValueError("TIFF band count does not match this model's declared input")
            if not src.crs or not src.crs.is_projected or src.crs.linear_units != "metre":
                raise ValueError("Use a projected, metre-based GeoTIFF")
            if not all(np.isclose(v, config["resolution_m"], atol=config["resolution_tolerance_m"]) for v in src.res):
                raise ValueError("Resolution does not match this model; resample deliberately, not silently")
        uploaded = settings.upload_dir / "input.tif"
        try:
            os.link(args.input.resolve(), uploaded)
        except OSError:
            shutil.copy2(args.input, uploaded)
        from platform_app.inference import InferenceService
        service = InferenceService(settings, registry)
        progress = lambda value, message: print(f"{value:.0%} {message}", file=sys.stderr, flush=True)
        result = service.predict_file(uploaded, mode="manual", model_id=config["model_id"],
                                      band_order="bgrn" if config["input_channels"] == 4 else None,
                                      progress=progress)
        from platform_app.cartography import remember_input
        remember_input(settings, result["result_id"], uploaded)
        if args.roads and config["family"] == "native_0_6m":
            result["model_roads"] = service.predict_model_roads(result["result_id"], progress)
        if args.vectors:
            result["five_class_shapefiles"] = service.generate_shapefiles_for_result(result["result_id"])
            if result.get("model_roads"):
                result["road_shapefiles"] = service.export_model_roads(result["result_id"])
        if args.zones:
            if config["family"] != "native_0_6m":
                raise ValueError("Neighbourhood head applies only to the four-band model")
            from platform_app.cartography import build as build_usable
            result["usable_map"] = build_usable(settings, result["result_id"], progress)
            from platform_app.structured_zones import build
            result["structured_zones"] = build(settings, registry, result["result_id"], progress)
        result["output_directory"] = str(settings.result_dir / result["result_id"])
        result["prediction_is_ground_truth"] = False
        (output / "prediction_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
