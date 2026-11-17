"""Native-grid, bounded-memory whole-TIFF road inference; no model promotion."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import torch

from road_network_v1.audit import ROOT, digest
from road_transfer_v3_coanet.model import load_checkpoint, prepare_input
from road_transfer_v4_chennai_review.workflow import json_digest, verify_reservation
from .workflow import HOME, RESERVED, SCENES


def starts(length, window, stride):
    positions = list(range(0, max(1, length - window + 1), stride))
    last = max(0, length - window)
    if positions[-1] != last:
        positions.append(last)
    return positions


def stream_probabilities(width, height, read_patch, predict, emit, window=768, stride=576, batch_size=2, progress=None):
    """Same Hann product/floor as predict_pack, keeping only one window stripe."""
    if min(width, height, window, stride, batch_size) <= 0 or not 0 < stride <= window:
        raise ValueError("Positive dimensions and a stride no larger than the window are required")
    xs, ys = starts(width, window, stride), starts(height, window, stride)
    kernel = np.maximum(np.outer(np.hanning(window), np.hanning(window)), .05).astype(np.float32)
    total = np.zeros((window, width), np.float32)
    weights = np.zeros_like(total)
    origin, completed = 0, 0
    for y in ys:
        delta = y - origin
        if delta:
            covered = weights[:delta] > 0
            emit(origin, total[:delta] / np.maximum(weights[:delta], 1e-8), covered)
            total[:window-delta] = total[delta:].copy()
            total[window-delta:] = 0
            weights[:window-delta] = weights[delta:].copy()
            weights[window-delta:] = 0
            origin = y
        for offset in range(0, len(xs), batch_size):
            columns = xs[offset:offset+batch_size]
            images, valid = zip(*(read_patch(x, y) for x in columns))
            probabilities = predict(np.stack(images))
            if probabilities.shape != (len(columns), window, window) or not np.isfinite(probabilities).all():
                raise ValueError("Invalid full-scene probability batch")
            for x, probability, mask in zip(columns, probabilities, valid):
                h, w = min(window, height-y), min(window, width-x)
                taper = kernel[:h, :w] * mask[:h, :w]
                total[:h, x:x+w] += probability[:h, :w] * taper
                weights[:h, x:x+w] += taper
            completed += len(columns)
        if progress:
            progress(completed, len(xs)*len(ys))
    remaining = height - origin
    emit(origin, total[:remaining] / np.maximum(weights[:remaining], 1e-8), weights[:remaining] > 0)


def surface_only(model, tensor):
    e1, e2, e3, e4 = model.backbone(tensor)
    decoded = model.decoder(e1, e2, e3, model.aspp(e4))
    return model.connect.seg_branch(decoded).sigmoid()


def pad_native(array, window):
    h, w = array.shape[-2:]
    if h > window or w > window:
        raise ValueError("Patch exceeds inference window")
    padding = [(0, 0)] * (array.ndim - 2) + [(0, window-h), (0, window-w)]
    return np.pad(array, padding, mode="reflect" if min(h, w) > 1 else "edge")


def render_preview(source_path, labels_path, limits, output, name, maximum=3200):
    with rasterio.open(source_path) as src, rasterio.open(labels_path) as labels:
        factor = maximum / max(src.width, src.height)
        shape = (max(1, round(src.height*factor)), max(1, round(src.width*factor)))
        raw = src.read(out_shape=(4, *shape), resampling=Resampling.bilinear).astype(np.float32)
        mask = labels.read(1, out_shape=shape, resampling=Resampling.nearest)
    scaled = np.clip((raw-limits[:, 0, None, None])/(limits[:, 1, None, None]-limits[:, 0, None, None]), 0, 1)
    rgb = (scaled[[2, 1, 0]].transpose(1, 2, 0)**.65*255).astype(np.uint8)
    rgb[mask == 255] = [38, 48, 55]
    marked = rgb.copy()
    roads = mask == 1
    marked[roads] = (.35*marked[roads]+.65*np.array([255, 105, 25])).astype(np.uint8)
    colored = np.full_like(rgb, [31, 42, 49])
    colored[roads] = [255, 125, 35]
    colored[mask == 255] = [80, 90, 98]
    Image.fromarray(rgb).save(output / "input_full_scene.png")
    Image.fromarray(marked).save(output / "roads_overlay_full_scene.png")
    Image.fromarray(colored).save(output / "roads_mask_full_scene.png")
    display_width = 1400
    display_height = round(shape[0]*display_width/shape[1])
    canvas = Image.new("RGB", (display_width*2+72, display_height+210), "#f4f6f3")
    draw = ImageDraw.Draw(canvas)
    font = "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf"
    title = ImageFont.truetype(font, 34)
    small = ImageFont.truetype(font, 24)
    draw.text((24, 18), name + " | WHOLE SCENE, not a crop", font=title, fill="#142b35")
    draw.text((24, 66), "Native-resolution inference; orange marks model-predicted roads. Experimental road-only candidate, not deployed.", font=small, fill="#536b75")
    draw.text((24, 108), "Original", font=small, fill="#142b35")
    draw.text((48+display_width, 108), "Predicted roads over original image", font=small, fill="#142b35")
    canvas.paste(Image.fromarray(rgb).resize((display_width, display_height)), (24, 146))
    canvas.paste(Image.fromarray(marked).resize((display_width, display_height)), (48+display_width, 146))
    draw.text((24, display_height+164), "Overview is downsampled for display only. No smoothing, gap filling, forced road width or hand-drawn labels.", font=small, fill="#536b75")
    canvas.save(output / "WHOLE_SCENE_ORIGINAL_VS_ROADS.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("267372781", "2673727101", "267372791"), default="267372791")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HOME.resolve()) or output.exists():
        raise ValueError("Use a new distinct full-scene output folder inside V5")
    scene = next(row for row in json.loads(SCENES.read_text())["scenes"] if row["scene_id"] == args.scene)
    experiment = HOME / "experiments/precision_surface_512_v1"
    summary = json.loads((experiment / "DEVELOPMENT_SUMMARY.json").read_text())
    setup = json.loads((experiment / "TRAINING_SETUP.json").read_text())
    checkpoint = experiment / "BEST_DEVELOPMENT_WEIGHTS.pt"
    if digest(checkpoint) != summary["checkpoint_sha256"] or summary["selected"]["threshold"] != .1:
        raise ValueError("Retained road candidate or its frozen threshold changed")
    guards = {**setup["production_guards"], **setup["reference_hashes"]}
    if not all(digest(path) == value for path, value in guards.items()):
        raise ValueError("Protected models/references changed")
    old_scenes = json.loads((ROOT / "native_06m_specialist/runs/native06m_multilocation_v4_road_only_20261021_010815/training_setup.json").read_text())["manifest"]["scenes"]
    limits = np.asarray(next(row["normalization_limits"] for row in old_scenes if row["source_scene_id"] == args.scene), np.float32)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if args.require_mps and device.type != "mps":
        raise RuntimeError("MPS required; no silent CPU whole-scene inference")
    model, verification = load_checkpoint(checkpoint)
    model.to(device).eval()
    output.mkdir()
    source_path = Path(scene["images"][0])
    reservation = verify_reservation(RESERVED)
    exposed = [row["id"] for row in reservation["tiles"] if row["source_scene_id"] == args.scene]
    ledger = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_scene_id": args.scene,
              "reason": "User requested complete TIFF-scene visualization after tile examples",
              "reserved_prediction_geometry_exposed_by_this_full_scene_run": exposed,
              "reserved_reference_labels_opened": False, "reserved_lock_modified": False,
              "future_claim_boundary": "These scene pixels now have displayed model predictions. Do not claim they stayed prediction-blind for a future reference review.",
              "production_changed": False}
    (output / "FULL_SCENE_EXPOSURE_LEDGER.json").write_text(json.dumps(ledger, indent=2))
    with rasterio.Env(GDAL_CACHEMAX=128*1024*1024), rasterio.open(source_path) as src:
        width, height = src.width, src.height
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="lzw",
                       tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        run = {"status": "whole_scene_native_grid_inference_started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_scene_id": args.scene, "source_vrt": str(source_path), "source_files": list(src.files),
               "width": width, "height": height, "resolution_m": list(src.res), "crs": str(src.crs),
               "transform": list(src.transform), "model": "V5 CoANet road-only surface candidate",
               "checkpoint_sha256": digest(checkpoint), "threshold": .1, "native_grid_resampling": False,
               "window": 768, "stride": 576, "kernel": "max(outer(hann,hann),0.05)",
               "gamma": 1., "flips": False, "normalization_limits": limits.tolist(), "device": str(device),
               "strict_load_verification": verification, "labels": {"0": "Not predicted as road", "1": "Predicted road", "255": "Source NoData"},
               "training_performed": False, "weights_or_threshold_retuned": False,
               "shapefiles_generated": False, "production_changed": False,
               "audit_reference_labels_loaded": False, "reserved_geometry_exposed": exposed,
               "independent_test_accuracy_claim": None, "code_sha256": digest(Path(__file__))}
        (output / "RUN_SETUP.json").write_text(json.dumps(run, indent=2))

        def read_patch(x, y):
            h, w = min(768, height-y), min(768, width-x)
            window = Window(x, y, w, h)
            raw = src.read(window=window).astype(np.float32)
            valid = (src.read_masks(window=window) > 0).all(0) & np.any(raw != 0, axis=0)
            normalized = np.clip((raw-limits[:, 0, None, None])/(limits[:, 1, None, None]-limits[:, 0, None, None]), 0, 1)
            image = prepare_input(pad_native(normalized, 768))
            return image, np.pad(valid, ((0, 768-h), (0, 768-w)), constant_values=False)

        # Compare the optimized surface-only route against the exact author forward.
        check = json.loads((HOME / "qualitative_preview_20261023_v1/RESULTS.json").read_text())["results"]
        row = next(row for row in check if row["source_scene_id"] == args.scene)
        example, _ = read_patch(*row["source_window"][:2])
        tensor = torch.from_numpy(example[None]).to(device)
        with torch.inference_mode():
            expected = model(tensor)[0].cpu().numpy()
            observed = surface_only(model, tensor).cpu().numpy()
            duplicate = surface_only(model, tensor.repeat(args.batch_size, 1, 1, 1)).cpu().numpy()[:1]
        equivalence = {"surface_only_max_absolute_difference": float(np.abs(expected-observed).max()),
                       "batch_max_absolute_difference": float(np.abs(expected-duplicate).max()),
                       "batch_threshold_disagreement_pixels": int(((expected >= .1) != (duplicate >= .1)).sum())}
        if equivalence["surface_only_max_absolute_difference"] > 1e-6:
            raise ValueError("Surface-only route does not match genuine author output")
        batch_size = args.batch_size if equivalence["batch_max_absolute_difference"] <= 1e-5 and not equivalence["batch_threshold_disagreement_pixels"] else 1
        equivalence["selected_batch_size"] = batch_size
        (output / "SURFACE_EQUIVALENCE.json").write_text(json.dumps(equivalence, indent=2))
        print(json.dumps({"grid": [width, height], "device": str(device), "equivalence": equivalence}), flush=True)
        destination = output / "roads_native_grid.tif"
        counts = {"valid_pixels": 0, "road_pixels": 0, "nodata_pixels": 0}
        next_row = 0
        started = time.monotonic()
        with rasterio.open(destination, "w", **profile) as dst:
            def predict(images):
                with torch.inference_mode():
                    values = surface_only(model, torch.from_numpy(images).to(device))[:, 0].cpu().numpy()
                return np.clip(values, 0, 1)

            def emit(y, probability, valid):
                nonlocal next_row
                if y != next_row:
                    raise AssertionError("Output row missing or written more than once")
                labels = np.where(valid, probability >= .1, 255).astype(np.uint8)
                dst.write(labels, 1, window=Window(0, y, width, labels.shape[0]))
                counts["valid_pixels"] += int(valid.sum())
                counts["road_pixels"] += int((labels == 1).sum())
                counts["nodata_pixels"] += int((~valid).sum())
                next_row += labels.shape[0]

            def progress(done, total):
                result = {"patches_completed": done, "patches_total": total,
                          "percent": round(100*done/total, 1), "elapsed_seconds": round(time.monotonic()-started, 1),
                          "native_rows_finalized": next_row}
                (output / "PROGRESS.json").write_text(json.dumps(result, indent=2))
                print(json.dumps(result), flush=True)

            stream_probabilities(width, height, read_patch, predict, emit, batch_size=batch_size, progress=progress)
            if next_row != height or counts["valid_pixels"]+counts["nodata_pixels"] != width*height:
                raise AssertionError("Incomplete whole-scene output")
            dst.update_tags(checkpoint_sha256=run["checkpoint_sha256"], threshold="0.1", whole_scene="true",
                            native_grid="true", no_gap_filling="true", experimental_road_only="true",
                            labels_are_predictions_not_ground_truth="true")
        inference_seconds = time.monotonic()-started
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    name = {"267372781": "Pune west", "2673727101": "Pune central", "267372791": "Chennai"}[args.scene]
    render_preview(source_path, destination, limits, output, name)
    if not all(digest(path) == value for path, value in guards.items()):
        raise AssertionError("Protected models, references or reservations changed")
    with rasterio.open(source_path) as source, rasterio.open(destination) as predicted:
        if (source.shape, source.crs, source.transform) != (predicted.shape, predicted.crs, predicted.transform):
            raise AssertionError("Whole-scene output is not aligned with the full input TIFFs")
    result = {**run, "status": "whole_scene_prediction_complete_not_deployed", **counts,
              "predicted_road_coverage_percent": 100*counts["road_pixels"]/max(1, counts["valid_pixels"]),
              "coverage_is_not_accuracy": True, "native_rows_written": next_row,
              "inference_seconds": inference_seconds, "total_seconds": time.monotonic()-started,
              "outputs": {"native_road_mask": str(destination), "input_preview": str(output / "input_full_scene.png"),
                          "overlay_preview": str(output / "roads_overlay_full_scene.png"),
                          "comparison": str(output / "WHOLE_SCENE_ORIGINAL_VS_ROADS.png")},
              "production_unchanged": True}
    (output / "RESULTS.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "inference_seconds": inference_seconds,
                      "road_coverage_percent_not_accuracy": result["predicted_road_coverage_percent"],
                      "comparison": result["outputs"]["comparison"]}), flush=True)


if __name__ == "__main__":
    main()
