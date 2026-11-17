"""Reproducible frozen-output road ablations on the existing development labels."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from scipy import ndimage

from .network import NetworkConfig, decode_network, road_metrics, thin


ROOT = Path(__file__).resolve().parents[1]
OLD_RUN = ROOT / "native_06m_specialist/runs/native06m_multilocation_v4_road_only_20261021_010815"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def aggregate(rows):
    sums = {name: sum(row[name] for row in rows) for name in
            ("true_positive", "false_positive", "false_negative", "evaluated_pixels", "reference_road_pixels")}
    tp, fp, fn = (sums[k] for k in ("true_positive", "false_positive", "false_negative"))
    return {**sums, "road_iou": tp/(tp+fp+fn) if tp+fp+fn else None,
            "road_precision": tp/(tp+fp) if tp+fp else None,
            "road_recall": tp/(tp+fn) if tp+fn else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    setup = json.loads((OLD_RUN / "training_setup.json").read_text())
    existing = json.loads((OLD_RUN / "platform_adapter_audit/results.json").read_text())
    registry = ROOT / "urban_feature_platform/config/model_registry.json"
    unchanged = {str(registry): digest(registry)}
    cached, inventory = [], []
    for scene in setup["manifest"]["scenes"]:
        unchanged[scene["labels"]] = digest(scene["labels"])
        candidate = next(row for row in existing["rows"]["candidate"] if row["scene_id"] == scene["scene_id"])
        baseline = next(row for row in existing["rows"]["baseline"] if row["scene_id"] == scene["scene_id"])
        directory = OLD_RUN / "platform_adapter_audit/predictions" / candidate["result_id"]
        previous = OLD_RUN / "platform_adapter_audit/predictions" / baseline["result_id"]
        with rasterio.open(scene["labels"]) as src:
            original, gsd, transform, crs = src.read(1), max(src.res), src.transform, src.crs
        target = np.where(original == 255, 255, original == 1).astype(np.uint8)
        with rasterio.open(directory / "road_model_evidence_georef.tif") as src:
            evidence = src.read()
            assert (src.shape, src.crs, src.transform) == (original.shape, crs, transform)
        with rasterio.open(directory / "segmentation_labels_georef.tif") as src:
            deployed = src.read(1)
        with rasterio.open(previous / "segmentation_labels_georef.tif") as src:
            default = src.read(1)
        with rasterio.open(directory / "segmentation_labels_base_georef.tif") as src:
            base = src.read(1)
        probability = evidence[0].astype(np.float32)/255
        allowed = np.isin(base, [1, 2, 3]) & (deployed != 255)
        roads = original == 1
        skeleton = thin(roads)
        widths = 2*ndimage.distance_transform_edt(np.pad(roads, 1))[1:-1, 1:-1]*gsd
        known = original != 255
        row = {"tile_id": scene["scene_id"], "split": scene["split"], "location": scene["geographic_group"],
               "pixels": original.size, "reviewed_pixels": int(known.sum()),
               "road_pixels": int(roads.sum()), "unknown_fraction": float((~known).mean()),
               "road_core_width_p10_p50_p90_m": np.percentile(widths[skeleton], [10, 50, 90]).tolist() if skeleton.any() else None,
               "width_caveat": "Sparse annotation-stroke widths, NOT independently measured physical road widths",
               "road_probability_on_reviewed_roads_p10_p50_p90": np.percentile(probability[roads], [10, 50, 90]).tolist() if roads.any() else None,
               "candidate_road_error_to_class": {str(c): int((roads & (deployed == c)).sum()) for c in [0, 1, 2, 3, 4, 255]},
               "false_road_reference_class": {str(c): int(((deployed == 1) & (original == c)).sum()) for c in [0, 2, 3, 4]},
               "road_probability_blocked_by_base_class_pixels": int((roads & ~allowed).sum()),
               "reference_label_sha256": unchanged[scene["labels"]]}
        inventory.append(row)
        cached.append({"scene": scene, "target": target, "probability": probability, "allowed": allowed,
                       "default": default == 1, "candidate": deployed == 1,
                       "image": Image.open(directory / "input_preview.png").convert("RGB"),
                       "gsd": gsd, "transform": transform, "crs": crs})
    train = [entry for entry in cached if entry["scene"]["split"] == "train"]
    thresholds = []
    for threshold in (.25, .4, .55, .7, .85):
        metrics = [road_metrics((entry["probability"] >= threshold) & entry["allowed"], entry["target"], entry["gsd"])
                   for entry in train]
        thresholds.append({"threshold": threshold, "training_metrics": aggregate(metrics)})
    threshold = max(thresholds, key=lambda r: r["training_metrics"]["road_iou"] or 0)["threshold"]
    recipes = [NetworkConfig(seed_probability=.6, support_probability=.3, maximum_gap_m=0.),
               NetworkConfig(seed_probability=.6, support_probability=.3, maximum_gap_m=9.),
               NetworkConfig(seed_probability=.75, support_probability=.45, maximum_gap_m=9.)]
    recipe_rows = []
    for recipe in recipes:
        metrics = []
        for entry in train:
            mask, _, _ = decode_network(entry["probability"], entry["allowed"], entry["gsd"], recipe)
            metrics.append(road_metrics(mask, entry["target"], entry["gsd"]))
        recipe_rows.append({"recipe": asdict(recipe), "training_metrics": aggregate(metrics)})
    selected_recipe = max(recipe_rows, key=lambda r: r["training_metrics"]["road_iou"] or 0)["recipe"]
    rows = []
    for entry in cached:
        mask, skeleton, graph = decode_network(entry["probability"], entry["allowed"], entry["gsd"],
                                                selected_recipe, entry["transform"], entry["crs"])
        predictions = {"live_default": entry["default"], "previous_v4": entry["candidate"],
                       "threshold_only": (entry["probability"] >= threshold) & entry["allowed"], "graph_only": mask}
        record = {"tile_id": entry["scene"]["scene_id"], "split": entry["scene"]["split"],
                  "location": entry["scene"]["geographic_group"],
                  "methods": {name: road_metrics(p, entry["target"], entry["gsd"]) for name, p in predictions.items()},
                  "network": {k: graph[k] for k in ("total_length_m", "rejected_components")},
                  "supported_gaps": len(graph["gaps"])}
        rows.append(record)
        canvas = Image.new("RGB", (1536, 560), "white")
        draw = ImageDraw.Draw(canvas)
        rgb = np.asarray(entry["image"].resize(entry["target"].shape[::-1]))
        for index, (name, value) in enumerate((("Sparse reviewed roads", entry["target"] == 1),
                                               ("Previous V4", entry["candidate"]), ("Graph-only ablation", mask))):
            view = rgb.copy()
            view[value] = (.35*view[value]+.65*np.array([255, 100, 20])).astype(np.uint8)
            canvas.paste(Image.fromarray(view).resize((512, 512)), (512*index, 48))
            draw.text((512*index+10, 12), name+" | "+record["split"], fill="black")
        canvas.save(output / (record["tile_id"]+"_ablation.png"))
        (output / (record["tile_id"]+"_image_graph.json")).write_text(json.dumps(graph))
        print(json.dumps({"tile": record["tile_id"], "graph_iou": record["methods"]["graph_only"]["road_iou"]}), flush=True)
    validation = [row for row in rows if row["split"] == "validation"]
    methods = list(rows[0]["methods"])
    summary = {name: aggregate([row["methods"][name] for row in validation]) for name in methods}
    per_location = {group: {name: aggregate([row["methods"][name] for row in validation if row["location"] == group])
                            for name in methods} for group in sorted({r["location"] for r in validation})}
    report = {"scope": "Existing sparse ALL-road development references; NOT the new main-road task, not blind test",
              "source_run": str(OLD_RUN), "inventory": inventory, "rows": rows,
              "selection": {"scope": "Training labels only; no validation tuning", "threshold": threshold,
                            "threshold_candidates": thresholds, "graph_candidates": recipe_rows, "graph_recipe": selected_recipe},
              "validation": summary, "per_location": per_location,
              "human_labels_rewritten": False, "width_filtered_reference": False,
              "default_promoted": False, "road_40_percent_target_met": False,
              "runtime_seconds": time.monotonic()-started,
              "known_limitations": ["Annotation strokes do not identify a main-road benchmark.",
                 "Old references and earlier parent training have been inspected; this is development evidence only.",
                 "Protecting erroneous base water/vegetation also blocks some real roads; geometry cannot settle that ambiguity.",
                 "APLS is withheld until a reviewed junction graph exists."]}
    assert all(digest(path) == value for path, value in unchanged.items())
    report["live_registry_and_reference_hashes_unchanged"] = True
    (output / "ROAD_DIAGNOSIS_AND_ABLATIONS.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"validation": summary, "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
