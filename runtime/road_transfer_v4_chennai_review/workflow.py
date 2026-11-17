"""Reserve fresh evaluation geometry and prepare image-only training candidates."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import Window, bounds
from shapely.geometry import box

from road_network_v1.audit import ROOT, OLD_RUN, digest


HOME = Path(__file__).resolve().parent
SCENES = ROOT / "native_06m_specialist/config/multispectral_scenes.json"
OLD_LOCK = ROOT / "road_network_v1/review/main_roads_review_v2_alignment_fixed/PROJECT_LOCK.json"
POOL = ROOT / "road_network_v1/review/candidates_v1/candidate_manifest.json"
BUFFER_M = 300.


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def iter_records(value):
    if isinstance(value, dict):
        if any(key in value for key in ("source_pixel_window", "source_window", "window_xywh")):
            yield value
        for child in value.values():
            yield from iter_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_records(child)


def collect_exclusions():
    """Conservatively exclude even old unreviewed/weak proposals, without reading masks."""
    paths = sorted((ROOT / "native_06m_specialist/data").glob("**/*manifest*.json"))
    paths += [OLD_RUN / "training_setup.json", OLD_LOCK]
    unique, evidence = {}, {}
    for path in paths:
        data = json.loads(path.read_text())
        for row in iter_records(data):
            window = next((row[key] for key in ("source_pixel_window", "source_window", "window_xywh")
                           if key in row), None)
            if not isinstance(window, list) or len(window) != 4:
                raise ValueError(f"Unrecognized historical window in {path}")
            scene = str(row.get("source_scene_id", row.get("scene_id", "")))
            location = row.get("geographic_group", "")
            source = str(row.get("source_image", row.get("source", "")))
            if scene.startswith("chennai_06m_001"):
                scene = "267372791"
            if scene not in ("267372781", "2673727101", "267372791"):
                scene = next((sid for sid in ("2673727101", "267372781", "267372791")
                              if sid in source), "")
            if not scene:
                scene = {"pune_west_20260402": "267372781", "pune_central_20250216": "2673727101",
                         "chennai_20260106": "267372791"}.get(location, "")
            if not scene:
                raise ValueError(f"Cannot identify historical source scene in {path}")
            key = (scene, *window)
            unique[key] = {"source_scene_id": scene, "source_window": window}
            evidence[str(path)] = digest(path)
    return list(unique.values()), evidence


def metric_footprint(record):
    with rasterio.open(record["source"]) as src:
        if not src.crs or not src.crs.is_projected or src.crs.linear_units != "metre":
            raise ValueError("A projected metre grid is required")
        if not np.allclose(src.res, [.6, .6], rtol=0, atol=1e-6):
            raise ValueError("Not a nominal 0.6 m delivered grid")
        if src.transform.b or src.transform.d:
            raise ValueError("Rotated grids require polygon footprints, not bounding boxes")
        x, y, width, height = record["source_window"]
        if min(x, y) < 0 or min(width, height) <= 0 or x+width > src.width or y+height > src.height:
            raise ValueError("Tile extends outside source image")
        return box(*bounds(Window(x, y, width, height), src.transform))


def nearest_distance(record, others):
    own = metric_footprint(record)
    distances = [own.distance(metric_footprint({**other, "source": record["source"]})) for other in others
                 if other["source_scene_id"] == record["source_scene_id"]]
    return min(distances) if distances else None


def validate_separation(records, exclusions, reserved=(), minimum=BUFFER_M):
    for row in records:
        distance = nearest_distance(row, [*exclusions, *reserved])
        if distance is not None and distance < minimum:
            raise ValueError(f"{row['id']} violates the {minimum:g} m exclusion buffer ({distance:.2f} m)")
    for index, row in enumerate(records):
        for other in records[index+1:]:
            if row["source_scene_id"] != other["source_scene_id"]:
                continue
            distance = metric_footprint(row).distance(metric_footprint(other))
            required = minimum if row["split"] != other["split"] else 0.
            if distance < required or metric_footprint(row).intersection(metric_footprint(other)).area > 0:
                raise ValueError("New tiles overlap or violate the cross-split buffer")


def verify_reservation(path):
    data = json.loads(Path(path).read_text())
    payload = {key: value for key, value in data.items() if key != "reservation_sha256"}
    if data.get("reservation_sha256") != json_digest(payload):
        raise ValueError("Reserved evaluation geometry or protocol changed")
    return data


def reserve():
    destination = HOME / "RESERVED_HOLDOUT.json"
    if destination.exists():
        return verify_reservation(destination)
    exclusions, evidence = collect_exclusions()
    pool = json.loads(POOL.read_text())["tiles"]
    rng = np.random.default_rng(20261023)
    held, considered = [], []
    for sid in ("267372781", "2673727101", "267372791"):
        eligible = []
        for row in sorted(pool, key=lambda item: item["id"]):
            if row["source_scene_id"] != sid:
                continue
            distance = nearest_distance(row, exclusions)
            considered.append({"id": row["id"], "minimum_old_footprint_distance_m": distance})
            if distance is None or distance >= BUFFER_M:
                eligible.append(row)
        if not eligible:
            raise ValueError(f"No fresh reserved candidate for {sid}")
        row = eligible[int(rng.integers(len(eligible)))]
        with rasterio.open(row["source"]) as src:
            grid = {"crs": str(src.crs), "transform": list(src.transform), "shape": list(src.shape),
                    "resolution_m": list(src.res)}
        held.append({**row, "id": "SEALED_"+sid, "candidate_id": row["id"], "split": "sealed_holdout",
                     "source_vrt_sha256": digest(row["source"]), "source_grid": grid,
                     "training_allowed": False, "labels": None})
    validate_separation(held, exclusions)
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "One reserved, road-enriched within-scene audit tile per source scene; not a new-city test",
              "selection": "Seed 20261023, one uniform draw per scene from the pre-existing IMAGE-ONLY line-enriched candidate pool after conservative spatial exclusions",
              "selection_uses_model_predictions": False, "seed": 20261023,
              "minimum_exclusion_distance_m": BUFFER_M, "tiles": held,
              "historical_exclusion_count": len(exclusions), "historical_manifest_sha256": evidence,
              "candidate_pool_sha256": digest(POOL), "considered_candidates": considered,
              "label_policy": "Review image-only surface labels after the next model and decoder are frozen; never show model predictions while annotating the audit",
              "release_policy": "No audit pixel extraction, predictions or labels in active training/model selection. Freeze model and decoder hashes before a separate one-time audit. No retuning on that audit.",
              "limitations": ["Only three road-enriched tiles, not an area-representative sample",
                              "Two cities; previous platform inference and label-free adaptation saw these source scenes",
                              "Fresh spatially buffered reference, NOT fully unseen imagery or an independent new-city guarantee"],
              "audit_images_extracted": False, "audit_predictions_run": False, "audit_labels_reviewed": False,
              "production_changed": False, "shapefiles_generated": False}
    result["reservation_sha256"] = json_digest(result)
    destination.write_text(json.dumps(result, indent=2))
    return result


def export_candidate(row, output):
    folder = output / row["id"]
    folder.mkdir()
    with rasterio.open(row["source"]) as src:
        window = Window(*row["source_window"])
        raw = src.read(window=window)
        valid = np.any(raw != 0, axis=0) & (src.read_masks(window=window) > 0).all(0)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(driver="GTiff", width=int(window.width), height=int(window.height), count=4,
                       transform=transform, compress="lzw", tiled=True, blockxsize=256, blockysize=256)
        crs, res = str(src.crs), list(src.res)
    image = folder / "four_band_06m.tif"
    with rasterio.open(image, "w", **profile) as dst:
        dst.write(raw)
        for band, name in enumerate(("Blue", "Green", "Red", "NIR"), 1):
            dst.set_band_description(band, name)
    limits = np.asarray(row["normalization_limits"], np.float32)
    scaled = np.clip((raw-limits[:, 0, None, None])/(limits[:, 1, None, None]-limits[:, 0, None, None]), 0, 1)
    for name, channels in (("RGB", [2, 1, 0]), ("NIR", [3, 2, 1])):
        rgb = (scaled[channels].transpose(1, 2, 0)*255).astype(np.uint8)
        rgb[~valid] = 0
        Image.fromarray(rgb).save(folder / (name+".png"))
    return {**row, "image": str(image), "image_sha256": digest(image), "crs": crs,
            "transform": list(transform), "resolution_m": res, "valid_fraction": float(valid.mean()),
            "training_allowed": False, "label_provenance": "image_only_pending_assistant_draft_and_human_review"}


def candidates():
    reservation = reserve()
    exclusions, evidence = collect_exclusions()
    destination = HOME / "candidates_20261023"
    destination.mkdir(exist_ok=False)
    scenes = json.loads(SCENES.read_text())["scenes"]
    scene = next(row for row in scenes if row["scene_id"] == "267372791")
    old = json.loads((OLD_RUN / "training_setup.json").read_text())["manifest"]["scenes"]
    limits = next(row["normalization_limits"] for row in old if row["source_scene_id"] == scene["scene_id"])
    proposals = [("flyover_north", 15700, 4700), ("north_south_arterial", 17000, 4400),
                 ("southern_corridor", 16100, 8000), ("divided_junction", 14000, 13400),
                 ("curved_main_road", 1000, 13530), ("rail_roof_negative", 6140, 8518),
                 ("flyover_south", 17250, 7350), ("junction_east", 16400, 8700)]
    records, rejected = [], []
    for name, x, y in proposals:
        row = {"id": "CHENNAI_"+name, "source_scene_id": scene["scene_id"],
               "geographic_group": scene["geographic_group"], "source": scene["images"][0],
               "source_window": [x, y, 768, 768], "normalization_limits": limits, "split": "train"}
        try:
            validate_separation([row], exclusions, reservation["tiles"])
        except ValueError as exc:
            rejected.append({"id": row["id"], "reason": str(exc)})
            continue
        records.append(export_candidate(row, destination))
    canvas = Image.new("RGB", (1536, ((len(records)+2)//3)*550), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(records):
        x, y = i % 3*512, i//3*550
        canvas.paste(Image.open(Path(row["image"]).parent / "RGB.png").resize((512, 512)), (x, y+38))
        draw.text((x+5, y+7), row["id"], fill="black")
    canvas.save(destination / "IMAGE_ONLY_CONTACT.png")
    (destination / "CANDIDATES.json").write_text(json.dumps({"tiles": records, "rejected": rejected,
        "reservation_sha256": reservation["reservation_sha256"], "exclusion_manifest_sha256": evidence,
        "models_run": False, "reference_labels_created": False}, indent=2))
    print(json.dumps({"prepared": [row["id"] for row in records], "rejected": rejected,
                      "holdout_reserved_not_extracted": len(reservation["tiles"])}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["candidates", "reserve"])
    args = parser.parse_args()
    if args.command == "candidates":
        candidates()
    else:
        print(json.dumps(reserve(), indent=2))
