"""Spatially separate V5 training drafts from every prior reference and audit."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import Window

from road_network_v1.audit import OLD_RUN, ROOT, digest
from road_transfer_v4_chennai_review.workflow import (
    BUFFER_M, POOL, SCENES, collect_exclusions as historical_exclusions,
    json_digest, nearest_distance, validate_separation, verify_reservation,
)


HOME = Path(__file__).resolve().parent
V4 = ROOT / "road_transfer_v4_chennai_review"
RESERVED = HOME / "RESERVED_FUTURE_AUDIT.json"
INVENTORY = HOME / "EXCLUSION_INVENTORY.json"
SIZE = 768


def collect_exclusions():
    records, evidence = historical_exclusions()
    for relative in ("review_chennai_mainroads_v4/PROJECT_LOCK.json", "RESERVED_HOLDOUT.json"):
        path = V4 / relative
        records.extend(json.loads(path.read_text())["tiles"])
        evidence[str(path)] = digest(path)
    completed = V4 / "review_sealed_audit_v1/AUDIT_COMPLETED.json"
    data = json.loads(completed.read_text())
    if data.get("audit_consumed") is not True or data.get("training_allowed") is not False:
        raise ValueError("The completed V4 audit must remain consumed and excluded")
    evidence[str(completed)] = digest(completed)
    result = Path(data["result"])
    if digest(result) != data["result_sha256"]:
        raise ValueError("Completed audit results changed")
    evidence[str(result)] = digest(result)
    unique = {(row["source_scene_id"], *row["source_window"]): {
        "source_scene_id": row["source_scene_id"], "source_window": row["source_window"]}
        for row in records}
    return list(unique.values()), evidence


def reserve():
    if RESERVED.exists():
        return verify_reservation(RESERVED)
    excluded, evidence = collect_exclusions()
    pool = json.loads(POOL.read_text())["tiles"]
    rng = np.random.default_rng(2026102305)
    held = []
    for sid in ("267372781", "2673727101", "267372791"):
        eligible = [row for row in sorted(pool, key=lambda row: row["id"])
                    if row["source_scene_id"] == sid and nearest_distance(row, excluded) >= BUFFER_M]
        if not eligible:
            raise ValueError("No eligible future audit geometry for " + sid)
        row = eligible[int(rng.integers(len(eligible)))]
        held.append({**row, "candidate_id": row["id"], "id": "V5_FUTURE_AUDIT_" + sid,
                     "split": "sealed_holdout", "training_allowed": False, "labels": None})
    validate_separation(held, excluded)
    result = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 2026102305, "tiles": held, "minimum_exclusion_distance_m": BUFFER_M,
        "selection": "Seeded geometry-only draw from the previous image-only candidate pool; no audit scores or masks used",
        "historical_manifest_sha256": evidence, "candidate_pool_sha256": digest(POOL),
        "release_policy": "Do not extract or annotate these tiles for V5 until candidate weights and decoder are frozen. Then evaluate once; no retuning.",
        "audit_images_extracted_for_v5": False, "audit_labels_reviewed": False,
        "audit_model_predictions_run": False, "production_changed": False,
        "scope": "Buffered spatial holdouts in previously seen scenes, not unseen-city imagery or area-representative accuracy",
    }
    result["reservation_sha256"] = json_digest(result)
    with RESERVED.open("x") as handle:
        json.dump(result, handle, indent=2)
    inventory = {"records": excluded, "manifest_sha256": evidence,
                 "minimum_buffer_m": BUFFER_M, "old_audit_is_not_training": True}
    with INVENTORY.open("x") as handle:
        json.dump(inventory, handle, indent=2)
    return result


def verify_inventory():
    frozen = json.loads(INVENTORY.read_text())
    current, evidence = collect_exclusions()
    if evidence != frozen["manifest_sha256"] or current != frozen["records"]:
        raise ValueError("Historical inventory changed; explicitly re-audit spatial separation")
    return frozen


def training_candidates():
    reservation = reserve()
    inventory = verify_inventory()
    destination = HOME / "training_candidates_v1"
    destination.mkdir(exist_ok=False)
    excluded = inventory["records"] + reservation["tiles"]
    scenes = json.loads(SCENES.read_text())["scenes"]
    previous = json.loads((OLD_RUN / "training_setup.json").read_text())["manifest"]["scenes"]
    detector = cv2.createLineSegmentDetector()
    records = []
    for scene in scenes:
        sid = scene["scene_id"]
        limits = np.asarray(next(row["normalization_limits"] for row in previous
                                 if row["source_scene_id"] == sid), np.float32)
        possibilities = []
        with rasterio.open(scene["images"][0]) as src:
            positions = {(int(x), int(y)) for y in np.linspace(750, src.height - SIZE - 750, 10)
                         for x in np.linspace(750, src.width - SIZE - 750, 10)}
            positions.update((row["source_window"][0], row["source_window"][1])
                             for row in json.loads(POOL.read_text())["tiles"] if row["source_scene_id"] == sid)
            for x, y in sorted(positions):
                row = {"id": "candidate", "split": "train", "source_scene_id": sid,
                       "source": scene["images"][0], "source_window": [x, y, SIZE, SIZE]}
                distance = nearest_distance(row, excluded)
                if distance < BUFFER_M:
                    continue
                window = Window(x, y, SIZE, SIZE)
                raw = src.read([1, 2, 3, 4], window=window, out_shape=(4, 256, 256)).astype(np.float32)
                valid = (src.read_masks(window=window, out_shape=(4, 256, 256)) > 0).all(0) & np.any(raw != 0, axis=0)
                scaled = (raw - limits[:, 0, None, None]) / (limits[:, 1, None, None] - limits[:, 0, None, None])
                if valid.mean() < .98 or ((scaled[:3].min(0) > 1.02) & (scaled[:3].mean(0) > 1.1)).mean() > .025:
                    continue
                rgb = (np.clip(scaled[[2, 1, 0]].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                lines = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))[0]
                lengths = np.zeros(1) if lines is None else np.linalg.norm(lines[:, 0, :2] - lines[:, 0, 2:], axis=1)
                score = float(np.sort(lengths[lengths > 25])[-16:].sum())
                possibilities.append((score, x, y, distance))
            chosen = []
            for candidate in sorted(possibilities, reverse=True):
                row = {"source": scene["images"][0], "source_scene_id": sid,
                       "source_window": [candidate[1], candidate[2], SIZE, SIZE]}
                if chosen and nearest_distance(row, [entry[0] for entry in chosen]) < 250:
                    continue
                chosen.append((row, candidate))
                if len(chosen) == 8:
                    break
            canvas = Image.new("RGB", (1536, 2 * 420), "white")
            draw = ImageDraw.Draw(canvas)
            for index, (row, candidate) in enumerate(chosen):
                score, x, y, distance = candidate
                identifier = sid + f"_v5_training_candidate_{index + 1:02d}"
                raw = src.read(window=Window(x, y, SIZE, SIZE)).astype(np.float32)
                scaled = np.clip((raw - limits[:, 0, None, None]) / (limits[:, 1, None, None] - limits[:, 0, None, None]), 0, 1)
                rgb = (scaled[[2, 1, 0]].transpose(1, 2, 0) ** .65 * 255).astype(np.uint8)
                picture = destination / (identifier + ".png")
                Image.fromarray(rgb).save(picture)
                row.update(id=identifier, split="train", geographic_group=scene["geographic_group"],
                           normalization_limits=limits.tolist(), image_only_line_score=score,
                           minimum_excluded_distance_m=distance, preview=str(picture),
                           assistant_visual_reviewed=False, selection_uses_model_predictions=False)
                records.append(row)
                px, py = index % 4 * 384, index // 4 * 420
                canvas.paste(Image.fromarray(rgb).resize((384, 384)), (px, py + 36))
                draw.text((px + 4, py + 5), identifier[-24:] + f" ({x},{y})", fill="black")
            canvas.save(destination / (sid + "_TRAINING_CONTACT.png"))
            print(json.dumps({"scene": sid, "training_previews": len(chosen)}), flush=True)
    result = {"tiles": records, "reservation_sha256": reservation["reservation_sha256"],
              "exclusion_inventory_sha256": digest(INVENTORY), "display_gamma": .65,
              "raw_training_pixels_modified": False, "models_run": False,
              "labels_created": False, "purpose": "Image-only TRAINING choices; not accuracy data"}
    (destination / "CANDIDATES.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["reserve", "candidates"])
    args = parser.parse_args()
    if args.command == "reserve":
        data = reserve()
        print(json.dumps({"reserved_geometry_only": len(data["tiles"]), "audit_extracted": False}))
    else:
        training_candidates()
