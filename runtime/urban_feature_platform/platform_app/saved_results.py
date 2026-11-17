"""Read completed local predictions without loading their model checkpoints."""

import json
from pathlib import Path
import re
from types import SimpleNamespace

RESULT_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
ARTIFACTS = {
    "input_preview_url": "input_preview.png", "marked_preview_url": "marked_preview.png",
    "segmentation_preview_url": "segmentation_preview.png", "geotiff_url": "segmentation_labels_georef.tif",
    "base_geotiff_url": "segmentation_labels_base_georef.tif",
    "base_segmentation_preview_url": "base_segmentation_preview.png",
    "cloud_quality_geotiff_url": "cloud_quality_georef.tif",
    "cloud_screening_preview_url": "cloud_screening_preview.png",
    "class_shapefiles_zip_url": "class_shapefiles_georef.zip",
    "summary_json_url": "prediction_summary.json", "class_percentages_url": "class_percentages.json",
    "summary_text_url": "RESULTS_SIMPLE.txt",
}


def analysis_info(directory):
    """Optional display labels, kept separate from immutable prediction evidence."""
    path = Path(directory) / "saved_analysis_info.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 4096:
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            return {}
        return {key: value[key].strip()[:240 if key == "description" else 120]
                for key in ("title", "description") if isinstance(value.get(key), str) and value[key].strip()}
    except (OSError, ValueError):
        return {}


def read_saved_result(root, result_id, registry):
    if not RESULT_ID.fullmatch(result_id):
        raise ValueError("Invalid result identifier")
    directory = Path(root) / result_id
    if directory.is_symlink():
        raise ValueError("Linked result directories are not supported")
    summary = json.loads((directory / "prediction_summary.json").read_text())
    if not isinstance(summary, dict):
        raise ValueError("Saved prediction is incomplete")
    selected = summary["selected_model"]
    if not isinstance(selected, dict) or not isinstance(summary.get("class_percentages"), list):
        raise ValueError("Saved prediction is incomplete")
    try:
        record = registry.get(selected["id"])
        colors, names = record.display_class_colors, record.class_names
    except KeyError:
        colors = {0: (31, 119, 180), 1: (245, 245, 245), 2: (255, 215, 0),
                  3: (140, 86, 75), 4: (44, 160, 44)}
        names = {int(row["class_id"]): row["class_name"] for row in summary["class_percentages"]}
    legend = [{"class_id": index, "class_name": name, "color_rgb": colors.get(index, (180, 180, 180))}
              for index, name in names.items()]
    legend.append({"class_id": 255, "class_name": "Unknown / no-data", "color_rgb": [220, 220, 220]})
    scores = summary.get("score_breakdown") or {"final_score": None, "confidence_mean": None, "domain_similarity": None}
    map_assisted = None
    addon = directory / "map_assisted_summary.json"
    if addon.is_file() and not addon.is_symlink():
        try:
            value = json.loads(addon.read_text())
            if isinstance(value, dict) and value.get("mode") == "map_assisted_main_roads_v1":
                map_assisted = value
        except (OSError, ValueError):
            pass
    from .model_roads import saved_addon
    model_roads = saved_addon(directory, result_id)
    from .cartography import saved_layer
    usable_map = saved_layer(SimpleNamespace(result_dir=Path(root)), result_id)
    from .structured_zones import saved_layer as saved_zones
    structured_zones = saved_zones(SimpleNamespace(result_dir=Path(root)), result_id)
    return {**summary, "result_id": result_id, "saved_result": True, "legend": legend,
            "saved_analysis_info": analysis_info(directory),
            "model_roads": model_roads,
            "usable_map": usable_map,
            "structured_zones": structured_zones,
            "map_assisted_roads": map_assisted,
            "score_breakdown": scores, "cloud_screening": summary.get("cloud_screening"),
            "candidate_rankings": summary.get("candidate_rankings") or [{
                **selected, "score": scores.get("final_score"), "dominant_class": summary.get("dominant_class", "Unknown"),
                "score_breakdown": scores, "secondary_metric_value": selected.get("validation_macro_f1")}],
            "note": summary.get("note") or summary.get("generalization_warning") or "Saved local prediction. Coverage percentages are not accuracy.",
            "artifacts": {key: f"/results/{result_id}/{filename}" if (directory / filename).is_file() else None
                          for key, filename in ARTIFACTS.items()}}


def recent_results(root, limit=8):
    root = Path(root)
    if not root.exists():
        return []
    rows = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name, reverse=True):
        if not directory.is_dir() or directory.is_symlink() or not RESULT_ID.fullmatch(directory.name):
            continue
        try:
            summary = json.loads((directory / "prediction_summary.json").read_text())
            if not isinstance(summary, dict) or not all((directory / filename).is_file() for filename in
                    ("input_preview.png", "marked_preview.png", "segmentation_preview.png")):
                continue
            if summary.get("deployment_status") in {"candidate_verification", "rejected_candidate"}:
                continue
            metadata, model = summary.get("image", {}), summary["selected_model"]
            if not isinstance(metadata, dict) or not isinstance(model, dict):
                continue
            info = analysis_info(directory)
            rows.append({"result_id": directory.name, "filename": metadata.get("filename", directory.name),
                         "title": info.get("title"), "description": info.get("description"),
                         "model": model.get("display_name", model.get("id", "Saved model")),
                         "width": metadata.get("width"), "height": metadata.get("height"),
                         "dominant_class": summary.get("dominant_class", "Unknown"),
                         "preview_url": f"/results/{directory.name}/segmentation_preview.png"})
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(rows) >= limit:
            break
    return rows
