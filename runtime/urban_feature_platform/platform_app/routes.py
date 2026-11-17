from __future__ import annotations

import uuid
import time
import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory, Response
from werkzeug.utils import secure_filename

from .config import Settings
from .image_utils import ensure_dir, is_allowed_file
from .inference import InferenceService
from .model_registry import ModelRegistry
from .prediction_jobs import PredictionJobs
from .saved_results import read_saved_result, recent_results


platform_bp = Blueprint("platform", __name__)


def get_settings() -> Settings:
    return current_app.config["PLATFORM_SETTINGS"]


def get_registry() -> ModelRegistry:
    registry = current_app.extensions.get("urban_model_registry")
    if registry is None:
        registry = ModelRegistry(get_settings())
        current_app.extensions["urban_model_registry"] = registry
    return registry


def get_inference_service() -> InferenceService:
    service = current_app.extensions.get("urban_inference_service")
    if service is None:
        service = InferenceService(get_settings(), get_registry())
        current_app.extensions["urban_inference_service"] = service
    return service


def remember_prediction_input(settings, result, upload_path):
    if not result.get("result_id"):
        return
    try:
        from .cartography import remember_input
        remember_input(settings, result["result_id"], upload_path)
    except (OSError, ValueError, KeyError, TypeError):
        # Optional provenance must not turn a completed prediction into a failure.
        logging.getLogger(__name__).warning("Input-validity reference unavailable; cartographic Unknown filling stays protected")


@platform_bp.get("/")
def index():
    settings = get_settings()
    return render_template(
        "index.html",
        models=get_registry().list_models(),
        allowed_extensions=", ".join(settings.allowed_extensions),
        max_upload_mb=round(settings.max_content_length / (1024 * 1024)),
    )


@platform_bp.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "enabled_models": len(get_registry().list_models()),
        }
    )


@platform_bp.get("/api/models")
def list_models():
    return jsonify({"models": get_registry().list_models()})


@platform_bp.post("/api/predict")
def predict():
    upload_started = time.perf_counter()
    settings = get_settings()
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Please upload an image file."}), 400

    if not is_allowed_file(uploaded.filename, settings.allowed_extensions):
        return jsonify(
            {
                "error": f"Unsupported file type. Allowed extensions: {', '.join(settings.allowed_extensions)}",
            }
        ), 400

    mode = request.form.get("model_mode", "auto").strip().lower()
    if mode not in {"auto", "manual"}:
        return jsonify({"error": "Invalid model mode."}), 400

    model_id = request.form.get("model_id", "").strip() or None
    if mode == "manual" and not model_id:
        return jsonify({"error": "Choose a model or switch back to Auto mode."}), 400

    original_name = secure_filename(uploaded.filename)
    suffix = Path(original_name).suffix.lower()
    upload_id = uuid.uuid4().hex[:8]
    upload_path = ensure_dir(settings.upload_dir) / f"{Path(original_name).stem}_{upload_id}{suffix}"
    uploaded.save(upload_path)
    upload_seconds = time.perf_counter() - upload_started

    try:
        service = get_inference_service()
        band_order = request.form.get("native_band_order", "").strip() or None
        refinement = request.form.get("native_refinement", "on")
        if refinement not in {"on", "off"}:
            raise ValueError("Invalid 0.6 m refinement mode")
        if request.form.get("background") == "1":
            jobs = current_app.extensions.get("prediction_jobs")
            if jobs is None:
                jobs = PredictionJobs()
                current_app.extensions["prediction_jobs"] = jobs
            def execute(progress):
                result = service.predict_file(upload_path, mode=mode, model_id=model_id,
                                              band_order=band_order, progress=progress,
                                              native_refinement=refinement == "on")
                remember_prediction_input(settings, result, upload_path)
                result["upload_read_and_save_seconds"] = upload_seconds
                return result
            job = jobs.submit(execute)
            job["upload_read_and_save_seconds"] = upload_seconds
            return jsonify(job), 202
        result = service.predict_file(upload_path, mode=mode, model_id=model_id, band_order=band_order,
                                      native_refinement=refinement == "on")
        remember_prediction_input(settings, result, upload_path)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


@platform_bp.post("/api/results/<result_id>/model-roads")
def model_roads(result_id):
    from .model_roads import capability, result_directory, saved_addon
    try:
        settings = get_settings()
        directory = result_directory(settings, result_id)
        cached = saved_addon(directory, result_id)
        if cached:
            return jsonify({**cached, "cached_result": True})
        available = capability(settings, result_id)
        if not available["available"]:
            return jsonify({"error": available["reason"]}), 400
        jobs = current_app.extensions.get("prediction_jobs")
        if jobs is None:
            jobs = PredictionJobs()
            current_app.extensions["prediction_jobs"] = jobs
        service = get_inference_service()
        return jsonify(jobs.submit(lambda progress: service.predict_model_roads(result_id, progress))), 202
    except (OSError, ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


@platform_bp.post("/api/results/<result_id>/model-roads-shapefile")
def model_roads_shapefile(result_id):
    from .model_roads import result_directory
    try:
        result_directory(get_settings(), result_id)
        jobs = current_app.extensions.get("prediction_jobs")
        if jobs is None:
            jobs = PredictionJobs()
            current_app.extensions["prediction_jobs"] = jobs
        service = get_inference_service()
        def execute(progress):
            progress(.05, "Polygonizing model-predicted road surfaces on request; original classes unchanged")
            return service.export_model_roads(result_id)
        return jsonify(jobs.submit(execute)), 202
    except (OSError, ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


@platform_bp.get("/api/results/recent")
def list_recent_results():
    return jsonify({"results": recent_results(get_settings().result_dir)})


@platform_bp.get("/api/results/<result_id>")
def saved_result(result_id):
    try:
        return jsonify(read_saved_result(get_settings().result_dir, result_id, get_registry()))
    except (OSError, ValueError, KeyError, TypeError):
        return jsonify({"error": "This saved result is unavailable or incomplete."}), 404


@platform_bp.get("/api/jobs/<job_id>")
def prediction_job(job_id):
    jobs = current_app.extensions.get("prediction_jobs")
    try:
        if jobs is None:
            raise KeyError("Prediction job not found; the app may have restarted.")
        return jsonify(jobs.get(job_id))
    except KeyError as error:
        return jsonify({"error": str(error)}), 404


@platform_bp.get("/api/results/<result_id>/usable-map")
def usable_map_status(result_id):
    from .cartography import saved_layer
    try:
        return jsonify({"usable_map": saved_layer(get_settings(), result_id)})
    except (OSError, ValueError, KeyError):
        return jsonify({"error": "Open a completed analysis first."}), 400


@platform_bp.post("/api/results/<result_id>/usable-map")
def create_usable_map(result_id):
    from .cartography import build, capability, saved_layer
    try:
        settings = get_settings()
        capability(settings, result_id)
        cached = saved_layer(settings, result_id)
        if cached:
            return jsonify({**cached, "cached_result": True})
        jobs = current_app.extensions.get("prediction_jobs")
        if jobs is None:
            jobs = PredictionJobs()
            current_app.extensions["prediction_jobs"] = jobs
        return jsonify(jobs.submit(lambda progress: build(settings, result_id, progress))), 202
    except (OSError, ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


@platform_bp.post("/api/results/<result_id>/usable-map-vectors")
def usable_map_vectors(result_id):
    from .cartography import export_vectors, saved_layer
    try:
        settings = get_settings()
        if not saved_layer(settings, result_id):
            raise ValueError("Build the usable-map layer first")
        jobs = current_app.extensions.get("prediction_jobs")
        if jobs is None:
            jobs = PredictionJobs()
            current_app.extensions["prediction_jobs"] = jobs
        return jsonify(jobs.submit(lambda progress: export_vectors(settings, result_id, progress))), 202
    except (OSError, ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


@platform_bp.get("/api/results/<result_id>/usable-map/tiles/<layer>/<int:z>/<int:x>/<int:y>.png")
def usable_map_tile(result_id, layer, z, x, y):
    from .cartography import tile
    try:
        return Response(tile(get_settings(), result_id, layer, z, x, y), mimetype="image/png",
                        headers={"Cache-Control": "private, max-age=300"})
    except (OSError, ValueError, KeyError):
        return jsonify({"error": "Map tile unavailable."}), 404


@platform_bp.get("/maps/<result_id>/usable")
def usable_map_viewer(result_id):
    from .cartography import saved_layer
    try:
        addon = saved_layer(get_settings(), result_id)
        if not addon:
            raise ValueError("Build the usable-map layer first")
        return render_template("usable_map.html", result_id=result_id, addon=addon)
    except (OSError, ValueError, KeyError):
        return "Build this analysis's usable-map layer in the dashboard first.", 404


@platform_bp.get('/api/results/<result_id>/zones')
def structured_zone_status(result_id):
    from .structured_zones import saved_layer
    try:
        return jsonify({'structured_zones': saved_layer(get_settings(), result_id)})
    except (OSError, ValueError, KeyError):
        return jsonify({'error': 'Open a completed native analysis first.'}), 400


@platform_bp.post('/api/results/<result_id>/zones')
def structured_zone_build(result_id):
    from .structured_zones import build, source_inputs, saved_layer
    try:
        settings, registry = get_settings(), get_registry()
        source_inputs(settings, result_id)
        cached = saved_layer(settings, result_id)
        if cached:
            return jsonify(cached)
        jobs = current_app.extensions.setdefault('prediction_jobs', PredictionJobs())
        return jsonify(jobs.submit(lambda progress: build(settings, registry, result_id, progress))), 202
    except (OSError, ValueError, KeyError) as error:
        return jsonify({'error': str(error)}), 400


@platform_bp.post('/api/results/<result_id>/zones-vectors')
def structured_zone_vectors(result_id):
    from .structured_zones import saved_layer
    from .cartography import export_vectors
    try:
        settings = get_settings(); addon = saved_layer(settings, result_id)
        if not addon:
            raise ValueError('Build neighbourhood zones first')
        jobs = current_app.extensions.setdefault('prediction_jobs', PredictionJobs())
        return jsonify(jobs.submit(lambda progress: export_vectors(settings, result_id, progress, addon=addon))), 202
    except (OSError, ValueError, KeyError) as error:
        return jsonify({'error': str(error)}), 400


@platform_bp.get('/api/results/<result_id>/zones/tiles/<layer>/<int:z>/<int:x>/<int:y>.png')
def structured_zone_tile(result_id, layer, z, x, y):
    from .structured_zones import tile
    try:
        return Response(tile(get_settings(), result_id, layer, z, x, y), mimetype='image/png',
                        headers={'Cache-Control': 'private, max-age=300'})
    except (OSError, ValueError, KeyError):
        return jsonify({'error': 'Neighbourhood tile unavailable'}), 404


@platform_bp.get('/maps/<result_id>/zones')
def structured_zone_viewer(result_id):
    from .structured_zones import saved_layer
    try:
        addon = saved_layer(get_settings(), result_id)
        if not addon:
            return 'Build this native analysis neighbourhood model first.', 404
        return render_template('usable_map.html', result_id=result_id, addon=addon)
    except (OSError, ValueError, KeyError):
        return 'Neighbourhood map unavailable.', 404


@platform_bp.post("/api/results/<result_id>/map-assisted-roads")
def map_assisted_roads(result_id):
    if (request.get_json(silent=True) or {}).get("external_map_consent") is not True:
        return jsonify({"error": "Allow sharing geographic bounds with public OSM providers first. TIFF pixels stay local."}), 400
    from .map_assisted_roads import build_network, result_directory
    try:
        settings = get_settings()
        result_directory(settings, result_id)
        jobs = current_app.extensions.get("prediction_jobs")
        if jobs is None:
            jobs = PredictionJobs()
            current_app.extensions["prediction_jobs"] = jobs
        job = jobs.submit(lambda progress: build_network(settings, result_id, progress))
        return jsonify(job), 202
    except (OSError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@platform_bp.post("/api/results/<result_id>/map-assisted-shapefile")
def map_assisted_shapefile(result_id):
    from .map_assisted_roads import export_network_shapefile
    try:
        return jsonify(export_network_shapefile(get_settings(), result_id))
    except (OSError, ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        current_app.logger.exception("Map-assisted network shapefile export failed")
        return jsonify({"error": "Map-network export failed; your model-only outputs are unchanged."}), 500


@platform_bp.post("/api/results/<result_id>/shapefiles")
def generate_result_shapefiles(result_id: str):
    try:
        result = get_inference_service().generate_shapefiles_for_result(result_id)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception("Shapefile generation failed")
        return jsonify({"error": f"Shapefile generation failed: {exc}"}), 500

    return jsonify(result)


@platform_bp.get("/results/<result_id>/<path:filename>")
def serve_result_file(result_id: str, filename: str):
    result_dir = get_settings().result_dir / result_id
    return send_from_directory(result_dir, filename, as_attachment=False)
