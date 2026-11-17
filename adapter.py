"""Unified platform entry point. No pretrained downloads or silent fallback models."""

import importlib.util
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent


def _local_module(name, filename):
    # The host loads both repos under the same adapter name. Isolate dependencies.
    key = "urban_atlas_" + HERE.name.replace("-", "_").replace(".", "_") + "_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(key, None)
            raise
    return sys.modules[key]


_support = _local_module("support", "contract_runtime.py")
CLASS_NAMES = _support.CLASS_NAMES
CLASS_MAPPING = dict(enumerate(CLASS_NAMES))
SOURCE_TO_PLATFORM = _support.SOURCE_TO_PLATFORM


def get_model(weights_path=None):
    """Strictly restore trained weights and return five canonical logit channels."""
    path = Path(weights_path) if weights_path is not None else HERE / "model_weights.pth"
    if path.suffix in (".pt", ".ts"):
        return torch.jit.load(str(path), map_location="cpu").eval()
    try:
        architecture = _local_module("architecture", "deployment_model.py")
    except ModuleNotFoundError as error:
        if error.name != "segmentation_models_pytorch":
            raise
        # The host has torch but not SMP. Use the exact verified graph, never
        # heuristic output or an untrained replacement network.
        cfg = _support.config_at(HERE)
        if _support.sha256(path) != cfg["weights_sha256"]:
            raise ValueError("Install requirements-adapter.txt for custom state-dict loading") from error
        script = HERE / "model.pt"
        if _support.sha256(script) != cfg["torchscript_sha256"]:
            raise ValueError("Standalone graph checksum mismatch")
        return torch.jit.load(str(script), map_location="cpu").eval()
    return _support.load_weights(architecture.DeploymentModel(), path)


def preprocess(raw_np):
    """Normalize the full input scene ONCE before the host extracts overlapping tiles."""
    config = _support.config_at(HERE)
    return _support.preprocess_array(raw_np, config["input_channels"], (2, 98))


def preprocess_with_limits(raw_np, limits, valid_mask=None, nodata=0):
    """Streaming API: reuse scene-wide (low, high) limits for every input tile."""
    config = _support.config_at(HERE)
    return _support.preprocess_array(raw_np, config["input_channels"], (2, 98),
                                     limits=limits, valid_mask=valid_mask, nodata=nodata)
