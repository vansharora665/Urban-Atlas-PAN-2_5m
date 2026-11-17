"""Shared implementation copied into each standalone repository at export time."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

CLASS_NAMES = ("Water", "Vegetation", "Barren Land", "Built-up", "Roads")
SOURCE_TO_PLATFORM = (0, 4, 3, 2, 1)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_state(checkpoint):
    for name in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(name), dict):
            checkpoint = checkpoint[name]
            break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Expected a nonempty tensor state dictionary")
    result = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("State dictionary contains non-tensor data")
        while key.startswith("module."):
            key = key[7:]
        if key in result:
            raise ValueError("Duplicate state key after module-prefix removal")
        result[key] = value.detach().cpu()
    return result


def load_weights(model, path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing checkpoint: %s" % path)
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("This is a Git LFS pointer, not weights. Run git lfs pull.")
    state = clean_state(torch.load(path, map_location="cpu", weights_only=True))
    model.load_state_dict(state, strict=True)
    return model.eval()


def preprocess_array(raw_np, channels, percentiles, limits=None, valid_mask=None, nodata=0):
    """Scene-first percentile stretch; no invented bands or per-window contrast changes."""
    raw = np.asarray(raw_np)
    if raw.ndim == 2 and channels == 1:
        raw = raw[None]
    elif raw.ndim == 3:
        if raw.shape[0] == channels and raw.shape[-1] == channels:
            raise ValueError("Ambiguous channel axis; provide an unambiguous CHW array")
        if raw.shape[0] != channels and raw.shape[-1] == channels:
            raw = raw.transpose(2, 0, 1)
    if raw.ndim != 3 or raw.shape[0] != channels:
        raise ValueError("Expected exactly %d bands in CHW or HWC order" % channels)
    raw = raw.astype(np.float32, copy=False)
    valid = np.isfinite(raw).all(axis=0)
    if nodata is not None:
        valid &= ~np.all(raw == nodata, axis=0)
    if valid_mask is not None:
        if np.asarray(valid_mask).shape != raw.shape[1:]:
            raise ValueError("valid_mask must match the image spatial shape")
        valid &= np.asarray(valid_mask, dtype=bool)
    if not valid.any():
        return torch.zeros((1, channels, *raw.shape[1:]), dtype=torch.float32)
    if limits is not None:
        limits = np.asarray(limits, dtype=np.float32)
        if limits.shape != (channels, 2) or not np.isfinite(limits).all():
            raise ValueError("limits must be a finite (channels, 2) array")
        if np.any(limits[:, 1] <= limits[:, 0]):
            raise ValueError("Every upper normalization limit must exceed its lower limit")
    result = np.zeros_like(raw)
    for band in range(channels):
        if limits is None:
            values = raw[band][valid]
            if channels == 1:
                q1, q2, q50, q98, q99, q995 = np.percentile(values, [1, 2, 50, 98, 99, 99.5])
                core = max(float(q98 - q50), 1e-6)
                tail = (q99 - q98) / core > .5 and (q995 - q99) / core > .2
                low, high = (q2, q98) if tail else (q1, q99)
            else:
                low, high = np.percentile(values, percentiles)
            if high <= low:
                high = low + 1.
        else:
            low, high = limits[band]
        result[band][valid] = np.clip((raw[band][valid] - low) / (high - low), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(result))[None]


def compress_torch_zip(source, destination, level=9):
    """Lossless DEFLATE of PyTorch's ZIP records; tensor dtypes stay unchanged."""
    with zipfile.ZipFile(source, "r") as original, zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level,
            allowZip64=True) as packed:
        for info in original.infolist():
            packed.writestr(info.filename, original.read(info.filename))


def config_at(folder):
    return json.loads((Path(folder) / "model_config.json").read_text())
