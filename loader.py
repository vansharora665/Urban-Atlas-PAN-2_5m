"""Portable entry point for one frozen Urban Atlas pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def ensure_weights(folder=HERE):
    folder = Path(folder).resolve()
    config = json.loads((folder / "model_config.json").read_text())
    path = folder / "model_weights.pth"
    pointer = False
    if path.is_file():
        with path.open("rb") as f:
            pointer = f.read(100).startswith(b"version https://git-lfs.github.com/spec/v1")
        if not pointer:
            if sha256(path) != config["weights_sha256"]:
                raise ValueError("Weight checksum mismatch; refusing to load a modified checkpoint")
            return path
    url = config["weights_url"]
    if not url.startswith("https://github.com/vansharora665/Urban-Atlas-Final-Models/releases/download/"):
        raise ValueError("Unexpected weight download destination")
    print("Downloading the actual model weights (not an LFS pointer)...", file=sys.stderr)
    temporary = path.with_suffix(".downloading")
    request = urllib.request.Request(url, headers={"User-Agent": "UrbanAtlasModelLoader/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            for block in iter(lambda: response.read(8 * 1024**2), b""):
                out.write(block)
        if sha256(temporary) != config["weights_sha256"]:
            raise ValueError("Downloaded weights failed SHA256 verification")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


class UrbanAtlasModel:
    """File-based pipeline adapter, not a generic torch.nn.Module/state_dict."""

    def __init__(self, folder=HERE, device="auto"):
        self.folder = Path(folder).resolve()
        self.config = json.loads((self.folder / "model_config.json").read_text())
        if device not in {"auto", "cpu"}:
            raise ValueError("Use auto (CUDA/MPS/CPU) or cpu")
        self.device = device

    def _run(self, arguments):
        ensure_weights(self.folder)
        command = [sys.executable, str(self.folder / "worker.py"), "--device", self.device, *arguments]
        result = subprocess.run(command, cwd=self.folder, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr + "\n" + result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return json.loads(result.stdout.strip().splitlines()[-1])

    def verify(self):
        return self._run(["--verify"])

    def predict_tif(self, image, output_dir, *, roads=True, vectors=False, zones=False):
        arguments = ["--input", str(Path(image).resolve()), "--output", str(Path(output_dir).resolve())]
        if roads:
            arguments.append("--roads")
        if vectors:
            arguments.append("--vectors")
        if zones:
            arguments.append("--zones")
        return self._run(arguments)


def load_model(model_dir=None, device="auto"):
    return UrbanAtlasModel(model_dir or HERE, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--no-road-specialist", action="store_true")
    parser.add_argument("--vectors", action="store_true")
    parser.add_argument("--zones", action="store_true")
    args = parser.parse_args()
    model = load_model(device=args.device)
    if args.verify:
        result = model.verify()
    elif args.input and args.output:
        result = model.predict_tif(args.input, args.output, roads=not args.no_road_specialist,
                                   vectors=args.vectors, zones=args.zones)
    else:
        parser.error("Use --verify, or --input image.tif --output new_directory")
    print(json.dumps(result, indent=2))
