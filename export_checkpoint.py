"""Export tensor-only weights and traced TorchScript from the verified source bundle."""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time

import torch

HERE = Path(__file__).resolve().parent


def load_adapter():
    spec = importlib.util.spec_from_file_location("export_adapter", HERE / "adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path,
                        help="Optional trusted local source checkpoint, never untrusted pickle")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    adapter = load_adapter()
    helper = adapter._support
    config = helper.config_at(HERE)
    legacy = json.loads((HERE / "pipeline_config.json").read_text())
    if args.source_checkpoint:
        # Pickle is used only for an explicitly supplied trusted training artifact.
        checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
        source_hash = helper.sha256(args.source_checkpoint)
    else:
        bundle_path = HERE / "pipeline_weights.bundle"
        if helper.sha256(bundle_path) != legacy["weights_sha256"]:
            raise ValueError("Original pipeline bundle checksum mismatch")
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
        component = next(row for row in bundle["components"]
                         if row["path"] == config["source_component"])
        raw = component["data"].numpy().tobytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash != component["sha256"]:
            raise ValueError("Embedded source checkpoint checksum mismatch")
        # Own checksum-pinned training checkpoint, not a downloaded arbitrary pickle.
        checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    state = helper.clean_state(checkpoint)
    if config["family"] == "pan_2_5m":
        state = {("network." + key if not key.startswith("network.") else key): value
                 for key, value in state.items()}
    model = adapter._local_module("architecture", "deployment_model.py").DeploymentModel()
    model.load_state_dict(state, strict=True)
    model.eval()
    dummy = torch.rand(1, config["input_channels"], 512, 512)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="urban-export-") as temp:
        temp = Path(temp)
        torch.save(model.state_dict(), temp / "weights.pth")
        helper.compress_torch_zip(temp / "weights.pth", temp / "packed.pth", level=1)
        # Confirm lossless tensor content before replacing any exported artifact.
        restored = torch.load(temp / "packed.pth", map_location="cpu", weights_only=True)
        assert all(torch.equal(state[key], restored[key]) for key in state)
        if (temp / "packed.pth").stat().st_size >= 100_000_000:
            raise ValueError("Tensor-only FP32 checkpoint is still >=100 MB; use LFS, not silent quantization")
        with torch.inference_mode():
            expected = model(dummy)
            traced = torch.jit.trace(model, dummy, check_trace=True)
            traced = torch.jit.freeze(traced.eval())
            torch.jit.save(traced, str(temp / "model.pt"))
            helper.compress_torch_zip(temp / "model.pt", temp / "packed.pt")
            standalone = torch.jit.load(str(temp / "packed.pt"), map_location="cpu").eval()
            actual = standalone(dummy)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-4)
            assert actual.shape == (1, 5, 512, 512)
        if (temp / "packed.pt").stat().st_size >= 100_000_000:
            raise ValueError("TorchScript is >=100 MB; no precision has been changed")
        import shutil
        shutil.copyfile(temp / "packed.pth", HERE / "model_weights.pth")
        shutil.copyfile(temp / "packed.pt", HERE / "model.pt")
    config["weights_sha256"] = helper.sha256(HERE / "model_weights.pth")
    config["weights_bytes"] = (HERE / "model_weights.pth").stat().st_size
    config["torchscript_sha256"] = helper.sha256(HERE / "model.pt")
    config["torchscript_bytes"] = (HERE / "model.pt").stat().st_size
    config["source_checkpoint_sha256"] = source_hash
    (HERE / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {"status": "PASS", "source_sha256": source_hash,
              "tensor_count": len(state), "storage_dtype": "original FP32; lossless ZIP DEFLATE",
              "optimizer_scaler_scheduler_included": False,
              "output_shape": list(actual.shape),
              "trace_max_abs_error": float((actual - expected).abs().max()),
              "weights_bytes": config["weights_bytes"], "torchscript_bytes": config["torchscript_bytes"],
              "export_seconds": round(time.perf_counter() - started, 2),
              "accuracy_test": False}
    (HERE / "EXPORT_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
