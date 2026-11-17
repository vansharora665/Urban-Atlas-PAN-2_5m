"""CPU contract verification, not an accuracy or training/inference parity claim."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also check TorchScript, prefixes and edge cases")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("preflight_adapter", HERE / "adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    cfg = adapter._support.config_at(HERE)
    expected_names = ("Water", "Vegetation", "Barren Land", "Built-up", "Roads")
    assert adapter.CLASS_NAMES == expected_names
    assert cfg["classes"] == {str(i): name for i, name in enumerate(expected_names)}
    assert tuple(cfg["source_to_platform_channels"]) == (0, 4, 3, 2, 1)
    rng = np.random.default_rng(23)
    raw = rng.integers(1, 2048, (cfg["input_channels"], 512, 512), dtype=np.uint16)
    tensor = adapter.preprocess(raw)
    assert tensor.shape == (1, cfg["input_channels"], 512, 512)
    assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    assert 0 <= tensor.min() and tensor.max() <= 1
    model = adapter.get_model()
    assert isinstance(model, torch.nn.Module) and not model.training
    with torch.inference_mode():
        output = model(tensor)
        assert output.shape == (1, 5, 512, 512) and torch.isfinite(output).all()
        if hasattr(model, "source_logits"):
            source = model.source_logits(tensor)
            torch.testing.assert_close(output, source[:, [0, 4, 3, 2, 1]], rtol=0, atol=0)
        else:
            assert adapter._support.sha256(HERE / "model.pt") == cfg["torchscript_sha256"]
        assert output.std() > 0, "Output must not be an all-zero or constant placeholder"
    checks = ["strict weights or checksum-pinned TorchScript restore", "512x512 finite five-class logits",
              "canonical class mapping", "correct input bands and percentile normalization"]
    if args.full:
        assert adapter._support.sha256(HERE / "model_weights.pth") == cfg["weights_sha256"]
        assert adapter._support.sha256(HERE / "model.pt") == cfg["torchscript_sha256"]
        with torch.inference_mode():
            scripted = torch.jit.load(str(HERE / "model.pt"), map_location="cpu").eval()
            torch.testing.assert_close(scripted(tensor), output, atol=3e-5, rtol=2e-4)
            second = model(1 - tensor)
            assert not torch.allclose(second, output), "Model must respond to different imagery"
        if hasattr(model, "source_logits"):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "prefixed.pth"
                torch.save({"state_dict": {"module." + key: value for key, value in model.state_dict().items()}}, path)
                prefixed = adapter.get_model(path)
                with torch.inference_mode():
                    torch.testing.assert_close(prefixed(tensor), output, rtol=0, atol=0)
        with torch.inference_mode():
            narrow = torch.rand(1, cfg["input_channels"], 301, 479)
            eager = model(narrow)
            assert eager.shape == (1, 5, 301, 479)
            torch.testing.assert_close(scripted(narrow), eager, atol=3e-5, rtol=2e-4)
        zeros = adapter.preprocess(np.zeros_like(raw))
        assert torch.isfinite(zeros).all() and not zeros.any()
        nan = adapter.preprocess(np.full(raw.shape, np.nan, dtype=np.float32))
        assert torch.isfinite(nan).all() and not nan.any()
        hwc = adapter.preprocess(raw.transpose(1, 2, 0))
        torch.testing.assert_close(hwc, tensor, rtol=0, atol=0)
        try:
            adapter.preprocess(np.zeros((3, 512, 512), dtype=np.float32))
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong channel count must be rejected")
        assert (HERE / "model_weights.pth").stat().st_size < 100_000_000
        assert (HERE / "model.pt").stat().st_size < 100_000_000
        checks.extend(["standalone TorchScript numeric parity", "module-prefix cleanup when SMP is installed", "arbitrary-edge tile padding",
                       "input sensitivity", "CHW/HWC parity", "no-data and nonfinite safety",
                       "wrong-band rejection", "both artifacts below 100 MB"])
    result = {"status": "PASS", "output_shape": list(output.shape), "checks": checks,
              "seconds": round(time.perf_counter() - started, 2), "accuracy_test": False,
              "scope": "Interface and numerical export checks only; not accuracy or full pipeline equivalence"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
