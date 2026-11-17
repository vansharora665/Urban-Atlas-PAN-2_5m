# Applying The Unified Platform Fix

Target: https://github.com/shindeanand-587/btp-

Base revision: `f401dbec06d6330aa802cc2a67af30293aa0c8d6`.
This model repository does not control the hosted Streamlit application.

## App Owner: Apply Once, Not Once Per Model

From the platform source checkout, use the included patch:

```bash
git apply --check /path/to/platform_compatibility.patch
git apply /path/to/platform_compatibility.patch
```

Review the changes, commit them and redeploy/reboot the Streamlit app. The patch
is the same in both model repositories. Do not apply it twice. If the source has
changed since the base revision, use `git apply --check` to detect the conflict
and merge deliberately; do not force the patch.

## What It Fixes

- Calls `BaseModelAdapter` with its actual constructor signature, then sets the requested device.
- Imports `sys` before launching a contributor preflight subprocess.
- Uses the declared `weights_file` and preselects that checkpoint in the registration form.
- Preserves the contributor's preprocessing when loading a TorchScript file.
- Keeps full-scene normalized inputs on CPU; only inference tiles move to the model device.
- Stops with an error on model/preprocessing failures instead of returning simulated heuristic predictions.
- Preserves missing metrics as null; no default OA, fabricated class IoUs or `OA - 28` mIoU for GitHub imports.
- Renders unreported contributed metrics without crashing the leaderboard.
- Preserves contributor architecture names and supports the existing four-band sensor choice.
- Pads/crops non-square narrow images correctly during overlapping tiled inference.

## Register Both Repositories

1. Fetch and inspect the model repository URL again to refresh stale metadata.
2. Select `model_weights.pth`, with `adapter.py` as the custom script.
3. PAN input is one band at 2.5 m; native input is four B/G/R/NIR bands at 0.6 m.
4. Leave the measured-accuracy checkbox unchecked: raw adapter metrics are not independently measured.
5. Confirm registration. The preflight should pass and inference must say a live network loaded.
6. Test a small TIFF first. A successful interface test does not establish accuracy or large-scene capacity.

Do not repeat bands, drop NIR, normalize by 255 after the custom stretch, remap
the five canonical output channels again, or treat Unknown/no-data as water.
The original live Urban Atlas platform and its model registry were not changed.

## Verification Scope And Remaining Limits

`HOST_INTEGRATION_VERIFICATION.json` records local execution of the actual public
host importer and both Python/TorchScript inference paths. Only registry writes
and the image-bundle container were replaced with in-memory stubs. Actual model
weights, preprocessing and overlapping tile inference were exercised. The UI
files were syntax-checked, not deployed or browser-tested after patching.

The hosted app itself has NOT been modified. The owner must apply and redeploy.
Existing stale model registrations should be removed/refetched so their old RGB,
GSD and placeholder metrics do not persist. This patch focuses on GitHub imports;
other manually entered benchmark claims still require independent review.

The host still holds full-scene probability arrays in RAM; this is not a promise
of memory-bounded inference for 1 GB TIFFs. Use the original `loader.py` pipeline
for streamed georeferenced output and optional shapefiles. Correct host-side
no-data/cloud handling and scientific evaluation remain separate responsibilities.
