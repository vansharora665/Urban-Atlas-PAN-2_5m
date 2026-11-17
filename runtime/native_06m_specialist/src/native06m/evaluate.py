from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from .data import load_manifest, training_preflight
from .losses import confusion_metrics
from .model import NativeAttentionUNetPP, choose_device
from .predict import native_stream_prediction


def main():
    parser = argparse.ArgumentParser(description="Evaluate the selected checkpoint on whole held-out test scenes")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,default=Path("config/scenes.json"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if not checkpoint.get("training_complete"):
        raise ValueError("A completed training checkpoint is required")
    manifest = load_manifest(args.manifest)
    config = checkpoint["config"]
    report = training_preflight(manifest,config)
    if not report["ready_for_training"]:
        raise ValueError("\n".join(report["errors"]))
    setup = json.loads((args.checkpoint.parent/"training_setup.json").read_text())
    if setup["manifest"] != manifest:
        raise ValueError("Evaluation manifest differs from the frozen training split; do not silently redefine the held-out test")
    args.output.mkdir(parents=True,exist_ok=False)
    model = NativeAttentionUNetPP(config["encoder_name"],config.get("in_channels",1))
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    device = choose_device();model.to(device)
    rows,groups,pooled = [],defaultdict(list),np.zeros((5,5),np.int64)
    for scene in manifest["scenes"]:
        if scene["split"]!="test":
            continue
        for variant,path in enumerate(scene["images"]):
            destination = args.output/f"{scene['scene_id']}_variant{variant+1}_labels.tif"
            native_stream_prediction(model,path,destination,device,config["patch_size"],
                                     config["inference_stride"],config["batch_size"],scene.get("quality_mask"))
            matrix = np.zeros((5,5),np.int64)
            with rasterio.open(scene["labels"]) as truth,rasterio.open(destination) as prediction:
                for y in range(0,truth.height,256):
                    window = Window(0,y,truth.width,min(256,truth.height-y))
                    target,predicted = truth.read(1,window=window),prediction.read(1,window=window)
                    if not np.isin(target,[0,1,2,3,4,255]).all():
                        raise ValueError("Held-out labels contain unsupported class IDs")
                    valid = (target<5)&(predicted<5)
                    matrix += np.bincount(target[valid].astype(int)*5+predicted[valid],minlength=25).reshape(5,5)
            metrics = confusion_metrics(matrix)
            metrics["evaluated_fraction_of_grid"] = metrics["evaluated_pixels"]/(truth.width*truth.height)
            rows.append({"scene_id":scene["scene_id"],"variant":variant+1,"geographic_group":scene["geographic_group"],**metrics})
            groups[scene["geographic_group"]].append(metrics["mean_iou"])
            pooled += matrix
    group_means = {key:float(np.mean([v for v in values if v is not None])) for key,values in groups.items()
                   if any(v is not None for v in values)}
    result = {"scope":"Whole held-out geographic test scenes; labels declared human-reviewed",
              "source_variants_are_not_independent_geographies":True,
              "per_variant":rows,"pooled_pixels":confusion_metrics(pooled),
              "mean_iou_per_geographic_group":group_means,
              "macro_mean_geographic_group_iou":float(np.mean(list(group_means.values()))) if group_means else None}
    (args.output/"held_out_test_metrics.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
