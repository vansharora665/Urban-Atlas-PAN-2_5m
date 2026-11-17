"""Verify a native-resolution forward/backward pass; never save a trained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import rasterio
import torch

from .data import file_digest, load_manifest, normalization, read_json, read_native, resolve
from .losses import segmentation_loss
from .model import NativeAttentionUNetPP, choose_device


def run_smoke(manifest_path,config_path,output):
    config,base=read_json(config_path)
    manifest=load_manifest(manifest_path)
    source_path=manifest["scenes"][0]["images"][0]
    checkpoint_path=resolve(base,config["encoder_initialization_checkpoint"])
    digest=file_digest(checkpoint_path)
    model=NativeAttentionUNetPP(config["encoder_name"])
    initialization=model.initialize_encoder(checkpoint_path)
    device=choose_device();model.to(device);model.train()
    size=config["patch_size"]
    with rasterio.open(source_path) as source:
        image,_=read_native(source,(source.width-size)//2,(source.height-size)//2,size,normalization(source_path))
    tensor=torch.from_numpy(image[None,None]).float().to(device)
    # Synthetic stripes only check autograd plumbing; these are never saved as labels.
    target=(torch.arange(size,device=device)[None,None,:].expand(1,size,size)*5//size).long()
    target[:,:8]=255
    started=time.monotonic()
    result=model(tensor)
    loss=segmentation_loss(result,target,torch.ones(5,device=device),config)
    if not torch.isfinite(loss):
        raise RuntimeError("Native forward/backward smoke has non-finite loss")
    loss.backward()
    if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
        raise RuntimeError("Non-finite gradients")
    if device.type=="mps":
        torch.mps.synchronize()
    report={"status":"forward_backward_smoke_only_not_finetuned","device":str(device),
            "source":source_path,"input_shape":list(tensor.shape),"native_gsd_m":.6,
            "output_shapes":{key:list(value.shape) for key,value in result.items()},
            "model_parameters":sum(p.numel() for p in model.parameters()),
            "encoder_initialization":initialization,"elapsed_seconds":time.monotonic()-started,
            "target":"Synthetic stripes solely for autograd verification; not real training ground truth",
            "optimizer_steps":0,"model_checkpoint_saved":False,"accuracy_measured":False,
            "initialization_checkpoint_unchanged":file_digest(checkpoint_path)==digest}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=Path("config/scenes.json"))
    parser.add_argument("--config",type=Path,default=Path("config/training.json"))
    parser.add_argument("--output",type=Path,default=Path("audit/native_forward_backward_smoke.json"))
    args=parser.parse_args()
    print(json.dumps(run_smoke(args.manifest,args.config,args.output),indent=2))


if __name__=="__main__":
    main()
