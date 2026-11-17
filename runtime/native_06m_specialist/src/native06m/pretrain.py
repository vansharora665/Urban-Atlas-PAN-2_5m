"""Real label-free encoder adaptation; never reports land-cover accuracy."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .data import file_digest, inspect_manifest, load_manifest, normalization, read_json, read_native, resolve
from .model import NativeAttentionUNetPP, choose_device


def build_cache(manifest, config, destination):
    destination = Path(destination)
    signature={"manifest":manifest,"patch_size":config["patch_size"],"train_tiles":config["train_tiles"],
               "validation_tiles":config["validation_tiles"],"seed":config["seed"],
               "minimum_valid_fraction":config["minimum_valid_fraction"]}
    index_path=destination/"cache_index.json"
    if index_path.exists():
        old=json.loads(index_path.read_text())
        if old["signature"]!=signature:
            raise ValueError("Cache contract changed; choose a new cache_directory")
        if not all((destination/r["cache_file"]).is_file() for r in old["tiles"]):
            raise ValueError("Incomplete cache; rebuild in a new cache_directory")
        return old
    destination.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(config["seed"])
    size=config["patch_size"]
    records=[]
    for scene in manifest["scenes"]:
        if scene["split"] not in ("train","validation"):
            continue  # Do not even read held-out test pixels for pretraining.
        count=config["train_tiles" if scene["split"]=="train" else "validation_tiles"]
        path=scene["images"][0]
        limits=normalization(path,scene.get("quality_mask"))
        with rasterio.open(path) as src,rasterio.open(scene["quality_mask"]) as quality:
            candidates=[(x,y) for y in range(0,src.height-size+1,size)
                        for x in range(0,src.width-size+1,size)]
            rng.shuffle(candidates)
            found=0
            for x,y in candidates:
                image,valid=read_native(src,x,y,size,limits)
                valid &= quality.read(1,window=Window(x,y,size,size))==0
                if valid.mean()<config["minimum_valid_fraction"] or image[valid].std()<.015:
                    continue
                image[~valid]=.5
                name=f"{scene['scene_id']}_{x}_{y}.npz"
                np.savez_compressed(destination/name,image=image.astype(np.float16),valid=valid)
                records.append({"scene_id":scene["scene_id"],"geographic_group":scene["geographic_group"],
                    "split":scene["split"],"cache_file":name,"source":path,
                    "quality_mask":scene["quality_mask"],"window":[x,y,size,size],
                    "normalization_limits":np.asarray(limits).tolist()})
                found+=1
                if found>=count:
                    break
        if found<count:
            raise ValueError(f"Insufficient readable valid native tiles for {scene['scene_id']}: {found}/{count}")
    result={"signature":signature,"tiles":records,"test_pixels_used":False,
            "labels_generated":False,"normalization":"Per-band scene q02/q98 on valid footprint; not calibrated reflectance"}
    index_path.write_text(json.dumps(result,indent=2)+"\n")
    return result


class CachedSpectralTiles(Dataset):
    def __init__(self, root, records, augment=False):
        self.root,self.records,self.augment=Path(root),records,augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with np.load(self.root/self.records[index]["cache_file"]) as saved:
            image,valid=saved["image"].astype(np.float32),saved["valid"].copy()
        if self.augment:
            k=int(np.random.randint(4));image,valid=np.rot90(image,k),np.rot90(valid,k)
            if np.random.rand()<.5:
                image,valid=np.fliplr(image),np.fliplr(valid)
            if np.random.rand()<.5:
                image,valid=np.flipud(image),np.flipud(valid)
            # Common radiometric change preserves the channel order; no spatial resizing.
            image=np.clip(image*np.random.uniform(.9,1.1)+np.random.uniform(-.025,.025),0,1)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))),torch.from_numpy(np.ascontiguousarray(valid))


class MaskedSpectralEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder=encoder
        self.project=nn.ModuleList(nn.Conv2d(channels,24,1) for channels in encoder.out_channels[1:])
        self.reconstruct=nn.Sequential(nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,4,1))

    def forward(self, image):
        features=self.encoder(image)[1:]
        target_size=(image.shape[-2]//4,image.shape[-1]//4)
        decoded=sum(F.interpolate(layer(feature),size=target_size,mode="bilinear",align_corners=False)
                    for layer,feature in zip(self.project,features))/len(features)
        return F.interpolate(self.reconstruct(decoded),size=image.shape[-2:],mode="bilinear",align_corners=False)


def masked_input(image, fraction, block, generator=None):
    if not 0<fraction<1 or image.shape[-2]%block or image.shape[-1]%block:
        raise ValueError("Mask fraction must be in (0,1) and block must divide the patch")
    # A shared mask hides all four channels, preventing cross-band copying of hidden pixels.
    coarse=torch.rand((len(image),1,image.shape[-2]//block,image.shape[-1]//block),generator=generator)
    mask=F.interpolate((coarse<fraction).float(),size=image.shape[-2:],mode="nearest").to(image.device).bool()
    return torch.where(mask,torch.tensor(.5,device=image.device),image),mask


def reconstruction_loss(prediction, target, masked, valid):
    eligible=masked[:,0]&valid
    denominator=eligible.sum().clamp_min(1)*target.shape[1]
    loss=((prediction-target).abs()*eligible[:,None]).sum()/denominator
    return loss


def run_pretraining(manifest_path, config_path):
    config,base=read_json(config_path)
    manifest=load_manifest(manifest_path)
    report=inspect_manifest(manifest)
    if report["errors"]:
        raise ValueError("\n".join(report["errors"]))
    if manifest.get("input_channels")!=4 or config.get("in_channels")!=4:
        raise ValueError("This pretraining experiment requires explicit four-band inputs")
    if not any(s["split"]=="test" for s in manifest["scenes"]):
        raise ValueError("Reserve at least one independent source area for future testing")
    seed=config["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    cache_path=resolve(base,config["cache_directory"])
    cache=build_cache(manifest,config,cache_path)
    train_records=[r for r in cache["tiles"] if r["split"]=="train"]
    val_records=[r for r in cache["tiles"] if r["split"]=="validation"]
    if not train_records or not val_records:
        raise ValueError("Independent training and reconstruction-validation areas are required")
    train=DataLoader(CachedSpectralTiles(cache_path,train_records,True),batch_size=config["batch_size"],shuffle=True)
    validation=DataLoader(CachedSpectralTiles(cache_path,val_records),batch_size=config["batch_size"])
    candidate=NativeAttentionUNetPP(config["encoder_name"],4)
    initialization_path=resolve(base,config["encoder_initialization_checkpoint"])
    initialization=candidate.initialize_encoder(initialization_path)
    initialization["sha256_before"]=file_digest(initialization_path)
    encoder=candidate.network.encoder
    initial_conv=encoder.conv1.weight.detach().cpu().clone()
    del candidate
    device=choose_device()
    model=MaskedSpectralEncoder(encoder).to(device)
    encoder_ids={id(p) for p in model.encoder.parameters()}
    optimizer=torch.optim.AdamW([
        {"params":model.encoder.parameters(),"lr":config["encoder_lr"]},
        {"params":[p for p in model.parameters() if id(p) not in encoder_ids],"lr":config["decoder_lr"]}
    ],weight_decay=config["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["epochs"])
    run_dir=resolve(base,config["runs_directory"])/(config["model_name"]+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True,exist_ok=False)
    (run_dir/"pretraining_setup.json").write_text(json.dumps({"config":config,"manifest":manifest,
        "initialization":initialization,"cache_index":str(cache_path/"cache_index.json"),
        "train_tiles":len(train_records),"validation_tiles":len(val_records),"device":str(device),
        "semantic_labels_used":False,"test_pixels_used":False,
        "method":"CNN masked reconstruction, inspired by masked-image pretraining; NOT SatMAE or semantic training"},indent=2))
    history=[];best=float("inf");bad=0;steps=0;started=time.monotonic()

    def validate():
        model.eval();total=0.;count=0
        generator=torch.Generator().manual_seed(seed+1100)
        with torch.inference_mode():
            for image,valid in validation:
                image,valid=image.to(device),valid.to(device)
                hidden,mask=masked_input(image,config["mask_fraction"],config["mask_block_pixels"],generator)
                loss=reconstruction_loss(model(hidden),image,mask,valid)
                total+=float(loss.cpu())*len(image);count+=len(image)
        return total/max(1,count)

    baseline=validate()
    print(json.dumps({"stage":"pretraining_started","run":str(run_dir),"device":str(device),
        "baseline_masked_l1":baseline,"train_tiles":len(train_records),"validation_tiles":len(val_records)}),flush=True)
    for epoch in range(config["epochs"]):
        model.train();total=0.;count=0
        for image,valid in train:
            image,valid=image.to(device),valid.to(device)
            hidden,mask=masked_input(image,config["mask_fraction"],config["mask_block_pixels"])
            optimizer.zero_grad(set_to_none=True)
            loss=reconstruction_loss(model(hidden),image,mask,valid)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite self-supervised loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();steps+=1
            total+=float(loss.detach().cpu())*len(image);count+=len(image)
        scheduler.step();score=validate()
        row={"epoch":epoch+1,"training_masked_l1":total/count,"validation_masked_l1":score,
             "optimizer_steps":steps,"elapsed_seconds":time.monotonic()-started}
        history.append(row)
        if score<best:
            best,bad=score,0
            torch.save({"encoder_state_dict":{k:v.detach().cpu() for k,v in model.encoder.state_dict().items()},
                "reconstruction_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith("encoder.")},
                "stage":"self_supervised_masked_reconstruction","pretraining_complete":False,
                "training_complete":False,"semantic_training_complete":False,"accuracy_measured":False,
                "input_channels":4,"band_order":manifest["band_order"],"native_gsd_m":.6,
                "epoch":epoch+1,"validation_masked_l1":score,"config":config,"initialization":initialization},
                run_dir/"best_encoder.pt")
        else:
            bad+=1
        (run_dir/"history.json").write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
        if bad>=config["early_stopping_patience"]:
            break
    checkpoint=torch.load(run_dir/"best_encoder.pt",map_location="cpu",weights_only=True)
    checkpoint["pretraining_complete"]=True
    torch.save(checkpoint,run_dir/"best_encoder.pt")
    unchanged=file_digest(initialization_path)==initialization["sha256_before"]
    if not unchanged:
        raise RuntimeError("Original initialization checkpoint changed unexpectedly")
    result={"status":"self_supervised_encoder_pretrained_not_a_landcover_model",
        "run_directory":str(run_dir),"checkpoint":str(run_dir/"best_encoder.pt"),"epochs_run":len(history),
        "optimizer_steps":steps,"baseline_masked_l1":baseline,"best_validation_masked_l1":best,
        "encoder_conv1_mean_absolute_change":float((checkpoint["encoder_state_dict"]["conv1.weight"]-initial_conv).abs().mean()),
        "elapsed_seconds":time.monotonic()-started,"original_checkpoint_unchanged":unchanged,
        "semantic_accuracy":None,"semantic_miou":None,"semantic_training_complete":False,
        "platform_deployed":False,"test_pixels_used":False,
        "next_requirement":"Reviewed five-class labels and additional independent training areas"}
    (run_dir/"pretraining_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=Path("config/multispectral_scenes.json"))
    parser.add_argument("--config",type=Path,default=Path("config/pretraining_multispectral.json"))
    args=parser.parse_args()
    print(json.dumps(run_pretraining(args.manifest,args.config),indent=2))
