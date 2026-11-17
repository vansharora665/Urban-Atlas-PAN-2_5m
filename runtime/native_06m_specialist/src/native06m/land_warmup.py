"""Diagnostic four-land-class warm-up, never a deployable five-class model."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import CLASSES, NativePatches, file_digest, inspect_manifest, load_manifest, read_json, resolve
from .model import NativeAttentionUNetPP, choose_device


STAGE = "partial_land_only_supervised_warmup"
SUPPORTED = [1,2,3,4]


def land_targets(target):
    mapped = torch.full_like(target,255)
    valid = (target>=1)&(target<=4)
    mapped[valid] = target[valid]-1
    return mapped, valid


def land_loss(output, target, weights, road_aux_weight=.15, road_positive_weight=2.0, label_smoothing=0.0):
    mapped, valid = land_targets(target)
    if not valid.any():
        return output["semantic"].sum()*0
    logits = output["semantic"][:,1:5]
    ce = F.cross_entropy(logits,mapped,weight=weights,ignore_index=255,label_smoothing=label_smoothing)
    probabilities = logits.softmax(1)*valid[:,None]
    truth = F.one_hot(mapped.clamp_max(3),4).permute(0,3,1,2).float()*valid[:,None]
    intersection = (probabilities*truth).sum((0,2,3))
    denominator = (probabilities+truth).sum((0,2,3))
    present = truth.sum((0,2,3))>0
    dice = (1-(2*intersection+1e-5)/(denominator+1e-5))[present].mean()
    road = F.binary_cross_entropy_with_logits(output["road"].squeeze(1)[valid],(target==1).float()[valid],
              pos_weight=torch.as_tensor(road_positive_weight,device=target.device))
    return ce+.5*dice+road_aux_weight*road


def land_metrics(matrix):
    matrix = np.asarray(matrix,np.int64)
    true, predicted = matrix.sum(1),matrix.sum(0)
    tp = np.diag(matrix)
    union = true+predicted-tp
    rows = {}
    for i,class_id in enumerate(SUPPORTED):
        rows[CLASSES[class_id]] = {
            "reviewed_pixels":int(true[i]),
            "iou":float(tp[i]/union[i]) if union[i] else None,
            "recall":float(tp[i]/true[i]) if true[i] else None,
            "precision":float(tp[i]/predicted[i]) if predicted[i] else None,
            "f1":float(2*tp[i]/(true[i]+predicted[i])) if true[i]+predicted[i] else None}
    values = [r["iou"] for r in rows.values() if r["iou"] is not None]
    return {"conditional_land_pixel_accuracy":float(tp.sum()/matrix.sum()) if matrix.sum() else None,
            "conditional_land_mean_iou":float(np.mean(values)) if values else None,
            "land_class_order":SUPPORTED,"confusion_matrix_land_only":matrix.tolist(),
            "per_land_class":rows,"evaluated_reviewed_land_pixels":int(matrix.sum()),
            "water_class_trained":False,"water_class_evaluated":False,
            "scope":"Four-way prediction conditional on reviewed land pixels. Not five-class scene accuracy or independent test performance."}


class NativeLandViews(Dataset):
    def __init__(self,base,crop_size=384,repeats=4,road_focus_probability=0.0):
        self.base,self.crop_size,self.repeats = base,crop_size,repeats
        self.road_focus_probability = road_focus_probability

    def __len__(self):
        return len(self.base)*self.repeats

    def __getitem__(self,index):
        image,target = self.base[index%len(self.base)]
        height,width = target.shape
        road = torch.nonzero(target==1) if self.road_focus_probability>0 else None
        focus_road = road is not None and len(road)>0 and np.random.rand()<self.road_focus_probability
        for _ in range(12):
            if focus_road:
                cy,cx = road[int(np.random.randint(len(road)))].tolist()
                y = int(np.random.randint(max(0,cy-self.crop_size+1),min(cy,height-self.crop_size)+1))
                x = int(np.random.randint(max(0,cx-self.crop_size+1),min(cx,width-self.crop_size)+1))
            else:
                y = int(np.random.randint(height-self.crop_size+1))
                x = int(np.random.randint(width-self.crop_size+1))
            local = target[y:y+self.crop_size,x:x+self.crop_size]
            if int(((local>=1)&(local<=4)).sum())>=256 and (not focus_road or int((local==1).sum())>=32):
                return image[:,y:y+self.crop_size,x:x+self.crop_size].contiguous(),local.contiguous()
        # Do not train on an empty crop merely to satisfy a requested view count.
        return image,target


def evaluate_land(model,loader,device,transform=None,retain=False):
    matrix = np.zeros((4,4),np.int64)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for image,target in loader:
            if transform:
                image = transform(image)
            predicted = model(image.to(device))["semantic"][:,1:5].argmax(1).cpu().numpy()+1
            truth = target.numpy()
            valid = (truth>=1)&(truth<=4)
            matrix += np.bincount((truth[valid]-1)*4+predicted[valid]-1,minlength=16).reshape(4,4)
            if retain:
                predictions.extend([p.copy() for p in predicted])
    return land_metrics(matrix),predictions


def validate_warmup_manifest(manifest):
    report = inspect_manifest(manifest)
    if report["errors"]:
        raise ValueError("Invalid source/label grid: "+"; ".join(report["errors"]))
    for scene in manifest["scenes"]:
        if scene["label_provenance"]!="human_reviewed" or not scene.get("quality_status","").startswith("accepted_for_development"):
            raise ValueError("Warm-up still requires reviewed and QA-accepted labels")
        if scene.get("normalization_method")!="source_scene_p02_p98_per_band":
            raise ValueError("Warm-up needs source-scene normalization, not independently stretched tiles")
    train_groups = {s["geographic_group"] for s in manifest["scenes"] if s["split"]=="train"}
    val_groups = {s["geographic_group"] for s in manifest["scenes"] if s["split"]=="validation"}
    if not train_groups or not val_groups or train_groups & val_groups:
        raise ValueError("Warm-up needs disjoint fixed train and validation geographies")


def run_candidate(manifest,config,name,initialization_path,output):
    seed = config["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    # Test records are removed before a Dataset is constructed; no test target enters optimization or selection.
    working = {**manifest,"scenes":[s for s in manifest["scenes"] if s["split"] in {"train","validation"}]}
    validate_warmup_manifest(working)
    dataset_config = {"patch_size":512,"minimum_labeled_pixels_per_patch":256}
    train = NativePatches(working,"train",dataset_config,augment=True)
    validation = NativePatches(working,"validation",dataset_config,augment=False)
    if np.any(train.counts[1:5]<256) or np.any(validation.counts[1:5]<256):
        raise ValueError("Each supported land class needs reviewed train and validation examples")
    if train.counts[0]:
        raise ValueError("This diagnostic is only for the documented missing-training-water situation; use the full trainer when eligible")
    views = NativeLandViews(train,config["native_crop_size"],config["augmented_views_per_tile"],
                            config.get("road_focused_crop_probability",0.0))
    train_loader = DataLoader(views,batch_size=1,shuffle=True,num_workers=0)
    val_loader = DataLoader(validation,batch_size=1,shuffle=False,num_workers=0)
    train_check_loader = DataLoader(NativePatches(working,"train",dataset_config,augment=False),batch_size=1,shuffle=False)
    device = choose_device()
    if device.type=="cpu" and config.get("require_accelerator",True):
        raise RuntimeError("No accelerator available; do not silently start a slow CPU run")
    model = NativeAttentionUNetPP(in_channels=4)
    initialization = model.initialize_encoder(initialization_path)
    before_hash = file_digest(initialization_path)
    model.to(device)
    encoder = list(model.network.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{"params":encoder,"lr":config["encoder_lr"]},
                                  {"params":decoder,"lr":config["decoder_lr"]}],weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config["epochs"])
    weights = np.sqrt(train.counts[1:5].sum()/train.counts[1:5])
    weights = np.clip(weights/weights.mean(),.5,2.0)
    class_weights = torch.tensor(weights,dtype=torch.float32,device=device)
    output.mkdir(parents=True,exist_ok=False)
    setup = {"stage":STAGE,"candidate":name,"initialization":initialization,"initialization_sha256":before_hash,
             "supported_classes":SUPPORTED,"unsupported_class":0,"training_complete":False,
             "full_five_class_training_started":False,"platform_deployed":False,
             "device":str(device),"unique_train_tiles":len(train),"unique_validation_tiles":len(validation),
             "augmented_views_per_epoch":len(views),"views_are_not_independent_samples":True,
             "train_class_pixels_0_to_4":train.counts.tolist(),"validation_class_pixels_0_to_4":validation.counts.tolist(),
             "class_weights_land_only":weights.tolist(),"train_tile_ids":[s["tile_id"] for s in working["scenes"] if s["split"]=="train"],
             "validation_tile_ids":[s["tile_id"] for s in working["scenes"] if s["split"]=="validation"],
             "reserved_test_tile_ids":[s["tile_id"] for s in manifest["scenes"] if s["split"]=="test"],
             "test_predictions_used":False,"test_label_pixels_used_for_training":False,
             "normalization":"Fixed per-parent-scene p02/p98, identical native scale to inference",
             "augmentation":"Existing paired flips/90-degree rotations, common-channel brightness/contrast/gamma, mild blur/noise, random native crops; no downsampling",
             "boundary_and_topology_losses":"Disabled for sparse warm-up: true class interfaces are mostly unreviewed, not known background",
             "full_training_guard_unchanged":True,"config":config}
    (output/"training_setup.json").write_text(json.dumps(setup,indent=2)+"\n")
    initial,_ = evaluate_land(model,val_loader,device)
    history,best,bad,updates = [],-1.0,0,0
    started = time.monotonic()
    accumulation = config["gradient_accumulation"]
    for epoch in range(config["epochs"]):
        model.train()
        for parameter in encoder:
            parameter.requires_grad_(epoch>=config["encoder_freeze_epochs"])
        for module in model.network.encoder.modules():
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step,(image,target) in enumerate(train_loader):
            image,target = image.to(device),target.to(device)
            loss = land_loss(model(image),target,class_weights,label_smoothing=config.get("label_smoothing",0.0))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite warm-up loss")
            start = (step//accumulation)*accumulation
            size = min(accumulation,len(train_loader)-start)
            (loss/size).backward()
            losses.append(float(loss.detach().cpu()))
            if (step+1)%accumulation==0 or step+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
        scheduler.step()
        metrics,_ = evaluate_land(model,val_loader,device)
        row = {"epoch":epoch+1,"training_loss":float(np.mean(losses)),"optimizer_updates":updates,**metrics}
        history.append(row)
        score = metrics["conditional_land_mean_iou"]
        if score is not None and score>best:
            best,bad = score,0
            torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        "stage":STAGE,"warmup_complete":False,"training_complete":False,
                        "platform_deployed":False,"supported_classes":SUPPORTED,"unsupported_classes":[0],
                        "config":{**config,"in_channels":4,"encoder_name":"resnet34"},
                        "epoch":epoch+1,"validation_metrics_land_only":metrics,"initialization":initialization},output/"best_partial_warmup.pt")
        else:
            bad+=1
        (output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        print(json.dumps({"candidate":name,"epoch":epoch+1,"land_only_validation_miou":score,
                          "road_iou":metrics["per_land_class"][CLASSES[1]]["iou"],"updates":updates,
                          "elapsed_seconds":round(time.monotonic()-started,2)},allow_nan=False),flush=True)
        if (epoch+1>=config["minimum_epochs"] and bad>=config["patience"]) or time.monotonic()-started>config["max_seconds_per_candidate"]:
            break
    checkpoint = torch.load(output/"best_partial_warmup.pt",map_location="cpu",weights_only=True)
    checkpoint["warmup_complete"] = True
    torch.save(checkpoint,output/"best_partial_warmup.pt")
    model.load_state_dict(checkpoint["state_dict"])
    train_metrics,_ = evaluate_land(model,train_check_loader,device)
    validation_metrics,predictions = evaluate_land(model,val_loader,device,retain=True)
    stress = {}
    for name_stress,transform in {
        "darker_20pct":lambda x:torch.clamp(x*.8,0,1),
        "brighter_20pct":lambda x:torch.clamp(x*1.2,0,1),
        "gamma_0_8":lambda x:torch.clamp(x,0,1)**.8,
        "gamma_1_2":lambda x:torch.clamp(x,0,1)**1.2}.items():
        stress[name_stress] = evaluate_land(model,val_loader,device,transform)[0]
    np.savez_compressed(output/"validation_predictions_land_only.npz",**{
        validation.items[i][0]["tile_id"]:p for i,p in enumerate(predictions)})
    result = {"candidate":name,"run_directory":str(output.resolve()),"stage":STAGE,"epochs_run":len(history),
              "best_epoch":checkpoint["epoch"],"optimizer_updates":updates,"device":str(device),
              "elapsed_seconds":round(time.monotonic()-started,3),"initial_validation_land_only":initial,
              "training_land_only":train_metrics,"validation_land_only":validation_metrics,
              "illumination_stress_validation_only":stress,"original_checkpoint_unchanged":file_digest(initialization_path)==before_hash,
              "water_class_trained":False,"training_complete":False,"warmup_complete":True,
              "platform_deployed":False,"test_evaluated":False,
              "warning":"Conditional four-land-class warm-up on tiny sparse pilot labels. Not a five-class trained model, not independent test accuracy, and not an 80-90% generalization claim."}
    (output/"warmup_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    del optimizer,model
    if device.type=="mps":
        torch.mps.empty_cache()
    return result


def run(manifest_path,config_path):
    config,base = read_json(config_path)
    manifest = load_manifest(manifest_path)
    root = resolve(base,config["runs_directory"])/("land_warmup_ablation_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    root.mkdir(parents=True,exist_ok=False)
    results = []
    for candidate in config["candidates"]:
        result = run_candidate(manifest,config,candidate["name"],resolve(base,candidate["initialization"]),root/candidate["name"])
        results.append(result)
        (root/"ablation_progress.json").write_text(json.dumps(results,indent=2)+"\n")
    best = max(results,key=lambda r:r["validation_land_only"]["conditional_land_mean_iou"])
    report = {"stage":STAGE,"manifest":str(manifest_path.resolve()),"manifest_sha256":file_digest(manifest_path),
              "candidates":results,"best_partial_candidate":best["candidate"],
              "selection_rule":"Largest clean validation conditional four-land-class macro IoU; no test selection",
              "full_five_class_training_started":False,"full_training_guard_unchanged":True,
              "platform_deployed":False}
    (root/"ablation_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"ablation_directory":str(root),"best_partial_candidate":best["candidate"],
                      "full_five_class_model_ready":False},indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("config/land_warmup_ablation.json"))
    args=parser.parse_args()
    run(args.manifest,args.config)
