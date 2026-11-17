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
from torch.utils.data import DataLoader

from .data import GeographicSampler, NativePatches, file_digest, load_manifest, read_json, resolve, training_preflight
from .losses import confusion_metrics, segmentation_loss
from .model import NativeAttentionUNetPP, choose_device


def evaluate(model, loader, device):
    matrix = np.zeros((5,5),np.int64)
    variant_matrices = {}
    offset = 0
    model.eval()
    with torch.inference_mode():
        for image,target in loader:
            prediction = model(image.to(device))["semantic"].argmax(1).cpu().numpy()
            target = target.numpy()
            valid = target!=255
            for index in range(len(target)):
                sample_valid = valid[index]
                sample_matrix = np.bincount(target[index][sample_valid]*5+prediction[index][sample_valid],minlength=25).reshape(5,5)
                matrix += sample_matrix
                scene,_,_,variant = loader.dataset.items[offset+index]
                # Different review tiles are pieces of one geography/variant, not independent variants.
                key = (scene["geographic_group"],int(variant))
                variant_matrices.setdefault(key,np.zeros((5,5),np.int64))
                variant_matrices[key] += sample_matrix
            offset += len(target)
    metrics = confusion_metrics(matrix)
    groups = {}
    for (group,_),local_matrix in variant_matrices.items():
        score = confusion_metrics(local_matrix)["mean_iou"]
        if score is not None:
            groups.setdefault(group,[]).append(score)
    metrics["mean_iou_per_geographic_group"] = {key:float(np.mean(values)) for key,values in groups.items()}
    metrics["geographic_macro_mean_iou"] = float(np.mean(list(metrics["mean_iou_per_geographic_group"].values()))) if groups else None
    metrics["geographic_aggregation"] = "Pool crop/patch confusion matrices within each geography and channel variant, then average variants and geographies"
    return metrics


PILOT_STAGE = "limited_geography_fiveclass_development_pilot"


def run_training(manifest_path, config_path, *, development_pilot=False):
    config,base = read_json(config_path)
    if config.get("model_stage")==PILOT_STAGE and not development_pilot:
        raise ValueError("Use the separate native06m.pilot command; a pilot config cannot produce a completed full model")
    if config.get("model_stage") not in {None, "full_five_class_training", PILOT_STAGE}:
        raise ValueError("A specialized development config must use its dedicated runner, not the full trainer")
    manifest = load_manifest(manifest_path)
    preflight = training_preflight(manifest,config)
    full_training_preflight = preflight
    if development_pilot:
        if config.get("model_stage")!=PILOT_STAGE:
            raise ValueError("A development pilot requires an explicit non-deployable stage")
        # This separate stage never marks training_complete or changes the full-run config.
        preflight = training_preflight(manifest,{**config,"minimum_train_geographic_groups":1})
    if not preflight["ready_for_training"]:
        raise ValueError("Fine-tuning has not started.\n"+"\n".join(preflight["errors"]))
    seed = config["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    train = NativePatches(manifest,"train",config,augment=True)
    validation = NativePatches(manifest,"validation",config)
    minimum = config["minimum_class_pixels_in_train_and_validation"]
    for name,data in [("train",train),("validation",validation)]:
        if np.any(data.counts<minimum):
            raise ValueError(f"Insufficient reviewed pixels for every class in {name}: {data.counts.tolist()}")
    loader_args = {"batch_size":config["batch_size"],"num_workers":config["num_workers"]}
    train_sampler = GeographicSampler(train,config["max_train_patches_per_epoch"],seed)
    train_loader = DataLoader(train,sampler=train_sampler,**loader_args)
    val_loader = DataLoader(validation,shuffle=False,**loader_args)
    device = choose_device()
    model = NativeAttentionUNetPP(config["encoder_name"],config.get("in_channels",1))
    initialization_path = resolve(base,config["encoder_initialization_checkpoint"])
    if development_pilot and config.get("partial_initialization_checkpoint"):
        initialization_path = resolve(base,config["partial_initialization_checkpoint"])
        partial = torch.load(initialization_path,map_location="cpu",weights_only=True)
        if (partial.get("stage")!="partial_land_only_supervised_warmup" or
                not partial.get("warmup_complete") or partial.get("training_complete") is not False or
                partial.get("supported_classes")!=[1,2,3,4] or partial.get("unsupported_classes")!=[0]):
            raise ValueError("Pilot initialization must be a completed, explicitly partial land warm-up")
        model.load_state_dict(partial["state_dict"],strict=True)
        with torch.no_grad():
            nn.init.xavier_uniform_(model.network.segmentation_head[0].weight[0:1])
            model.network.segmentation_head[0].bias[0]=0
        initialization = {"path":str(initialization_path),"scope":"reviewed_partial_land_model_with_fresh_water_logit",
                          "water_logit_reset":True,"old_checkpoint_modified":False}
    else:
        initialization = model.initialize_encoder(initialization_path)
    initialization["sha256"] = file_digest(initialization_path)
    model.to(device)
    encoder_parameters = list(model.network.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_parameters}
    other_parameters = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{"params":encoder_parameters,"lr":config["encoder_lr"]},
                                  {"params":other_parameters,"lr":config["decoder_lr"]}],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["epochs"])
    weights = np.sqrt(train.counts.sum()/np.maximum(train.counts,1))
    weights = weights/weights.mean()
    weights = np.clip(weights,.5,config["max_class_weight"])
    weights = torch.tensor(weights,dtype=torch.float32,device=device)
    run_dir = resolve(base,config["runs_directory"])/f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True,exist_ok=False)
    split_snapshot = {"manifest":manifest,"config":config,"initialization":initialization,
                      "train_patches":len(train),"validation_patches":len(validation),
                      "train_class_pixels":train.counts.tolist(),"validation_class_pixels":validation.counts.tolist(),
                      "class_weights":weights.cpu().tolist(),"device":str(device),
                      "patches_per_epoch":len(train_sampler),"sampling":"Equal capped budget per geography, without replacement",
                      "test_label_pixels_read_for_training":False,
                      "validation_channel_variants":"All variants; pooled metrics do not count them as independent geographies",
                      "label_scope":"Human-reviewed labels as declared in manifest; provenance is a user attestation, not automatic verification"}
    split_snapshot.update(stage=PILOT_STAGE if development_pilot else "full_five_class_training",
                          development_pilot=development_pilot,full_training_preflight=full_training_preflight,
                          training_complete=False,platform_deployed=False)
    (run_dir/"training_setup.json").write_text(json.dumps(split_snapshot,indent=2))
    best,bad,history = -1.0,0,[]
    started = time.monotonic()
    accumulation = config["gradient_accumulation"]
    if accumulation<1:
        raise ValueError("gradient_accumulation must be positive")
    for epoch in range(config["epochs"]):
        train_sampler.epoch=epoch
        model.train()
        for parameter in encoder_parameters:
            parameter.requires_grad_(epoch>=config.get("encoder_freeze_epochs",0))
        if config.get("freeze_encoder_batchnorm",False):
            for module in model.network.encoder.modules():
                if isinstance(module,nn.BatchNorm2d):module.eval()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step,(image,target) in enumerate(train_loader):
            image,target = image.to(device),target.to(device)
            output = model(image)
            loss = segmentation_loss(output,target,weights,config)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; checkpoint not promoted")
            # A shorter final accumulation group is scaled by its actual size.
            group_start = (step//accumulation)*accumulation
            group_size = min(accumulation,len(train_loader)-group_start)
            (loss/group_size).backward()
            total_loss += float(loss.detach().cpu())
            if (step+1)%accumulation==0 or step+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        metrics = evaluate(model,val_loader,device)
        row = {"epoch":epoch+1,"training_loss":total_loss/max(1,len(train_loader)),**metrics}
        history.append(row)
        score = metrics["geographic_macro_mean_iou"]
        if score is not None and score>best:
            best,bad = score,0
            # Saving a CPU copy avoids device-dependent inference checkpoints.
            state = {key:value.detach().cpu() for key,value in model.state_dict().items()}
            torch.save({"state_dict":state,"config":config,"initialization":initialization,
                        "native_gsd_m":.6,"class_schema":manifest["classes"],
                        "epoch":epoch+1,"validation_metrics":metrics,
                        "validation_scope":"Held-out geographic groups with declared reviewed labels",
                        "stage":PILOT_STAGE if development_pilot else "full_five_class_training",
                        "development_pilot":development_pilot,
                        "supported_classes":[0,1,2,3,4],"unsupported_classes":[],
                        "water_class_trained":True,
                        "training_complete":False,"platform_deployed":False},run_dir/"best_model.pt")
        else:
            bad += 1
        (run_dir/"history.json").write_text(json.dumps(history,indent=2))
        print(json.dumps({"epoch":epoch+1,"validation_geographic_macro_miou":score,
                          "validation_pooled_pixel_miou":metrics["mean_iou"],"road_iou":metrics["class_iou"][1],
                          "road_as_water_rate":metrics["road_as_water_rate"],"run":str(run_dir)}),flush=True)
        if bad>=config["early_stopping_patience"]:
            break
    if not (run_dir/"best_model.pt").exists():
        raise RuntimeError("No valid validation checkpoint was produced")
    checkpoint = torch.load(run_dir/"best_model.pt",map_location="cpu",weights_only=True)
    checkpoint["training_complete"] = not development_pilot
    checkpoint["pilot_complete"] = development_pilot
    torch.save(checkpoint,run_dir/"best_model.pt")
    result = {"status":"pilot_complete_not_deployable" if development_pilot else "trained_not_deployed","run_directory":str(run_dir),
              "best_validation_geographic_macro_iou":best,"epochs_run":len(history),
              "stage":PILOT_STAGE if development_pilot else "full_five_class_training",
              "training_complete":not development_pilot,"platform_deployed":False,
              "source_initialization_unchanged":file_digest(initialization_path)==initialization["sha256"],
              "elapsed_seconds":time.monotonic()-started,
              "held_out_test_evaluation":"Not run; use the separate evaluate command after model selection"}
    (run_dir/"training_summary.json").write_text(json.dumps(result,indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,default=Path("config/scenes.json"))
    parser.add_argument("--config",type=Path,default=Path("config/training.json"))
    args = parser.parse_args()
    print(json.dumps(run_training(args.manifest,args.config),indent=2))


if __name__=="__main__":
    main()
