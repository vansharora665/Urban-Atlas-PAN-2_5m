from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from shapely.geometry import box
import torch
from torch.utils.data import Dataset, Sampler


IGNORE = 255
CLASSES = {0: "Water Bodies", 1: "Roads", 2: "Urban Built-up", 3: "Barren / Empty Land", 4: "Vegetation"}


def read_json(path):
    path = Path(path).resolve()
    return json.loads(path.read_text()), path.parent


def resolve(base, path):
    return (base / path).resolve() if path else None


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    manifest, base = read_json(path)
    if {int(k): v for k,v in manifest["classes"].items()} != CLASSES or manifest["ignore_index"] != IGNORE:
        raise ValueError("Use the fixed five-class schema and ignore value 255")
    ids = set()
    for scene in manifest["scenes"]:
        if scene["scene_id"] in ids:
            raise ValueError("Scene IDs must be unique")
        ids.add(scene["scene_id"])
        if not scene.get("geographic_group"):
            raise ValueError("Each scene needs a geographic_group")
        if scene["split"] not in {"train", "validation", "test", "unassigned"}:
            raise ValueError("Unknown split")
        scene["images"] = [str(resolve(base, p)) for p in scene["images"]]
        for key in ("labels", "quality_mask", "normalization_reference"):
            scene[key] = str(resolve(base, scene[key])) if scene.get(key) else None
    return manifest


def inspect_manifest(manifest):
    rows, errors, footprints = [], [], []
    split_by_group = {}
    for scene in manifest["scenes"]:
        group, split = scene["geographic_group"], scene["split"]
        if group in split_by_group and split_by_group[group] != split:
            errors.append(f"Geographic group {group} is present in multiple splits")
        split_by_group[group] = split
        first_grid = None
        for image in scene["images"]:
            try:
                with rasterio.open(image) as source:
                    grid = (source.shape, source.crs, source.transform)
                    meter_crs = bool(source.crs and source.crs.is_projected and source.crs.linear_units == "metre")
                    compatible = meter_crs and all(abs(r-manifest["target_gsd_m"]) <= manifest["gsd_tolerance_m"] for r in source.res)
                    if not compatible:
                        errors.append(f"{image}: not a verified 0.6 m metric grid")
                    channels = manifest.get("input_channels",1)
                    if channels not in (1,4) or source.count != channels:
                        errors.append(f"{image}: expected exactly {channels} declared input channels")
                    if channels==4 and (len(scene["images"])!=1 or manifest.get("band_order")!=["blue","green","red","nir"]):
                        errors.append(f"{image}: four-band inputs must be one explicit B/G/R/NIR stack, not channel variants")
                    if source.transform.b or source.transform.d or source.transform.a <= 0 or source.transform.e >= 0:
                        errors.append(f"{image}: requires a north-up grid; reproject explicitly before training")
                    if first_grid is None:
                        first_grid = grid
                        bounds = transform_bounds(source.crs, "EPSG:4326", *source.bounds)
                        footprints.append((scene, box(*bounds)))
                    elif grid != first_grid:
                        errors.append(f"{scene['scene_id']}: channel variants are not exactly co-registered")
                    row = {"scene_id":scene["scene_id"],"geographic_group":group,"split":split,
                           "image":image,"width":source.width,"height":source.height,
                           "gsd_m":list(source.res),"crs":str(source.crs),
                           "labels_available":bool(scene.get("labels")),
                           "label_provenance":scene.get("label_provenance")}
                    sample = source.read(1,out_shape=(min(512,source.height),min(512,source.width)))
                    valid = np.isfinite(sample)
                    if source.nodata is not None:
                        valid &= sample != source.nodata
                    row["sample_q02_q98"] = np.percentile(sample[valid],[2,98]).tolist() if valid.any() else None
                    rows.append(row)
            except (rasterio.errors.RasterioError, OSError) as ex:
                errors.append(f"{image}: {ex}")
        for key in ("labels", "quality_mask"):
            if not scene.get(key):
                continue
            try:
                with rasterio.open(scene[key]) as source:
                    if (source.shape,source.crs,source.transform) != first_grid or source.count != 1:
                        errors.append(f"{scene['scene_id']}: {key} is not exactly aligned with the input")
                    if source.dtypes[0] != "uint8":
                        errors.append(f"{scene['scene_id']}: {key} must be uint8")
                    if key == "labels" and source.nodata != IGNORE:
                        errors.append(f"{scene['scene_id']}: label nodata must be 255, never class 0 (water)")
            except (rasterio.errors.RasterioError, OSError) as ex:
                errors.append(f"{scene['scene_id']}: {key}: {ex}")
    for i, (a, shape_a) in enumerate(footprints):
        for b, shape_b in footprints[i+1:]:
            if a["split"] == b["split"] or "unassigned" in {a["split"],b["split"]}:
                continue
            overlap = shape_a.intersection(shape_b).area
            if overlap > 1e-8 * min(shape_a.area,shape_b.area):
                errors.append(f"Overlapping footprints cross splits: {a['scene_id']} and {b['scene_id']}")
    return {"rasters":rows,"raster_count":len(rows),"geographic_groups":len(split_by_group),
            "reviewed_label_scene_count":sum(bool(s.get("labels")) and s.get("label_provenance")=="human_reviewed" for s in manifest["scenes"]),
            "errors":errors,"ready_for_training":False}


def training_preflight(manifest, config):
    report = inspect_manifest(manifest)
    errors = report["errors"]
    for scene in manifest["scenes"]:
        if scene.get("quality_status")=="needs_revision":
            errors.append(f"{scene['scene_id']}: annotation quality review requires revision before training")
        if scene.get("quality_status") in {"draft_pending_human_review","awaiting_visual_qa"}:
            errors.append(f"{scene['scene_id']}: draft/review QA has not been accepted for supervised training")
    if manifest.get("input_channels",1)!=config.get("in_channels",1):
        errors.append("Manifest and model input-channel counts differ")
    if not report["reviewed_label_scene_count"]:
        errors.append("No scenes have declared human-reviewed five-class pixel labels")
    required = {"train":config["minimum_train_geographic_groups"],
                "validation":config["minimum_validation_geographic_groups"],
                "test":config["minimum_test_geographic_groups"]}
    for split, minimum in required.items():
        scenes = [s for s in manifest["scenes"] if s["split"]==split]
        if len({s["geographic_group"] for s in scenes}) < minimum:
            errors.append(f"Need at least {minimum} independent {split} geographic groups")
        for scene in scenes:
            if not scene.get("labels"):
                errors.append(f"{scene['scene_id']}: reviewed pixel labels are missing")
            if scene.get("label_provenance") != "human_reviewed":
                errors.append(f"{scene['scene_id']}: labels must be declared human_reviewed; predictions/pseudo-labels are not independent truth")
    report["ready_for_training"] = not errors
    return report


def normalization(path, quality_path=None):
    with rasterio.open(path) as source:
        scale = min(1, 2048/max(source.shape))
        shape = (max(1,round(source.height*scale)),max(1,round(source.width*scale)))
        image = source.read(out_shape=(source.count,*shape)).astype(np.float32)
        valid = np.isfinite(image).all(0)
        valid &= (source.read_masks(out_shape=(source.count,*shape))>0).all(0)
        if source.nodata is not None:
            valid &= (image != source.nodata).all(0)
        if source.count==4:
            valid &= np.any(image!=0,axis=0)
        if quality_path:
            with rasterio.open(quality_path) as quality:
                if quality.shape!=source.shape or quality.crs!=source.crs or quality.transform!=source.transform:
                    raise ValueError("Quality mask and source grids differ")
                valid &= quality.read(1,out_shape=shape,resampling=rasterio.enums.Resampling.nearest)==0
        if not valid.any():
            raise ValueError(f"No valid source pixels: {path}")
        limits = np.array([np.percentile(channel[valid],[2,98]) for channel in image])
        if np.any(limits[:,1]<=limits[:,0]):
            raise ValueError(f"No usable intensity variation in at least one band: {path}")
        return tuple(map(float,limits[0])) if source.count==1 else limits


def starts(length, patch, stride):
    if length <= patch:
        return [0]
    result = list(range(0,length-patch+1,stride))
    if result[-1] != length-patch:
        result.append(length-patch)
    return result


def read_native(source, x, y, size, limits):
    height,width = min(size,source.height-y),min(size,source.width-x)
    # No out_shape or interpolation: these are native 0.6 m pixels.
    raw = source.read(window=Window(x,y,width,height)).astype(np.float32)
    valid = valid_native_pixels(source,raw,Window(x,y,width,height))
    limits = np.asarray(limits).reshape(source.count,2)
    image = np.clip((raw-limits[:,0,None,None])/np.maximum(limits[:,1,None,None]-limits[:,0,None,None],1e-6),0,1)
    image = image[0] if source.count==1 else image.transpose(1,2,0)
    image[~valid] = .5
    padding = ((0,size-height),(0,size-width)) + (((0,0),) if source.count>1 else ())
    image = np.pad(image,padding,mode="edge")
    valid = np.pad(valid,((0,size-height),(0,size-width)),constant_values=False)
    return image.astype(np.float32),valid


def valid_native_pixels(source, raw, window):
    """Use the same all-band validity rule for counting labels and loading them."""
    valid = np.isfinite(raw).all(0) & (source.read_masks(window=window)>0).all(0)
    if source.count==4:
        valid &= np.any(raw!=0,axis=0)
    return valid


class NativePatches(Dataset):
    def __init__(self, manifest, split, config, augment=False):
        self.patch = config["patch_size"]
        self.augment = augment
        self.items,self.counts,self.limits = [],np.zeros(5,np.int64),{}
        minimum = config["minimum_labeled_pixels_per_patch"]
        for scene in manifest["scenes"]:
            if scene["split"] != split:
                continue
            with rasterio.open(scene["labels"]) as labels,rasterio.open(scene["images"][0]) as image_source:
                quality = rasterio.open(scene["quality_mask"]) if scene.get("quality_mask") else None
                try:
                    for y in range(0,labels.height,self.patch):
                        for x in range(0,labels.width,self.patch):
                            window = Window(x,y,min(self.patch,labels.width-x),min(self.patch,labels.height-y))
                            target = labels.read(1,window=window)
                            if not np.isin(target,[0,1,2,3,4,IGNORE]).all():
                                raise ValueError(f"Invalid class IDs in {scene['labels']}")
                            valid = target != IGNORE
                            raw = image_source.read(window=window)
                            valid &= valid_native_pixels(image_source,raw,window)
                            if quality is not None:
                                valid &= quality.read(1,window=window)==0
                            if valid.sum() < minimum:
                                continue
                            counts = np.bincount(target[valid],minlength=5)
                            self.counts += counts
                            variants = [None] if augment else list(range(len(scene["images"])))
                            self.items.extend((scene,x,y,variant) for variant in variants)
                finally:
                    if quality:
                        quality.close()
            for path in scene["images"]:
                if scene.get("normalization_limits") is not None:
                    limits = np.asarray(scene["normalization_limits"],dtype=np.float32)
                    with rasterio.open(path) as source:
                        expected = (source.count,2)
                    if limits.shape!=expected or not np.isfinite(limits).all() or np.any(limits[:,1]<=limits[:,0]):
                        raise ValueError("Invalid cached source-scene normalization limits")
                    if scene.get("normalization_method")!="source_scene_p02_p98_per_band" or not scene.get("normalization_reference"):
                        raise ValueError("Cached limits require explicit source-scene normalization provenance")
                    self.limits[path] = limits
                else:
                    self.limits[path] = normalization(path,scene.get("quality_mask"))
        if not self.items:
            raise ValueError(f"No eligible {split} patches with reviewed labels")

    def __len__(self):
        return len(self.items)

    def __getitem__(self,index):
        scene,x,y,variant = self.items[index]
        # Channel variants never cross splits; validation uses a fixed variant.
        path = scene["images"][int(np.random.randint(len(scene["images"])))] if self.augment else scene["images"][variant]
        with rasterio.open(path) as source:
            image,valid = read_native(source,x,y,self.patch,self.limits[path])
        with rasterio.open(scene["labels"]) as source:
            target = source.read(1,window=Window(x,y,self.patch,self.patch),boundless=True,fill_value=IGNORE)
        if scene.get("quality_mask"):
            with rasterio.open(scene["quality_mask"]) as source:
                quality = source.read(1,window=Window(x,y,self.patch,self.patch),boundless=True,fill_value=IGNORE)
            valid &= quality==0
        target[~valid] = IGNORE
        image[~valid] = .5
        if self.augment:
            k = int(np.random.randint(4))
            image,target = np.rot90(image,k),np.rot90(target,k)
            if np.random.rand()<.5:
                image,target = np.fliplr(image),np.fliplr(target)
            if np.random.rand()<.5:
                image,target = np.flipud(image),np.flipud(target)
            image = np.clip(image*np.random.uniform(.85,1.15)+np.random.uniform(-.06,.06),0,1)
            image = image**np.random.uniform(.85,1.2)
            if np.random.rand()<.2:
                image = cv2.GaussianBlur(image,(3,3),np.random.uniform(.2,.6))
            if np.random.rand()<.2:
                image = np.clip(image+np.random.normal(0,.008,image.shape),0,1)
        # Unannotated pixels still provide image context; only their loss is ignored.
        channels = image[None] if image.ndim==2 else image.transpose(2,0,1)
        return torch.from_numpy(np.ascontiguousarray(channels)).float(),torch.from_numpy(np.ascontiguousarray(target)).long()


class GeographicSampler(Sampler):
    """Limit each epoch without letting one large scene dominate or repeat tiny scenes."""
    def __init__(self, dataset, budget, seed):
        self.groups = {}
        for index,(scene,*_) in enumerate(dataset.items):
            self.groups.setdefault(scene["geographic_group"],[]).append(index)
        if budget<len(self.groups):
            raise ValueError("Epoch budget must cover every training geography")
        self.per_group = max(1,budget//len(self.groups))
        self.seed,self.epoch = seed,0

    def __len__(self):
        return sum(min(self.per_group,len(indices)) for indices in self.groups.values())

    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        chosen=[]
        for indices in self.groups.values():
            chosen.extend(rng.sample(indices,min(self.per_group,len(indices))))
        rng.shuffle(chosen)
        return iter(chosen)
