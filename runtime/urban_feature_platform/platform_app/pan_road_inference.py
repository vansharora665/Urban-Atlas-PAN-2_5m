"""Streaming native-detail PAN road pass with bounded GPU/RAM use."""

from dataclasses import asdict
import tempfile
from pathlib import Path

import cv2
import hashlib
import joblib
import numpy as np
import rasterio
import torch

from .image_utils import read_normalized_raster_patch, read_raster_below_threshold_mask
from .pan_overlap import blend_probabilities
from .pan_roads import RoadContinuity, connected_roads
from .pan_road_features import road_structure_features


def apply_connected_pan_roads(service, loaded, prediction):
    labels = prediction.full_label_map[:loaded.height,:loaded.width]
    patch_size = prediction.record.patch_size
    source_size, effective_gsd, scale = service._raster_inference_geometry(loaded,patch_size)
    ratio = source_size / patch_size
    width, height = int(np.ceil(loaded.width/ratio)), int(np.ceil(loaded.height/ratio))
    model, device = service._get_learned_road_head(
        len(prediction.record.class_names),
        prediction.record.summary.get("road_specialist",{}).get("checkpoint_sha256"))
    config = RoadContinuity(**prediction.record.summary.get("road_specialist",{}).get("continuity",{}))
    verifier_spec=prediction.record.summary.get("road_specialist",{}).get("geometry_verifier")
    verifier=None
    if verifier_spec:
        verifier_path=(service.settings.root_dir/verifier_spec["path"]).resolve()
        key=(str(verifier_path),verifier_spec.get("sha256"))
        if getattr(service,"_pan_road_verifier_key",None)!=key:
            raw=verifier_path.read_bytes()
            if verifier_spec.get("sha256") and hashlib.sha256(raw).hexdigest()!=verifier_spec["sha256"]:
                raise ValueError("Road geometry verifier checkpoint fingerprint mismatch")
            bundle=joblib.load(verifier_path)
            if bundle["checkpoint_sha256"]!=service._road_head_fingerprint:
                raise ValueError("Road geometry verifier was trained with a different road expert")
            service._pan_road_verifier=bundle["classifier"]
            service._pan_road_verifier_key=key
        verifier=service._pan_road_verifier
    cloud_threshold = service._cloud_threshold(loaded)
    bright_threshold = service._road_cloud_threshold(loaded)
    audit = {}
    changed = 0
    # Disk-backed probabilities avoid the former scene-size-dependent 5 m reduction.
    with tempfile.TemporaryDirectory(prefix="pan-road-v9-",dir=service.settings.runtime_dir) as temporary:
        probabilities = np.memmap(Path(temporary)/"road.f32",mode="w+",dtype=np.float32,shape=(height,width))
        coverage = np.memmap(Path(temporary)/"valid.u8",mode="w+",dtype=np.uint8,shape=(height,width))
        try:
            with rasterio.open(loaded.source_path) as source:
                def read(x,y):
                    sx,sy=int(round(x*ratio)),int(round(y*ratio))
                    image, valid = read_normalized_raster_patch(
                        source,x0=sx,y0=sy,patch_size=patch_size,source_window_size=source_size,
                        normalization_low=float(loaded.normalization_low or 0),
                        normalization_high=float(loaded.normalization_high or 1),
                        cloud_threshold=cloud_threshold,cloud_mask=loaded.cloud_mask,
                        enhance_local_contrast=service._use_local_contrast(loaded,scale))
                    if bright_threshold is not None:
                        valid &= read_raster_below_threshold_mask(
                            source,x0=sx,y0=sy,patch_size=patch_size,
                            source_window_size=source_size,threshold=bright_threshold)
                    return image,valid

                def predict(images):
                    with torch.inference_mode():
                        tensor=torch.from_numpy(images).unsqueeze(1).float().to(device)
                        probability=model(tensor).softmax(1)[:,1:2].cpu().numpy()
                    weight=max(.1,service.settings.road_head_weight)
                    return weight*probability/(1-probability+weight*probability+1e-7)

                def emit(y,probability,valid):
                    probabilities[y:y+valid.shape[0]]=probability[0]
                    coverage[y:y+valid.shape[0]]=valid

                blend_probabilities(width,height,1,read,predict,emit,window=patch_size,
                                    stride=patch_size//2,batch_size=min(32,service.settings.prediction_batch_patches))
                probabilities.flush()
                coverage.flush()
                for y in range(0,height,512):
                    for x in range(0,width,512):
                        y0,x0=max(0,y-64),max(0,x-64)
                        y1,x1=min(height,y+576),min(width,x+576)
                        sy0,sx0=int(round(y0*ratio)),int(round(x0*ratio))
                        sy1,sx1=min(loaded.height,int(round(y1*ratio))),min(loaded.width,int(round(x1*ratio)))
                        shape=(y1-y0,x1-x0)
                        base=cv2.resize(labels[sy0:sy1,sx0:sx1],(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
                        valid=np.asarray(coverage[y0:y1,x0:x1],bool)
                        probability=np.asarray(probabilities[y0:y1,x0:x1])
                        raw=source.read(1,window=rasterio.windows.Window(sx0,sy0,sx1-sx0,sy1-sy0),
                                        out_shape=shape,resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
                        low,high=float(loaded.normalization_low or 0),float(loaded.normalization_high or 1)
                        image=np.clip((raw-low)/max(high-low,1e-6),0,1)
                        if verifier is not None:
                            features=road_structure_features(image,probability)
                            probability=verifier.predict_proba(features.reshape(-1,20))[:,1].reshape(shape).astype(np.float32)
                        local=connected_roads(probability,base,valid,
                                              gsd_m=effective_gsd or 2.5,config=config,audit=audit,image=image)
                        h,w=min(512,height-y),min(512,width-x)
                        core=local[y-y0:y-y0+h,x-x0:x-x0+w]
                        ty,tx=int(round(y*ratio)),int(round(x*ratio))
                        by,bx=min(loaded.height,int(round((y+h)*ratio))),min(loaded.width,int(round((x+w)*ratio)))
                        target=labels[ty:by,tx:bx]
                        candidate=cv2.resize(core.astype(np.uint8),(bx-tx,by-ty),interpolation=cv2.INTER_NEAREST)>0
                        candidate &= np.isin(target,(1,2,3))
                        changed += int(np.count_nonzero(candidate & (target!=1)))
                        target[candidate]=1
        finally:
            del probabilities,coverage
    loaded.source_metadata.update(
        road_head_fusion="pan_connected_roads_v9_hann_overlap50_hysteresis",
        road_analysis_gsd_m=effective_gsd,
        road_analysis_shape=[height,width],
        road_continuity=asdict(config),
        road_head_filter_audit=audit,
        road_overlap_stride=patch_size//2,
        road_size_dependent_downsampling=False,
        road_consensus_policy="weighted_probability_context_not_mandatory_two_views",
        road_geometry_verifier=verifier_spec,
        road_image_guard="two_sided_intensity_corridor_and_learned_road_support",
    )
    prediction.learned_road_head_applied=True
    prediction.learned_road_head_pixels_added=changed
    prediction.topology_postprocessing_applied=True
    service._refresh_prediction_percentages(prediction)
    return changed
