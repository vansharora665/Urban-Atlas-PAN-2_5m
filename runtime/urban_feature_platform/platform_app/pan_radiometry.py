"""Inherit radiometry only from pixel-verified local parent scenes, never labels."""

from pathlib import Path

import numpy as np
import rasterio

from .cloud_mask import detect_cloud_mask
from .image_utils import _read_raster_grayscale,native_radiometric_sample,robust_normalize,to_uint8

_PROFILES={}


def inherit_parent_normalization(loaded,parents):
    if loaded.source_metadata.get("band_count")!=1:
        return False
    with rasterio.open(loaded.source_path) as uploaded:
        for path in parents:
            path=Path(path)
            if not path.is_file():continue
            with rasterio.open(path) as parent:
                if parent.count!=1 or parent.crs!=uploaded.crs:
                    continue
                a,b=parent.transform,uploaded.transform
                if not np.allclose([a.a,a.b,a.d,a.e],[b.a,b.b,b.d,b.e],rtol=1e-9,atol=1e-8):
                    continue
                offset=(~a)*(b.c,b.f)
                x,y=map(lambda value:int(round(value)),offset)
                if max(abs(offset[0]-x),abs(offset[1]-y))>1e-4 or x<0 or y<0:
                    continue
                if x+uploaded.width>parent.width or y+uploaded.height>parent.height:
                    continue
                matched=True
                evidence=0
                size=min(16,uploaded.width,uploaded.height)
                for row in np.unique(np.linspace(0,max(0,uploaded.height-size),4).astype(int)):
                    for col in np.unique(np.linspace(0,max(0,uploaded.width-size),4).astype(int)):
                        child=uploaded.read(1,window=rasterio.windows.Window(int(col),int(row),size,size))
                        reference=parent.read(1,window=rasterio.windows.Window(x+int(col),y+int(row),size,size))
                        if not np.array_equal(child,reference,equal_nan=True):
                            matched=False
                            break
                        valid=np.isfinite(child)
                        if uploaded.nodata is not None:valid &= child!=uploaded.nodata
                        evidence+=int(valid.sum())
                    if not matched:break
                if not matched or evidence<min(512,uploaded.width*uploaded.height):
                    continue
                stat=path.stat()
                key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
                if key not in _PROFILES:
                    sample=native_radiometric_sample(parent)
                    _,stats=robust_normalize(sample,np.ones(sample.shape,bool))
                    _PROFILES[key]=stats
                stats=_PROFILES[key]
            previous=[loaded.normalization_low,loaded.normalization_high]
            loaded.normalization_low,loaded.normalization_high=stats["low"],stats["high"]
            metadata=loaded.source_metadata
            metadata.update(normalization_method="pixel_verified_parent_scene_original_pixels_v1",
                            normalization_reference=path.name,normalization_reference_verified_pixels=evidence,
                            previous_upload_normalization=previous,normalization_low=stats["low"],normalization_high=stats["high"],
                            normalization_percentiles=[stats["low_percentile"],stats["high_percentile"]],
                            radiometric_tail_detected=stats["radiometric_tail_detected"],cloud_threshold=stats["cloud_threshold"])
            h,w=loaded.preview_rgb.shape[:2]
            raw,valid=_read_raster_grayscale(uploaded,out_shape=(h,w))
            normalized=np.clip((raw-stats["low"])/max(stats["high"]-stats["low"],1e-6),0,1)
            normalized[~valid]=0
            loaded.preview_rgb=np.repeat(to_uint8(normalized)[:,:,None],3,axis=2)
            if loaded.cloud_mask is not None:
                loaded.cloud_mask=detect_cloud_mask(raw,valid,source_width=loaded.width,source_height=loaded.height,
                                                   pixel_size_m=max(abs(b.a),abs(b.e)) if metadata.get("pixel_size_units")=="metre" else None,
                                                   normalization_stats=stats)
                metadata["cloud_screening"]=loaded.cloud_mask.metadata
                metadata["cloud_masking_applied"]=loaded.cloud_mask.has_clouds
            if loaded.grayscale is not None:
                raw,valid=_read_raster_grayscale(uploaded)
                loaded.grayscale=np.clip((raw-stats["low"])/max(stats["high"]-stats["low"],1e-6),0,1)
                loaded.grayscale[~valid]=0
                if loaded.cloud_mask is not None and loaded.cloud_mask.has_clouds:
                    excluded=loaded.cloud_mask.window(0,0,loaded.width,loaded.height)>0
                    valid &= ~excluded
                    loaded.grayscale[excluded]=.5
                loaded.valid_mask=valid
            return True
    return False
