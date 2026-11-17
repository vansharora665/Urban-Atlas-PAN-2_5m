"""Native 0.6 m probability hysteresis, metric geometry and compact-spot rejection."""

from dataclasses import asdict,dataclass

import cv2
import numpy as np
from scipy import ndimage

from .native_refinement import bridge_short_road_gaps


@dataclass(frozen=True)
class NativeRoadRecipe:
    support_probability: float=.18
    seed_probability: float=.45
    road_to_other_ratio: float=.50
    minimum_length_m: float=24.
    maximum_width_m: float=18.
    maximum_gap_m: float=3.6
    bare_minimum_road_probability: float=0.
    vegetation_minimum_road_probability: float=1.01


def reject_isolated_native_dots(labels,evidence,valid,maximum_extent_m=1.8,gsd_m=.6):
    """A high softmax score cannot make an isolated sub-road-sized object real."""
    labels, evidence = np.asarray(labels), np.asarray(evidence)
    if evidence.shape != (4, *labels.shape) or np.shape(valid) != labels.shape:
        raise ValueError("Dot rejection requires four aligned learned evidence bands")
    if not np.isfinite(gsd_m) or gsd_m <= 0 or not np.isfinite(maximum_extent_m) or maximum_extent_m <= 0:
        raise ValueError("Dot rejection requires positive metric sizes")
    if not np.isin(evidence[3], [0, 2, 3, 4]).all():
        raise ValueError("Invalid learned non-road alternative class")
    valid=np.asarray(valid,bool)&(labels!=255)
    roads=valid&(labels==1)
    count,components,stats,_=cv2.connectedComponentsWithStats(roads.astype(np.uint8),8)
    candidate=np.zeros(count,bool);established=np.zeros(count,bool)
    if count>1:
        extent=np.maximum(stats[:,cv2.CC_STAT_WIDTH],stats[:,cv2.CC_STAT_HEIGHT])*gsd_m
        candidate=(extent<=maximum_extent_m+1e-6)
        established=(extent>=6.)&(stats[:,cv2.CC_STAT_AREA]>=12)
    candidate[0]=False;established[0]=False
    edges=np.unique(np.concatenate((components[0],components[-1],components[:,0],components[:,-1])))
    candidate[edges]=False
    nearby=cv2.dilate(established[components].astype(np.uint8),cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,(2*int(np.ceil(1.2/gsd_m))+1,)*2)).astype(bool)
    uncertain=cv2.dilate((~valid).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    protected=np.unique(components[nearby|uncertain]);candidate[protected]=False
    rejected=candidate[components]&roads
    output=labels.copy()
    confidence=np.maximum(evidence[1],evidence[2]).astype(np.float32)/255
    supported=rejected&(confidence>=.2)
    output[supported]=evidence[3][supported]
    output[rejected&~supported]=255
    return output,{"isolated_dot_pixels_removed":int(rejected.sum()),"maximum_dot_extent_m":maximum_extent_m}


def refine_connected_native_roads(labels,evidence,valid,recipe=NativeRoadRecipe(),gsd_m=.6,return_masks=False):
    """Never promote water, vegetation or Unknown using geometry alone.

    A separately trained road head may change semantic predictions; this
    post-pass grows only through model-supported built-up/bare-ground pixels.
    Rejected dots return to the learned non-road alternative, not a default class.
    """
    if isinstance(recipe,dict):recipe=NativeRoadRecipe(**recipe)
    labels,evidence=np.asarray(labels),np.asarray(evidence)
    if evidence.shape!=(4,*labels.shape) or np.shape(valid)!=labels.shape:
        raise ValueError("Native road continuity needs four aligned model-evidence bands")
    if gsd_m<=0 or not np.isfinite(gsd_m):
        raise ValueError("Native road continuity needs a positive metric resolution")
    valid=np.asarray(valid,bool)&(labels!=255)
    road=evidence[0].astype(np.float32)/255
    other=np.maximum(evidence[1],evidence[2]).astype(np.float32)/255
    allowed=valid&(np.isin(labels,[1,2,3])|((labels==4)&(road>=recipe.vegetation_minimum_road_probability)))
    support=allowed&(road>=recipe.support_probability)&(road>=recipe.road_to_other_ratio*other)
    support&=(evidence[3]!=3)|(road>=recipe.bare_minimum_road_probability)
    seeds=support&((road>=recipe.seed_probability)|(labels==1))
    distance=cv2.distanceTransform(np.pad(support.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
    radius=max(1,int(np.ceil(recipe.maximum_width_m/(2*gsd_m))))
    bulk=2*distance*gsd_m>recipe.maximum_width_m
    if bulk.any():
        bulk=cv2.dilate(bulk.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                         (2*radius+1,2*radius+1))).astype(bool)
        support&=~bulk
    grown=ndimage.binary_propagation(seeds&support,mask=support,structure=np.ones((3,3)))
    count,components,stats,_=cv2.connectedComponentsWithStats(grown.astype(np.uint8),8)
    accepted=np.zeros_like(grown)
    for index in range(1,count):
        x,y,width,height,area=map(int,stats[index])
        if max(width,height)*gsd_m<recipe.minimum_length_m:
            continue
        elongation=max(width,height)/max(1,min(width,height))
        fill=area/(width*height)
        if elongation<2.2 and fill>.35:
            continue
        region=np.s_[y:y+height,x:x+width]
        accepted[region]|=components[region]==index
    output=labels.copy()
    rejected=valid&(labels==1)&~accepted
    alternative=evidence[3]
    if not np.isin(alternative,[0,2,3,4]).all():
        raise ValueError("Invalid learned non-road class IDs")
    output[rejected]=alternative[rejected]
    output[accepted]=1
    before_gaps=output.copy()
    output,gaps=bridge_short_road_gaps(output,evidence,valid,gsd_m=gsd_m,
                                     maximum_gap_m=recipe.maximum_gap_m)
    details={"method":"native_connected_roads_v3","recipe":asdict(recipe),
                   "road_speck_pixels_removed":int(rejected.sum()),
                   "road_recovered_pixels":int(np.count_nonzero(accepted&(labels!=1))),
                   "road_gap_pixels_added":gaps["road_gap_pixels_added"],
                   "geometry_never_relabels_water_unknown":True,
                   "vegetation_recovery_requires_road_probability":recipe.vegetation_minimum_road_probability}
    if return_masks:
        details.update(_recovered_mask=accepted&(labels!=1),_removed_mask=rejected,
                       _gap_mask=(output==1)&(before_gaps!=1))
    return output,details
