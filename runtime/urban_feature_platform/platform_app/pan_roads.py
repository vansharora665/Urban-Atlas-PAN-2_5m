"""Probability-supported, metric-scale road continuity without land-cover quotas."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class RoadContinuity:
    support_probability: float = 0.25
    seed_probability: float = 0.65
    gap_probability: float = 0.15
    maximum_gap_m: float = 7.5
    minimum_length_m: float = 40.0
    maximum_width_m: float = 35.0


def image_corridors(image, probability, gsd_m):
    """Find long two-sided intensity strips, not arbitrary ground boundaries.

    Line segments propose local orientations only. Both flanks must contrast
    with the central strip, and the learned road expert must support it.
    """
    smooth=cv2.GaussianBlur(np.asarray(image,np.float32),(3,3),.6)
    detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected=detector.detect(np.clip(smooth*255,0,255).astype(np.uint8))[0]
    corridor=np.zeros(image.shape,np.uint8)
    if detected is None:
        return corridor.astype(bool)
    for segment in detected[:,0]:
        x0,y0,x1,y1=map(float,segment)
        dx,dy=x1-x0,y1-y0
        length=float(np.hypot(dx,dy))
        if length*gsd_m<65:
            continue
        nx,ny=-dy/length,dx/length
        n=max(8,int(length))
        xx,yy=np.linspace(x0,x1,n),np.linspace(y0,y1,n)
        best=None
        for offset in (-7,-5,-3,-1,1,3,5,7):
            mx=(xx+offset*nx).astype(np.float32)[None]
            my=(yy+offset*ny).astype(np.float32)[None]
            center=cv2.remap(smooth,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)[0]
            p=cv2.remap(probability,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)[0]
            if p.mean()<.20 or np.quantile(p,.65)<.30:
                continue
            for radius in (2,4,6):
                flank=radius+2
                left=cv2.remap(smooth,mx+flank*nx,my+flank*ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)[0]
                right=cv2.remap(smooth,mx-flank*nx,my-flank*ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)[0]
                a,b=center-left,center-right
                for sign in (-1,1):
                    local=np.minimum(sign*a,sign*b)
                    contrast=float(local.mean())
                    continuity=float((local>.015).mean())
                    if contrast<.035 or continuity<.60:
                        continue
                    score=contrast*continuity
                    if best is None or score>best[0]:
                        best=(score,offset,radius)
        if best is not None:
            _,offset,radius=best
            a=(int(round(x0+offset*nx)),int(round(y0+offset*ny)))
            b=(int(round(x1+offset*nx)),int(round(y1+offset*ny)))
            cv2.line(corridor,a,b,1,thickness=2*radius+1)
    return corridor.astype(bool)


def connected_roads(probability, base_labels, source_valid, gsd_m=2.5,
                    config=RoadContinuity(), audit=None,image=None):
    """Grow confident road seeds along thin model-supported land corridors.

    Semantic Unknown is not source validity: neither is eligible for conversion.
    Directional closing can add only short, non-protected, model-supported gaps.
    """
    if not (probability.shape == base_labels.shape == source_valid.shape):
        raise ValueError("Road probabilities, land labels and validity must share a grid")
    if gsd_m <= 0 or not np.isfinite(gsd_m):
        raise ValueError("Road grid resolution must be positive")
    allowed = source_valid & np.isin(base_labels, (1, 2, 3))
    if image is not None:
        corridor=image_corridors(image,probability,gsd_m)
        allowed &= corridor
        # Odds boost only within image-supported, narrow two-sided corridors.
        probability=3*probability/(1+2*probability+1e-7)
    support = allowed & np.isfinite(probability) & (probability >= config.support_probability)
    seeds = support & (probability >= config.seed_probability)
    if not seeds.any():
        return np.zeros(probability.shape, bool)
    # Width-limited hysteresis cannot flood broad low-confidence ground patches.
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    radius = max(1,int(np.ceil(config.maximum_width_m / (2*gsd_m))))
    broad = distance * 2 * gsd_m > config.maximum_width_m
    if broad.any():
        broad = cv2.dilate(broad.astype(np.uint8),cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(bool)
        support &= ~broad
    grown = ndimage.binary_propagation(seeds & support, mask=support, structure=np.ones((3, 3)))

    gap_cells = max(0, int(np.floor(config.maximum_gap_m / gsd_m)))
    bridge = np.zeros_like(grown)
    if gap_cells:
        size = gap_cells + 2
        nearby = ndimage.distance_transform_edt(~grown) <= (gap_cells + 1) / 2
        for angle in range(0, 180, 15):
            kernel = np.zeros((2 * size + 1, 2 * size + 1), np.uint8)
            dx = int(round(size * np.cos(np.deg2rad(angle))))
            dy = int(round(size * np.sin(np.deg2rad(angle))))
            cv2.line(kernel, (size-dx, size-dy), (size+dx, size+dy), 1)
            closed = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
            added = closed & ~grown
            # Every added run must stay close to existing road pixels.
            bridge |= added & nearby
        bridge &= allowed & (probability >= config.gap_probability)
    candidate = grown | bridge
    count, components, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    accepted = np.zeros_like(candidate)
    accepted_components = 0
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        region = np.s_[y:y+height, x:x+width]
        component = components[region] == index
        extent = max(width, height) * gsd_m
        elongation = max(width, height) / max(1, min(width, height))
        fill = area / (width * height)
        if extent < config.minimum_length_m or not (elongation >= 2.2 or fill <= 0.40):
            continue
        if not np.any(seeds[region] & component):
            continue
        accepted[region] |= component
        accepted_components += 1
    if audit is not None:
        for key, value in {"support_cells_with_halos": support.sum(),
                           "seed_cells_with_halos": seeds.sum(),
                           "accepted_cells_with_halos": accepted.sum(),
                           "gap_cells_with_halos": (bridge & accepted).sum(),
                           "accepted_components_with_halos": accepted_components}.items():
            audit[key] = audit.get(key, 0) + int(value)
    return accepted & allowed
