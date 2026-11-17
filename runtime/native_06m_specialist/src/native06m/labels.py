"""Rasterize reviewed class polygons; overlapping classes stay unknown, not road-biased."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window,bounds,transform
from shapely.geometry import box

from .predict import atomic_raster


def rasterize_polygons(image_path,vector_path,destination,class_field="class_id"):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    polygons=gpd.read_file(vector_path)
    if polygons.crs is None or class_field not in polygons:
        raise ValueError("Reviewed polygons need a CRS and explicit class_id field")
    if polygons.empty or polygons.geometry.isna().any() or polygons.geometry.is_empty.any():
        raise ValueError("No usable class polygons")
    if not polygons.geom_type.isin(["Polygon","MultiPolygon"]).all():
        raise ValueError("Use road-surface polygons, not centreline buffers with invented widths")
    if not polygons.geometry.is_valid.all():
        raise ValueError("Repair invalid annotation geometries before rasterizing")
    values=polygons[class_field]
    if not values.isin([0,1,2,3,4]).all():
        raise ValueError("class_id must contain integers 0..4")
    counts=np.zeros(5,np.int64)
    ambiguous=0
    with rasterio.open(image_path) as source:
        if not source.crs:
            raise ValueError("Source image has no CRS")
        polygons=polygons.to_crs(source.crs)
        polygons=polygons[polygons.intersects(box(*source.bounds))]
        if polygons.empty:
            raise ValueError("No annotation polygons overlap this TIFF")
        profile=source.profile.copy()
        profile.update(count=1,dtype="uint8",nodata=255,compress="lzw",tiled=True,blockxsize=256,blockysize=256)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with atomic_raster(destination,profile) as output:
            for y in range(0,source.height,1024):
                for x in range(0,source.width,1024):
                    window=Window(x,y,min(1024,source.width-x),min(1024,source.height-y))
                    shape=(int(window.height),int(window.width))
                    candidates=polygons.iloc[polygons.sindex.query(box(*bounds(window,source.transform)),predicate="intersects")]
                    labels=np.full(shape,255,np.uint8)
                    coverage=np.zeros(shape,np.uint8)
                    for class_id in range(5):
                        geometry=candidates[candidates[class_field]==class_id].geometry
                        if geometry.empty:
                            continue
                        mask=rasterize(((g,1) for g in geometry),out_shape=shape,transform=transform(window,source.transform),fill=0).astype(bool)
                        coverage+=mask
                        labels[mask]=class_id
                    valid=source.read_masks(1,window=window)>0
                    ambiguous+=int(((coverage>1)&valid).sum())
                    labels[(coverage!=1)|(~valid)]=255
                    counts+=np.bincount(labels[labels<5],minlength=5)
                    output.write(labels,1,window=window)
            output.update_tags(label_provenance="rasterized supplied polygons; human review status must be attested separately",
                               overlaps="Conflicting classes ignored as 255",unannotated="255",water="0")
    report={"labels":str(destination.resolve()),"vector_source":str(Path(vector_path).resolve()),
            "class_pixel_counts":counts.tolist(),"conflicting_pixels_ignored":ambiguous,
            "provenance":"Not automatically certified as human-reviewed"}
    destination.with_suffix(".json").write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",type=Path)
    parser.add_argument("polygons",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--class-field",default="class_id")
    args=parser.parse_args()
    print(json.dumps(rasterize_polygons(args.image,args.polygons,args.output,args.class_field),indent=2))


if __name__=="__main__":
    main()
