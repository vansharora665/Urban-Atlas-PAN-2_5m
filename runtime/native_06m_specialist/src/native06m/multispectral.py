"""Prepare explicit four-band scene records and streamed footprint masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

from .data import CLASSES, inspect_manifest


PRODUCTS = [
    ("267372781", "product_267372781_v2", "pune_west_20260402", "train"),
    ("2673727101", "product_2673727101", "pune_central_20250216", "validation"),
    ("267372791", "product_267372791_v2", "chennai_20260106", "test"),
]


def prepare(root):
    root = Path(root).resolve()
    manifest = {"schema_version":2,"input_channels":4,"band_order":["blue","green","red","nir"],
                "target_gsd_m":.6,"gsd_tolerance_m":.03,
                "classes":{str(k):v for k,v in CLASSES.items()},"ignore_index":255,
                "stage":"self_supervised_pretraining_not_semantic_training", "scenes":[]}
    for product_id, directory, group, split in PRODUCTS:
        audit = root/"audit"/directory
        report = json.loads((audit/"product_inventory.json").read_text())
        quality_path = audit/"footprint_exclusion_mask.tif"
        footprint = gpd.read_file(report["working_footprint"])
        with rasterio.open(report["virtual_stack"]) as src:
            if footprint.crs!=src.crs:
                footprint = footprint.to_crs(src.crs)
            if not quality_path.exists():
                profile = src.profile.copy()
                profile.update(driver="GTiff", count=1, dtype="uint8", nodata=None,
                               compress="lzw", tiled=True, blockxsize=256, blockysize=256)
                with rasterio.open(quality_path,"w",**profile) as out:
                    for y in range(0,src.height,256):
                        window = Window(0,y,src.width,min(256,src.height-y))
                        inside = geometry_mask(footprint.geometry,
                            out_shape=(int(window.height),int(window.width)),
                            transform=src.window_transform(window),invert=True)
                        out.write(np.where(inside,0,255).astype(np.uint8),1,window=window)
                    out.update_tags(scope="Image footprint only; NOT a cloud detector or class label")
        manifest["scenes"].append({"scene_id":product_id,"geographic_group":group,"split":split,
            "images":[report["virtual_stack"]],"quality_mask":str(quality_path),"labels":None,
            "label_provenance":None,"sensor":"CARTOSAT-2E MX MERGED",
            "acquisition_date":report["acquisition_date"],"product_inventory":str(audit/"product_inventory.json"),
            "notes":"Four co-registered spectral channels. Footprint-only quality mask; no cloud screening. Labels pending review."})
    report = inspect_manifest(manifest)
    if report["errors"]:
        raise ValueError("\n".join(report["errors"]))
    destination = root/"config/multispectral_scenes.json"
    destination.write_text(json.dumps(manifest,indent=2)+"\n")
    return {"manifest":str(destination),"independent_source_areas":3,
            "supervised_labels_available":False,"self_supervised_training_area":"pune_west_20260402",
            "reconstruction_validation_area":"pune_central_20250216",
            "excluded_from_pretraining":"chennai_20260106"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    args=parser.parse_args()
    print(json.dumps(prepare(args.root),indent=2))
