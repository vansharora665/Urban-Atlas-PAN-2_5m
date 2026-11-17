from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .data import inspect_manifest, load_manifest, read_json, training_preflight


def write_inventory(manifest_path,config_path,output):
    manifest = load_manifest(manifest_path)
    config,_ = read_json(config_path)
    report = training_preflight(manifest,config)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"dataset_readiness.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({key:value for key,value in report.items() if key!="rasters"},indent=2))
    return report


def export_review_tiles(manifest_path,output,count=16,size=512):
    manifest = load_manifest(manifest_path)
    report = inspect_manifest(manifest)
    if report["errors"]:
        raise ValueError("\n".join(report["errors"]))
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    tiles = []
    for scene in manifest["scenes"]:
        with rasterio.open(scene["images"][0]) as source:
            side = int(np.ceil(np.sqrt(count)))
            anchors = [(int(x),int(y)) for y in np.linspace(0,max(0,source.height-size),side+2)[1:-1]
                       for x in np.linspace(0,max(0,source.width-size),side+2)[1:-1]][:count]
            for index,(x,y) in enumerate(anchors):
                tile_id = f"{scene['scene_id']}_review_{index+1:02d}"
                window = Window(x,y,min(size,source.width-x),min(size,source.height-y))
                profile = source.profile.copy()
                profile.update(width=int(window.width),height=int(window.height),
                               transform=source.window_transform(window),compress="lzw")
                paths = []
                for variant,path in enumerate(scene["images"]):
                    image_path = output/f"{tile_id}_variant{variant+1}.tif"
                    with rasterio.open(path) as src, rasterio.open(image_path,"w",**profile) as dst:
                        dst.write(src.read(1,window=window),1)
                    paths.append(str(image_path.resolve()))
                template = output/f"{tile_id}_UNLABELLED_TEMPLATE.tif"
                profile.update(count=1,dtype="uint8",nodata=255)
                with rasterio.open(template,"w",**profile) as dst:
                    dst.write(np.full((int(window.height),int(window.width)),255,np.uint8),1)
                    dst.update_tags(label_status="EMPTY_TEMPLATE_NOT_GROUND_TRUTH",
                                    classes="0=water;1=roads;2=urban;3=barren;4=vegetation;255=unreviewed/cloud/shadow")
                tiles.append({"scene_id":tile_id,"geographic_group":scene["geographic_group"],
                              "split":"unassigned","images":paths,"labels":str(template.resolve()),
                              "label_provenance":"unreviewed_empty_template","quality_mask":None,
                              "source_pixel_window":[x,y,int(window.width),int(window.height)]})
    (output/"review_tiles_manifest.json").write_text(json.dumps({**manifest,"scenes":tiles},indent=2))
    (output/"NOT_LABELLED_YET.txt").write_text(
        "These are native-resolution image crops and EMPTY label templates, not training ground truth.\n"
        "All template pixels are 255 (unreviewed). Keep uncertain pixels, clouds and opaque shadows at 255.\n"
        "Assign geographic splits to whole source areas, never randomly to these neighbouring crops.\n"
        "Use independently reviewed pixel masks or polygons with class_id 0..4 before training.\n")
    return {"review_tiles":len(tiles),"directory":str(output.resolve()),"labels_created":False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,default=Path("config/scenes.json"))
    parser.add_argument("--config",type=Path,default=Path("config/training.json"))
    parser.add_argument("--output",type=Path,default=Path("audit"))
    parser.add_argument("--review-tiles",action="store_true")
    args = parser.parse_args()
    if args.review_tiles:
        print(json.dumps(export_review_tiles(args.manifest,args.output),indent=2))
    else:
        write_inventory(args.manifest,args.config,args.output)


if __name__=="__main__":
    main()
