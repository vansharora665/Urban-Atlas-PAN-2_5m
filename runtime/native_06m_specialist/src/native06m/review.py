"""Native four-band review tiles and a browser-based manual annotation workspace."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

from .data import CLASSES, file_digest, load_manifest, normalization, read_native


def rle_encode(array):
    flat=np.asarray(array,dtype=np.uint8).ravel()
    cuts=np.r_[0,np.flatnonzero(flat[1:]!=flat[:-1])+1,len(flat)]
    return [[int(flat[a]),int(b-a)] for a,b in zip(cuts[:-1],cuts[1:])]


def rle_decode(runs, shape, allowed):
    size=int(np.prod(shape));total=0
    out=np.empty(size,np.uint8)
    for value,count in runs:
        if type(value) is not int or value not in allowed or type(count) is not int or count<1 or total+count>size:
            raise ValueError("Invalid annotation run")
        out[total:total+count]=value;total+=count
    if total!=size:
        raise ValueError("Annotation size does not match the native tile")
    return out.reshape(shape)


def write_review_html(output, review):
    output=Path(output)
    tiles=[]
    for tile in review["scenes"]:
        folder=Path(tile["images"][0]).parent
        image_url=lambda p:"data:image/png;base64,"+base64.b64encode(p.read_bytes()).decode()
        entry={"id":tile["tile_id"],"area":tile["geographic_group"],"split":tile["split"],
            "width":tile["width"],"height":tile["height"],"valid_rle":tile["valid_rle"],
            "review_notes":tile.get("draft_review_notes",""),
            "rgb":image_url(folder/"true_colour.png"),"fcc":image_url(folder/"false_colour_nir.png")}
        if tile.get("draft_label_path"):
            with rasterio.open(tile["draft_label_path"]) as draft,rasterio.open(tile["images"][0]) as image:
                if (draft.shape,draft.crs,draft.transform)!=(image.shape,image.crs,image.transform) or draft.count!=1 or draft.nodata!=255:
                    raise ValueError("Draft mask must match the exact native image grid with unknown=255")
                mask=draft.read(1)
            if not np.isin(mask,[0,1,2,3,4,255]).all():
                raise ValueError("Unsupported draft label values")
            valid=rle_decode(tile["valid_rle"],mask.shape,{0,1}).astype(bool)
            if np.any((mask<5)&~valid):
                raise ValueError("Draft labels extend beyond the valid source footprint")
            entry.update(initial_mask_rle=rle_encode(mask),initial_draft_sha256=file_digest(tile["draft_label_path"]))
        tiles.append(entry)
    ui={"project_id":review["annotation_project_id"],"classes":review["classes"],"tiles":tiles,
        "review_mode":review.get("review_mode","manual_from_blank")}
    template=Path(__file__).resolve().parents[2]/"templates/annotation_review.html"
    html=template.read_text().replace("__PROJECT_JSON__",json.dumps(ui).replace("</","<\\/"))
    (output/"START_REVIEW.html").write_text(html)


def prepare_review(manifest_path, output, per_scene=4, size=512):
    manifest=load_manifest(manifest_path);output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    tiles=[]
    for scene in manifest["scenes"]:
        path=scene["images"][0];limits=normalization(path,scene["quality_mask"])
        candidates=[]
        with rasterio.open(path) as src,rasterio.open(scene["quality_mask"]) as qa:
            for y in np.linspace(0,src.height-size,7)[1:-1].astype(int):
                for x in np.linspace(0,src.width-size,7)[1:-1].astype(int):
                    image,valid=read_native(src,int(x),int(y),size,limits)
                    valid &= qa.read(1,window=Window(int(x),int(y),size,size))==0
                    if valid.mean()<.98:
                        continue
                    feature=np.r_[image[valid].mean(0),image[valid].std(0)]
                    candidates.append((int(x),int(y),feature))
            if len(candidates)<per_scene:
                raise ValueError(f"Insufficient usable review candidates for {scene['scene_id']}")
            features=np.array([c[2] for c in candidates]);scaled=(features-features.mean(0))/np.maximum(features.std(0),.025)
            chosen=[int(features[:,4:].sum(1).argmax())]
            while len(chosen)<per_scene:
                distances=((scaled[:,None]-scaled[chosen][None])**2).sum(2).min(1)
                distances[chosen]=-1
                chosen.append(int(distances.argmax()))
            for number,index in enumerate(chosen,1):
                x,y,_=candidates[index];window=Window(x,y,size,size)
                tile_id=f"{scene['scene_id']}_review_{number:02d}"
                folder=output/tile_id;folder.mkdir()
                profile=src.profile.copy();profile.update(driver="GTiff",height=size,width=size,
                    transform=src.window_transform(window),compress="lzw",tiled=True,blockxsize=256,blockysize=256)
                image_path=folder/"native_four_band.tif"
                with rasterio.open(image_path,"w",**profile) as out:
                    out.write(src.read(window=window))
                    for band,name in enumerate(("Blue","Green","Red","Near infrared"),1):
                        out.set_band_description(band,name)
                image,valid=read_native(src,x,y,size,limits)
                valid &= qa.read(1,window=window)==0
                image[~valid]=0
                rgb=(image[:,:,[2,1,0]]*255).astype(np.uint8)
                fcc=(image[:,:,[3,2,1]]*255).astype(np.uint8)
                Image.fromarray(rgb).save(folder/"true_colour.png")
                Image.fromarray(fcc).save(folder/"false_colour_nir.png")
                profile.update(count=1,dtype="uint8",nodata=255)
                template=folder/"UNLABELLED_TEMPLATE.tif"
                with rasterio.open(template,"w",**profile) as out:
                    out.write(np.full((size,size),255,np.uint8),1)
                    out.update_tags(status="EMPTY_NOT_GROUND_TRUTH")
                record={"tile_id":tile_id,"scene_id":tile_id,"source_scene_id":scene["scene_id"],
                    "geographic_group":scene["geographic_group"],"split":scene["split"],
                    "images":[str(image_path)],"labels":str(template),
                    "label_provenance":"unreviewed_empty_template","quality_mask":None,
                    "source":path,"source_pixel_window":[x,y,size,size],
                    "width":size,"height":size,"valid_rle":rle_encode(valid.astype(np.uint8))}
                tiles.append(record)
    project_id=output.name
    review={**manifest,"annotation_project_id":project_id,"scenes":tiles,
            "review_status":"All label templates empty; no automatic pseudo-labels",
            "tile_selection":"Spectral/texture diversity only; not a class assignment"}
    (output/"review_manifest.json").write_text(json.dumps(review,indent=2))
    write_review_html(output,review)
    (output/"CLASS_GUIDE.txt").write_text(
        "MANUAL FIVE-CLASS REVIEW: all templates initially contain 255, not training labels.\n"
        "0 Water: visible open water, rivers, canals and ponds. Do not infer water from darkness alone.\n"
        "1 Roads: visible road surface, including bridges; not rails, drains, rivers or building shadows.\n"
        "2 Urban built-up: roofs/buildings and hard built structures; exclude visible road surfaces.\n"
        "3 Barren/empty land: visible bare soil, exposed ground and construction earth, not paved roads.\n"
        "4 Vegetation: visible trees, shrubs, grass and crops; use NIR view as an aid, not automatic truth.\n"
        "255 Unknown: unreviewed, ambiguous, clouds, opaque shadows and padding. Leave these unlabelled.\n"
        "Annotate a few clear examples of EVERY class, including dark roads beside water and bright roofs.\n"
        "Do not label cloudy ground from imagination. These are visual reference labels, not surveyed truth.\n"
        "Open START_REVIEW.html, select a class, paint or draw polygons, then confirm reviewed tiles and export JSON.\n"
        "Georeferencing stays in native_four_band.tif; the JSON is imported with python3 -m native06m.review import.\n"
        "Keep geographic splits fixed; adjacent tiles and spectral bands are NOT independent datasets.\n")
    return {"review_directory":str(output),"tiles":len(tiles),"classes":CLASSES,
            "labels_created":False,"html":str(output/"START_REVIEW.html")}


def import_review(manifest_path, annotations_path, output):
    manifest=load_manifest(manifest_path);annotations=json.loads(Path(annotations_path).read_text())
    if annotations.get("project_id")!=manifest["annotation_project_id"] or not annotations.get("reviewed_by","").strip():
        raise ValueError("Annotation project and reviewer identity must match")
    by_id={s["tile_id"]:s for s in manifest["scenes"]};prepared=[];seen=set()
    for item in annotations["tiles"]:
        if item["id"] not in by_id or item["id"] in seen or item.get("reviewed") is not True:
            raise ValueError("Unknown, duplicate or unreviewed tile")
        seen.add(item["id"]);scene=by_id[item["id"]]
        labels=rle_decode(item["mask_rle"],(scene["height"],scene["width"]),{0,1,2,3,4,255})
        valid=rle_decode(scene["valid_rle"],labels.shape,{0,1}).astype(bool)
        labels[~valid]=255
        if scene.get("draft_label_path"):
            if annotations.get("review_mode")!="assistant_draft_trial" or item.get("draft_review_acknowledged") is not True:
                raise ValueError("Assistant drafts require explicit human-review acknowledgement")
            if item.get("initial_draft_sha256")!=file_digest(scene["draft_label_path"]):
                raise ValueError("Reviewed export does not match this version of the initial draft")
        if not (labels<5).any():
            raise ValueError("Reviewed tiles must contain at least some annotated class pixels")
        prepared.append((scene,labels))
    if not prepared:
        raise ValueError("No reviewed tiles in export")
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    scenes=[]
    for scene,labels in prepared:
        label_path=output/f"{scene['tile_id']}_reviewed_labels.tif"
        with rasterio.open(scene["images"][0]) as src:
            profile=src.profile.copy();profile.update(count=1,dtype="uint8",nodata=255,compress="lzw")
        with rasterio.open(label_path,"w",**profile) as out:
            out.write(labels,1);out.update_tags(label_provenance="human_reviewed_visual_annotation",reviewer=annotations["reviewed_by"])
        record={**scene,"labels":str(label_path),"label_provenance":"human_reviewed",
                "reviewed_by":annotations["reviewed_by"],"reviewed_at":annotations.get("reviewed_at")}
        if scene.get("draft_label_path"):
            with rasterio.open(scene["draft_label_path"]) as draft:original=draft.read(1)
            record.update(quality_status="awaiting_visual_qa",label_creation_method="assistant_visual_draft_reviewed_by_user",
                          changed_pixels_from_draft=int((original!=labels).sum()))
        scenes.append(record)
    final={**manifest,"scenes":scenes,"review_status":"User-attested visual annotation, not surveyed or independently verified ground truth"}
    destination=output/"reviewed_tiles_manifest.json"
    destination.write_text(json.dumps(final,indent=2))
    return {"manifest":str(destination),"reviewed_tiles":len(scenes),"training_started":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    make=sub.add_parser("prepare");make.add_argument("--manifest",type=Path,default=Path("config/multispectral_scenes.json"))
    make.add_argument("--output",type=Path,required=True)
    read=sub.add_parser("import");read.add_argument("--manifest",type=Path,required=True)
    read.add_argument("--annotations",type=Path,required=True);read.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=prepare_review(args.manifest,args.output) if args.command=="prepare" else import_review(args.manifest,args.annotations,args.output)
    print(json.dumps(result,indent=2))
