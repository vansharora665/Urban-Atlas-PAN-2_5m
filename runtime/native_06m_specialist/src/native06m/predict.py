from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import uuid

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
import torch
from shapely.geometry import shape, MultiPolygon
from shapely.validation import make_valid

from .data import CLASSES, normalization, read_native, starts, valid_native_pixels
from .model import NativeAttentionUNetPP, choose_device


@contextmanager
def atomic_raster(destination,profile):
    temporary=destination.with_name(f".{destination.stem}.{uuid.uuid4().hex[:8]}.partial.tif")
    try:
        with rasterio.open(temporary,"w",**profile) as output:
            yield output
        # Same-directory hard link publishes a complete file without overwriting another run.
        os.link(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)


def native_stream_prediction(model,source_path,destination,device,patch=512,stride=384,
                             batch_size=2,quality_path=None,normalization_limits=None,progress=None,evidence_path=None,
                             posterior_callback=None):
    if patch%32 or not 0<stride<=patch or batch_size<1:
        raise ValueError("Patch must be divisible by 32; stride must cover the grid; batch size must be positive")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Will not overwrite {destination}")
    limits = normalization(source_path,quality_path) if normalization_limits is None else np.asarray(normalization_limits)
    model.eval()
    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(source_path))
        if not (source.crs and source.crs.is_projected and source.crs.linear_units=="metre"
                and all(abs(r-.6)<=.03 for r in source.res)
                and source.count==getattr(model,"in_channels",1)):
            raise ValueError("Input must match the model's declared channel count on a 0.6 m metric grid")
        if source.transform.b or source.transform.d or source.transform.a<=0 or source.transform.e>=0:
            raise ValueError("Reproject to a north-up grid explicitly before inference")
        limits = np.asarray(limits).reshape(source.count,2)
        if not np.isfinite(limits).all() or np.any(limits[:,1]<=limits[:,0]):
            raise ValueError("Invalid inference normalization limits")
        if source.count==4:
            descriptions=[str(x or "").lower().replace(" ","") for x in source.descriptions]
            if descriptions not in (["blue","green","red","nearinfrared"],["blue","green","red","nir"]):
                raise ValueError("Four-channel inputs need explicit B/G/R/NIR band descriptions; do not guess from filenames")
        quality = stack.enter_context(rasterio.open(quality_path)) if quality_path else None
        if quality and (quality.shape!=source.shape or quality.crs!=source.crs or quality.transform!=source.transform):
            raise ValueError("Quality mask and source must be exactly aligned")
        profile = source.profile.copy()
        profile.update(driver="GTiff",count=1,dtype="uint8",nodata=255,compress="lzw",tiled=True,blockxsize=256,blockysize=256)
        destination.parent.mkdir(parents=True,exist_ok=True)
        output = stack.enter_context(atomic_raster(destination,profile))
        evidence = None
        if evidence_path:
            evidence_path = Path(evidence_path)
            if evidence_path.exists():
                raise FileExistsError(f"Will not overwrite {evidence_path}")
            evidence_profile = {**profile,"count":4,"nodata":None}
            evidence = stack.enter_context(atomic_raster(evidence_path,evidence_profile))
        width,height = source.width,source.height
        buffer = np.zeros((5,patch,width),np.float32)
        coverage = np.zeros((patch,width),np.float32)
        one_d = np.maximum(np.hanning(patch),.05).astype(np.float32)
        blend = one_d[:,None]*one_d[None,:]
        counts = np.zeros(5,np.int64)
        y_positions,x_positions = starts(height,patch,stride),starts(width,patch,stride)
        with torch.inference_mode():
            for y_index,y in enumerate(y_positions):
                for start in range(0,len(x_positions),batch_size):
                    xs = x_positions[start:start+batch_size]
                    images=[]
                    for x in xs:
                        image,valid = read_native(source,x,y,patch,limits)
                        if quality:
                            qa = quality.read(1,window=Window(x,y,patch,patch),boundless=True,fill_value=255)
                            valid &= qa==0
                        image[~valid] = .5
                        images.append(image[None] if image.ndim==2 else image.transpose(2,0,1))
                    tensor = torch.from_numpy(np.stack(images)).to(device)
                    probabilities = torch.softmax(model(tensor)["semantic"],1).cpu().numpy()
                    for x,probability in zip(xs,probabilities):
                        h,w = min(patch,height-y),min(patch,width-x)
                        buffer[:,:h,x:x+w] += probability[:,:h,:w]*blend[None,:h,:w]
                        coverage[:h,x:x+w] += blend[:h,:w]
                next_y = y_positions[y_index+1] if y_index+1<len(y_positions) else height
                ready = next_y-y
                if not np.all(coverage[:ready]>0):
                    raise RuntimeError("Overlap reconstruction left uncovered native pixels")
                posterior = buffer[:,:ready]/coverage[None,:ready]
                label = posterior.argmax(0).astype(np.uint8)
                window = Window(0,y,width,ready)
                raw = source.read(window=window)
                valid = valid_native_pixels(source,raw,window)
                if quality:
                    valid &= quality.read(1,window=window)==0
                # Optional structured models consume blended evidence before pixel argmax.
                if posterior_callback is not None:
                    posterior_callback(y, posterior, valid)
                label[~valid]=255
                counts += np.bincount(label[valid],minlength=5)
                output.write(label,1,window=window)
                if evidence is not None:
                    road_evidence = np.stack((posterior[1],posterior[3],posterior[[0,2,4]].max(0)))
                    nonroad_ids = np.array([0,2,3,4],np.uint8)
                    alternative = nonroad_ids[posterior[nonroad_ids].argmax(0)]
                    quantized = np.round(road_evidence*255).clip(0,255).astype(np.uint8)
                    evidence.write(np.concatenate((quantized,alternative[None])),window=window)
                if ready<patch:
                    buffer[:,:patch-ready] = buffer[:,ready:].copy()
                    coverage[:patch-ready] = coverage[ready:].copy()
                buffer[:,patch-ready:] = 0
                coverage[patch-ready:] = 0
                if progress:
                    progress((y_index+1)/len(y_positions),f"Segmenting native pixels: row {y_index+1}/{len(y_positions)}")
        output.update_tags(model="native06m_attention_unetpp_v1",target_gsd_m="0.6",actual_gsd_m=json.dumps(source.res),
                           native_pixels="No resizing or 4x inference downsampling",
                           ignore="255=source no-data or supplied cloud/uncertainty quality mask",
                           classes=json.dumps(CLASSES))
        if evidence is not None:
            evidence.descriptions = ("Road probability x255","Bare-ground probability x255","Maximum other-class probability x255",
                                     "Best non-road class ID (0,2,3,4); not a probability")
            evidence.update_tags(scope="Quantized learned probabilities and alternative class for constrained geometry refinement; not ground truth",
                                 evidence_schema="native_road_evidence_v2")
    return {"output":str(destination.resolve()),"class_pixel_counts":counts.tolist(),
            "class_percentages":{CLASSES[i]:round(float(100*counts[i]/max(1,counts.sum())),3) for i in range(5)},
            "percentage_scope":"Valid classified pixels only",
            "quality_mask_used":str(quality_path) if quality_path else None,
            "warning":None if quality_path else "No supplied cloud quality mask: clouds and their shadows have not been excluded automatically"}


def inspect_classification_raster(labels_path):
    counts = np.zeros(5,np.int64)
    with rasterio.open(labels_path) as source:
        if not source.crs:
            raise ValueError("Cannot export georeferenced shapefiles without a CRS")
        if source.count!=1 or source.dtypes[0]!="uint8" or source.nodata!=255:
            raise ValueError("Vector export requires a single-band uint8 classified raster with Unknown/nodata=255")
        for _,window in source.block_windows(1):
            values = source.read(1,window=window)
            if not np.isin(values,[0,1,2,3,4,255]).all():
                raise ValueError("Classified raster contains values outside classes 0-4 and Unknown=255")
            counts += np.bincount(values[values!=255],minlength=5)
        return {"source_raster":str(Path(labels_path).resolve()),"crs_wkt":source.crs.to_wkt(),
                "class_pixel_counts":counts.tolist(),"ignored_value":255,
                "pixel_area_crs_units_squared":abs(source.transform.a*source.transform.e-source.transform.b*source.transform.d),
                "model_accuracy_verified":False,
                "scope":"Raster/schema/georeferencing checks only; valid geometry does not establish correct classification"}


def valid_polygon_geometry(polygon):
    if polygon.is_valid and not polygon.is_empty:
        return polygon,False
    repaired = make_valid(polygon)
    def parts(geometry):
        if geometry.geom_type=="Polygon":return [geometry]
        if geometry.geom_type in {"MultiPolygon","GeometryCollection"}:
            return [part for child in geometry.geoms for part in parts(child)]
        return []
    polygons = parts(repaired)
    result = polygons[0] if len(polygons)==1 else MultiPolygon(polygons)
    if result.is_empty or not result.is_valid or not np.isclose(result.area,polygon.area,rtol=1e-10,atol=1e-8):
        raise ValueError("Polygon repair would change the class footprint; export remains INCOMPLETE")
    return result,True


def export_shapefiles(labels_path,destination):
    destination = Path(destination)
    # Fail before creating a result folder if a raw image or wrong schema was supplied.
    audit = inspect_classification_raster(labels_path)
    destination.mkdir(parents=True,exist_ok=False)
    counts = {i:0 for i in CLASSES}
    areas = {i:0.0 for i in CLASSES}
    repaired_count = 0
    incomplete = destination/"INCOMPLETE.txt"
    incomplete.write_text("Vector export is not complete until vector_export_audit.json is present.\n")
    with rasterio.open(labels_path) as source:
        pending = {i:[] for i in CLASSES}
        def flush(class_id):
            if not pending[class_id]:
                return
            path = destination/f"class_{class_id:02d}.shp"
            frame = gpd.GeoDataFrame(pending[class_id],crs=source.crs)
            frame.to_file(path,driver="ESRI Shapefile",engine="pyogrio",index=False,
                          mode="a" if path.exists() else "w")
            pending[class_id].clear()
        # Bounded feature batches avoid a scene-sized GeoDataFrame; do not sieve roads away.
        for geometry,value in shapes(rasterio.band(source,1),connectivity=4,transform=source.transform):
            class_id = int(value)
            if class_id not in pending:
                continue
            polygon,repaired = valid_polygon_geometry(shape(geometry))
            repaired_count += int(repaired)
            areas[class_id] += polygon.area
            pending[class_id].append({"geometry":polygon,"class_id":class_id,"class_name":CLASSES[class_id]})
            counts[class_id] += 1
            if len(pending[class_id])>=1000:
                flush(class_id)
        for class_id in pending:
            flush(class_id)
        for class_id,count in counts.items():
            expected_area = audit["class_pixel_counts"][class_id]*audit["pixel_area_crs_units_squared"]
            if not np.isclose(areas[class_id],expected_area,rtol=1e-8,atol=1e-6):
                raise ValueError("Vector polygon area differs from classified pixels; export remains INCOMPLETE")
            if count:
                info = pyogrio.read_info(destination/f"class_{class_id:02d}.shp")
                if info["features"]!=count or rasterio.crs.CRS.from_user_input(info["crs"])!=source.crs:
                    raise ValueError("Written shapefile feature count/CRS verification failed; export remains INCOMPLETE")
    (destination/"README.txt").write_text("Class polygons are model predictions, not surveyed truth. CRS is inherited from the input TIFF.\nCloud/no-data value 255 is omitted. No road-erasing polygon sieve is applied.\n")
    audit.update(export_complete=True,class_feature_counts=counts,geometry_validity_checked=True,
                 zero_area_change_geometry_repairs=repaired_count,
                 class_vector_area_crs_units_squared=areas,raster_vector_area_verified=True,
                 written_feature_counts_and_crs_verified=True,
                 classes_with_no_features=[class_id for class_id,count in counts.items() if not count])
    (destination/"vector_export_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    incomplete.unlink()
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image",type=Path)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--quality-mask",type=Path)
    parser.add_argument("--shapefiles",action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if not checkpoint.get("training_complete") or checkpoint.get("native_gsd_m")!=.6:
        raise ValueError("Only completed native-0.6 m training checkpoints can produce user predictions")
    config = checkpoint["config"]
    model = NativeAttentionUNetPP(config["encoder_name"],config.get("in_channels",1))
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    device = choose_device();model.to(device)
    result = native_stream_prediction(model,args.image,args.output,device,
                                     config["patch_size"],config["inference_stride"],
                                     config["batch_size"],args.quality_mask)
    if args.shapefiles:
        result["shapefiles"] = export_shapefiles(args.output,args.output.with_suffix(""))
    args.output.with_suffix(".json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
