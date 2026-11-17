"""Bounded-memory refinement; Unknown is written consistently to maps and exports."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .native_refinement import (bridge_short_road_gaps, detect_fourband_clouds,
                                recover_narrow_roads, remove_road_specks)
from .native_road_continuity import refine_connected_native_roads,reject_isolated_native_dots


def screen_reference_clouds(path, limits, quality_path=None, maximum_dimension=2048):
    with rasterio.open(path) as source:
        scale = min(1., maximum_dimension/max(source.shape))
        shape = max(1, round(source.height*scale)), max(1, round(source.width*scale))
        raw = source.read(out_shape=(4, *shape)).astype(np.float32)
        valid = np.isfinite(raw).all(0) & (source.read_masks(out_shape=(4, *shape)) > 0).all(0)
        valid &= np.any(raw != 0, axis=0)
        if quality_path:
            with rasterio.open(quality_path) as quality:
                if quality.shape != source.shape or quality.transform != source.transform or quality.crs != source.crs:
                    raise ValueError("Cloud reference quality grid differs from the image")
                valid &= quality.read(1, out_shape=shape) == 0
        return detect_fourband_clouds(raw, valid, limits, source_width=source.width,
                                     source_height=source.height, pixel_size_m=max(source.res))


def refine_native_raster(base_path, destination, cloud_quality_path, clouds, *, parent_origin=(0, 0),
                         evidence_path=None, road_recipe=None, cleanup_recipe=None,
                         progress=None, block_size=1024, halo=96,continuity_recipe=None,
                         strict_dot_extent_m=None,maximum_gap_m=1.2):
    from native06m.predict import atomic_raster
    destination, cloud_quality_path = Path(destination), Path(cloud_quality_path)
    if destination.exists() or cloud_quality_path.exists():
        raise FileExistsError("Do not overwrite a previous refinement")
    counts = np.zeros(5, np.int64)
    cloud_count, fringe_count, initial_unknown, recovered_count = 0, 0, 0, 0
    gap_count, removed_count, reassigned_count, road_unknown_count, changed_count = 0, 0, 0, 0, 0
    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(base_path))
        evidence = stack.enter_context(rasterio.open(evidence_path)) if evidence_path else None
        if (road_recipe or cleanup_recipe or continuity_recipe or strict_dot_extent_m) and evidence is None:
            raise ValueError("Road recovery needs learned model probabilities, not geometry alone")
        if evidence and (evidence.shape != source.shape or evidence.crs != source.crs or evidence.transform != source.transform or evidence.count not in (3, 4)):
            raise ValueError("Model road-evidence grid differs from the class raster")
        if cleanup_recipe and evidence.count != 4:
            raise ValueError("Road cleanup requires the learned non-road alternative evidence band")
        if continuity_recipe and evidence.count!=4:
            raise ValueError("Connected-road refinement requires four model-evidence bands")
        if strict_dot_extent_m and (not cleanup_recipe or continuity_recipe or evidence.count != 4):
            raise ValueError("Strict dot rejection requires compact cleanup and four evidence bands")
        if continuity_recipe and halo*min(source.res)<continuity_recipe["minimum_length_m"]:
            raise ValueError("Connected-road refinement needs a full minimum road length in its halo")
        if cleanup_recipe and halo*min(source.res) < 15:
            raise ValueError("Road cleanup needs at least 15 m of block context")
        profile = source.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255)
        output = stack.enter_context(atomic_raster(destination, profile))
        quality = stack.enter_context(atomic_raster(cloud_quality_path, profile))
        total_blocks = int(np.ceil(source.width/block_size)*np.ceil(source.height/block_size))
        complete = 0
        for y in range(0, source.height, block_size):
            for x in range(0, source.width, block_size):
                width, height = min(block_size, source.width-x), min(block_size, source.height-y)
                left, top = max(0, x-halo), max(0, y-halo)
                right, bottom = min(source.width, x+width+halo), min(source.height, y+height+halo)
                window = Window(left, top, right-left, bottom-top)
                labels = source.read(1, window=window)
                if not np.isin(labels, [0, 1, 2, 3, 4, 255]).all():
                    raise ValueError("Invalid class IDs in the base-model raster")
                original = labels.copy()
                codes = clouds.window(parent_origin[0]+left, parent_origin[1]+top, right-left, bottom-top)
                valid = labels != 255
                codes = codes.copy()
                codes[~valid] = 255
                labels[codes != 0] = 255
                learned = evidence.read(window=window) if evidence is not None else None
                before_recovery = labels.copy()
                continuity_stats=None
                if continuity_recipe:
                    labels,continuity_stats=refine_connected_native_roads(labels,learned,labels!=255,
                        continuity_recipe,gsd_m=max(source.res),return_masks=True)
                elif road_recipe:
                    labels, _ = recover_narrow_roads(labels, learned, labels != 255,
                                                     road_recipe, gsd_m=max(source.res))
                after_recovery = labels.copy()
                if cleanup_recipe and not continuity_recipe:
                    labels, _ = bridge_short_road_gaps(labels, learned, labels != 255, gsd_m=max(source.res),maximum_gap_m=maximum_gap_m)
                before_cleanup = labels.copy()
                if cleanup_recipe and not continuity_recipe:
                    labels, _ = remove_road_specks(labels, learned, labels != 255, cleanup_recipe, gsd_m=max(source.res))
                    if strict_dot_extent_m:
                        labels,_=reject_isolated_native_dots(labels,learned,labels!=255,
                            maximum_extent_m=strict_dot_extent_m,gsd_m=max(source.res))
                    codes[(labels == 255) & (before_cleanup == 1) & (codes == 0)] = 3
                crop = (slice(y-top, y-top+height), slice(x-left, x-left+width))
                labels, codes, original = labels[crop], codes[crop], original[crop]
                before_recovery, after_recovery, before_cleanup = (value[crop] for value in
                                                                   (before_recovery, after_recovery, before_cleanup))
                out_window = Window(x, y, width, height)
                output.write(labels, 1, window=out_window)
                quality.write(codes, 1, window=out_window)
                known = labels != 255
                counts += np.bincount(labels[known], minlength=5)
                cloud_count += int(np.count_nonzero(codes == 1))
                fringe_count += int(np.count_nonzero(codes == 2))
                initial_unknown += int(np.count_nonzero(original == 255))
                if continuity_stats is not None:
                    recovered_count+=int(continuity_stats["_recovered_mask"][crop].sum())
                    gap_count+=int(continuity_stats["_gap_mask"][crop].sum())
                    removed=continuity_stats["_removed_mask"][crop]
                else:
                    recovered_count += int(np.count_nonzero((before_recovery == 3) & (after_recovery == 1) & (labels == 1)))
                    gap_count += int(np.count_nonzero((after_recovery != 1) & (before_cleanup == 1) & (labels == 1)))
                    removed = (before_cleanup == 1) & (labels != 1)
                removed_count += int(np.count_nonzero(removed))
                reassigned_count += int(np.count_nonzero(removed & (labels != 255)))
                road_unknown_count += int(np.count_nonzero(codes == 3))
                changed_count += int(np.count_nonzero(original != labels))
                complete += 1
                if progress:
                    progress(complete/total_blocks, f"Cloud/road refinement: block {complete}/{total_blocks}")
        output.update_tags(**source.tags())
        output.update_tags(refinement=("native_connected_roads_v3" if continuity_recipe else
                                      "native_supported_gap_refinement_v3" if maximum_gap_m != 1.2 or strict_dot_extent_m else
                                      "fourband_cloud_and_connectivity_road_refinement_v2"),
                           road_recipe=road_recipe or "disabled", road_recovered_pixels=str(recovered_count),
                           road_cleanup_recipe=cleanup_recipe or "disabled", road_gap_pixels_added=str(gap_count),
                           road_speck_pixels_removed=str(removed_count), road_speck_unknown_pixels=str(road_unknown_count),
                           cloud_unknown_pixels=str(cloud_count+fringe_count),
                           maximum_gap_m=str(maximum_gap_m), strict_dot_extent_m=str(strict_dot_extent_m),
                           base_model_raster=Path(base_path).name, ignore="255=Unknown; never Water")
        quality.set_band_description(1, "0=not excluded, 1=likely cloud, 2=cloud fringe, 3=uncertain road speck, 255=source no-data")
        quality.update_tags(method=clouds.metadata["method"], warning=clouds.metadata["limitation"])
        size = source.width*source.height
    return {"class_pixel_counts": counts.tolist(), "source_nodata_pixels": initial_unknown,
            "likely_cloud_pixels": cloud_count, "uncertain_fringe_pixels": fringe_count,
            "excluded_cloud_pixels": cloud_count+fringe_count,
            "excluded_percent_of_full_grid": round(100*(cloud_count+fringe_count)/max(1, size), 4),
            "classified_pixels": int(counts.sum()), "road_recovered_pixels": recovered_count,
            "road_gap_pixels_added": gap_count, "road_speck_pixels_removed": removed_count,
            "road_speck_pixels_reassigned": reassigned_count, "road_speck_unknown_pixels": road_unknown_count,
            "road_cleanup_recipe": cleanup_recipe, "refinement_changed_pixels": changed_count,
            "road_recipe": "native_connected_v3" if continuity_recipe else road_recipe,
            "strict_dot_extent_m":strict_dot_extent_m,"maximum_gap_m":maximum_gap_m,
            "continuity_recipe":continuity_recipe,"model_output_preserved": changed_count == 0}
