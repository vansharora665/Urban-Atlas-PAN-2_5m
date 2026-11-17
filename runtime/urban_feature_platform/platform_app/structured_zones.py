"""Separate pre-argmax neighbourhood model; accepted roads stay immutable."""
import hashlib
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
import torch
from PIL import Image
from scipy import ndimage as ndi
from rasterio.vrt import WarpedVRT

from . import cartography
from .model_roads import result_directory

VERSION = 'structured_zones_v1'
POINTER = 'structured_zones_latest.json'


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def native_modules(settings):
    from .native06m_inference import Native06mInference
    return Native06mInference(settings)


def dependency_snapshot(*paths):
    pending = [Path(p).resolve() for p in paths if p]
    records = {}
    while pending:
        path = pending.pop()
        if str(path) in records:
            continue
        stat = path.stat()
        records[str(path)] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}
        if path.suffix.lower() == '.vrt':
            for element in ET.parse(path).iter('SourceFilename'):
                value = element.text or ''
                if value.startswith('/vsi') or '://' in value:
                    raise ValueError('Structured inference requires local source dependencies')
                child = Path(value)
                pending.append((path.parent / child if element.get('relativeToVRT') == '1' else child).resolve())
    return records


def model_path(settings):
    return settings.root_dir.parent / 'native_06m_specialist/structured_zones_v1/context_head.pt'


def saved_layer(settings, identifier):
    directory = result_directory(settings, identifier)
    try:
        pointer = directory / POINTER
        if pointer.is_symlink():
            return None
        folder = json.loads(pointer.read_text())['folder']
        if Path(folder).name != folder or not folder.startswith(VERSION + '_'):
            return None
        home = directory / folder
        if home.is_symlink():
            return None
        if (home / 'summary.json').is_symlink():
            return None
        value = json.loads((home / 'summary.json').read_text())
        roads = cartography.saved_layer(settings, identifier)
        if not roads or value['accepted_roads_folder'] != roads['folder']:
            return None
        for name in ('landcover_native.tif', 'edits_native.tif', 'map_preview.png'):
            if not (home / name).is_file() or (home / name).is_symlink():
                return None
        value['folder'] = folder
        value['artifacts'] = {k: f'/results/{identifier}/{folder}/{name}' if (home / name).is_file() else None
                              for k, name in cartography.ARTIFACTS.items()}
        for key in ('road_geotiff_url', 'inferred_geotiff_url', 'graph_url', 'network_geojson_url', 'road_preview_url'):
            value['artifacts'][key] = roads['artifacts'][key]
        value['artifacts']['viewer_url'] = f'/maps/{identifier}/zones'
        value['tiles_url'] = f'/api/results/{identifier}/zones/tiles/{{layer}}/{{z}}/{{x}}/{{y}}.png?v={folder}'
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def source_inputs(settings, identifier):
    directory = result_directory(settings, identifier)
    source, quality = directory / 'native_input_stack.vrt', directory / 'native_quality.vrt'
    roads = cartography.saved_layer(settings, identifier)
    if not roads or not source.is_file():
        raise ValueError('This branch needs a saved four-band 0.6 m input and its accepted usable road layer first')
    with rasterio.open(source) as image:
        if image.count != 4 or any(abs(r - .6) > .03 for r in image.res):
            raise ValueError('Neighbourhood model currently supports four-band 0.6 m imagery only')
    return directory, source, quality if quality.exists() else None, roads


def evidence(settings, registry, identifier, progress):
    import fcntl
    directory = result_directory(settings, identifier)
    lock_path = directory / '.structured_evidence.lock'
    if lock_path.is_symlink():
        raise ValueError('Linked model lock not supported')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('The neural evidence pass is already running for this scene')
        return _evidence(settings, registry, identifier, progress)


def _evidence(settings, registry, identifier, progress):
    from .model_roads import ModelRoadService
    from native06m.predict import native_stream_prediction, atomic_raster
    from native06m.structured_zones import CoarsePosteriorWriter
    directory, source, quality, roads = source_inputs(settings, identifier)
    record = registry.get('native_06m_joint_attention_v1')
    limits, normalization_scope = ModelRoadService(settings, registry).normalization(directory,
                  json.loads((directory / 'prediction_summary.json').read_text()))
    home = directory / (VERSION + '_evidence'); home.mkdir(exist_ok=True)
    cache = home / 'neural_scores_2p4m.tif'
    dependencies = dependency_snapshot(source, quality)
    dependency_path = home / 'source_dependencies.json'
    signature = {'checkpoint': record.summary['checkpoint_sha256'], 'source_vrt': digest(source),
                 'limits': np.asarray(limits).tolist(), 'factor': 4,
                 'quality_vrt': digest(quality) if quality else None, 'normalization_scope': normalization_scope}
    with rasterio.open(source) as original:
        signature['source_files'] = [{'path': p, 'bytes': Path(p).stat().st_size,
            'mtime_ns': Path(p).stat().st_mtime_ns} for p in original.files]
    tag = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if cache.exists():
        with rasterio.open(cache) as stored:
            if stored.tags().get('fingerprint') != tag:
                raise ValueError('Existing evidence cache differs from the model/input; preserve it before rebuilding')
        if dependency_path.is_file():
            if json.loads(dependency_path.read_text()) != dependencies:
                raise ValueError('A source band or quality dependency changed; do not reuse cached neural evidence')
        else:
            # One-time migration of the cache generated during this build. All
            # actual source files must predate that completed evidence artifact.
            if any(max(row['mtime_ns'], row['ctime_ns']) > cache.stat().st_mtime_ns for row in dependencies.values()):
                raise ValueError('Cannot establish provenance of this older evidence cache')
            cartography.atomic_json(dependency_path, dependencies)
        return cache, limits, signature
    model, device = native_modules(settings).load_model(record)
    with rasterio.open(source) as original:
        profile = original.profile.copy()
        profile.update(driver='GTiff', width=math.ceil(original.width / 4), height=math.ceil(original.height / 4),
            transform=original.transform * Affine.scale(4, 4), count=6, dtype='uint16', nodata=None,
            compress='lzw', tiled=True, blockxsize=256, blockysize=256)
        height, width = original.shape
    scratch = home / 'fresh_argmax_temporary.tif'
    scratch.unlink(missing_ok=True)
    with atomic_raster(cache, profile) as output:
        writer = CoarsePosteriorWriter(output, width)
        native_stream_prediction(model, source, scratch, device, patch=512, stride=384, batch_size=4,
            quality_path=quality, normalization_limits=limits, posterior_callback=writer,
            progress=lambda fraction, message: progress(.02 + .65 * fraction, message))
        writer.finish(height)
        output.descriptions = tuple([f'Frozen posterior class {k} x65535' for k in range(5)] + ['Valid native sample count'])
        output.update_tags(fingerprint=tag, inference='Native 0.6 m neural evidence; validity-weighted 4x4 context pooling',
                           calibrated='false')
    scratch.unlink()
    if dependency_snapshot(source, quality) != dependencies:
        raise ValueError('Source imagery changed during neural inference; do not publish this evidence')
    cartography.atomic_json(dependency_path, dependencies)
    return cache, limits, signature


def read_context(source, cache, road_path, limits):
    with rasterio.open(cache) as evidence:
        values = evidence.read()
        shape, transform = evidence.shape, evidence.transform
    valid = values[5] > 0
    posterior = values[:5].astype(np.float32) / 65535.
    with rasterio.open(source) as image, WarpedVRT(image, crs=image.crs, transform=transform,
            width=shape[1], height=shape[0], resampling=Resampling.average) as aligned:
        raw = aligned.read().astype(np.float32)
    limits = np.asarray(limits, np.float32)
    image = np.clip((raw - limits[:, 0, None, None]) / (limits[:, 1] - limits[:, 0])[:, None, None], 0, 1)
    image[:, ~valid] = 0
    with rasterio.open(road_path) as roads:
        mask = roads.read(1, out_shape=shape, resampling=Resampling.nearest) == 1
    return image, posterior, valid, mask, transform


def build(settings, registry, identifier, progress=lambda *_: None):
    started = time.perf_counter()
    cached = saved_layer(settings, identifier)
    if cached:
        return cached
    native_modules(settings)
    from native06m.structured_zones import ZoneOptions, infer_regions
    directory, source, quality, accepted = source_inputs(settings, identifier)
    head_path = model_path(settings)
    if not head_path.is_file():
        raise ValueError('Prepare and validate the context-head candidate before structured inference')
    checkpoint = torch.load(head_path, map_location='cpu', weights_only=True)
    urban_path = head_path.with_name('urban_zone_head.pt')
    urban_checkpoint = torch.load(urban_path, map_location='cpu', weights_only=True) if urban_path.is_file() else None
    if urban_checkpoint and urban_checkpoint['base_checkpoint_sha256'] != checkpoint['base_checkpoint_sha256']:
        raise ValueError('Urban context head and frozen backbone fingerprints differ')
    options = ZoneOptions()
    cache, limits, signature = evidence(settings, registry, identifier, progress)
    road_home = directory / accepted['folder']
    road_hash = digest(road_home / 'roads_native.tif')
    key = hashlib.sha256(json.dumps({'evidence': signature, 'options': asdict(options),
        'roads': road_hash, 'context_candidate': digest(head_path),
        'urban_context': digest(urban_path) if urban_checkpoint else None,
        'inference_code': digest(settings.root_dir.parent / 'native_06m_specialist/src/native06m/structured_zones.py')}, sort_keys=True).encode()).hexdigest()
    home = directory / (VERSION + '_' + key[:12]); home.mkdir(exist_ok=False)
    (home / 'INCOMPLETE.txt').write_text('Not published until summary and final pointer exist.\n')
    progress(.69, 'Building image-guided neighbourhoods from four bands and neural scores')
    image, posterior, valid, roads, transform = read_context(source, cache, road_home / 'roads_native.tif', limits)
    labels, regions, audit, _ = infer_regions(image, posterior, valid, roads, 2.4, checkpoint, options, urban_checkpoint)
    if (roads & valid).any():
        if not ((regions > 0) & valid).any():
            raise ValueError('No non-road regions available')
        nearest = ndi.distance_transform_edt(regions == 0, return_distances=False, return_indices=True)
        labels[roads & valid] = labels[tuple(index[roads & valid] for index in nearest)]
    progress(.83, 'Writing native-grid zones and preserving the accepted road surface exactly')
    counts, changed, unknown = np.zeros(5, np.int64), 0, 0
    with rasterio.open(directory / 'segmentation_labels_georef.tif') as raw, rasterio.open(road_home / 'roads_native.tif') as road:
        profile = raw.profile.copy(); profile.update(driver='GTiff', compress='lzw', tiled=True,
            blockxsize=256, blockysize=256)
        with rasterio.open(home / 'landcover_native.tif', 'w', **profile) as output, \
             rasterio.open(home / 'edits_native.tif', 'w', **profile) as edits:
            for _, window in raw.block_windows(1):
                y, x = int(window.row_off), int(window.col_off)
                h, w = int(window.height), int(window.width)
                yy = (np.arange(y, y + h) // 4).clip(0, labels.shape[0] - 1)
                xx = (np.arange(x, x + w) // 4).clip(0, labels.shape[1] - 1)
                value = labels[yy[:, None], xx[None, :]].copy()
                road_pixels = road.read(1, window=window)
                # Accepted surface is authoritative; never let coarse masking widen it.
                value[road_pixels == 1] = 1
                value[road_pixels == 255] = 255
                original = raw.read(1, window=window)
                delta = np.where(value == 255, 255, (value != original).astype('uint8'))
                counts += np.bincount(value[value < 5], minlength=5)
                changed += int(((value != original) & (value < 5)).sum()); unknown += int((value == 255).sum())
                output.write(value, 1, window=window); edits.write(delta, 1, window=window)
            output.update_tags(product=VERSION, mapping_target='Dominant neighbourhood land-cover zones',
                backbone_retrained='false', learned_context_head_deployed=str(bool(urban_checkpoint and urban_checkpoint.get('deployment_accepted'))), accuracy_claim='none',
                roads='Exact accepted native road surface, unchanged')
            cartography.write_palette(output)
            for raster in (output, edits):
                factors = [f for f in (2, 4, 8, 16, 32, 64) if min(raw.shape) // f >= 1]
                raster.build_overviews(factors, Resampling.nearest)
    for name in ('roads_native.tif', 'inferred_links_native.tif', 'road_graph.json', 'road_network_wgs84.geojson'):
        shutil.copy2(road_home / name, home / name)
    cartography.previews(directory, home)
    progress(.95, 'Verifying road identity, source grid and published class counts')
    if digest(home / 'roads_native.tif') != road_hash:
        raise ValueError('Road identity check failed')
    with rasterio.open(home / 'landcover_native.tif') as out, rasterio.open(source) as native:
        if out.shape != native.shape or out.transform != native.transform or out.crs != native.crs:
            raise ValueError('Native-grid export mismatch')
    summary = {'version': VERSION, 'input_fingerprint': key, 'product_kind': 'structured_zones',
        'accepted_roads_folder': accepted['folder'], 'road_sha256_unchanged': road_hash,
        'native_grid': accepted['native_grid'], 'working_grid': {'gsd_m': 2.4},
        'network': accepted['network'], 'road_cleanup': accepted['road_cleanup'],
        'maximum_tile_zoom': accepted['maximum_tile_zoom'], 'tile_size': 256,
        'class_percentages': [{'class_id': k, 'class_name': cartography.NAMES[k], 'pixel_count': int(counts[k]),
            'percentage': float(100 * counts[k] / max(1, counts.sum()))} for k in range(5)],
        'unknown_pixels': unknown, 'change_audit': {'changed_pixels': changed},
        'structured_model': audit, 'base_evidence': signature,
        'context_head_decision': 'Rejected: validation dominant-zone accuracy 85.32% vs pooled backbone 95.41%; head weight=0',
        'urban_context_head': {'deployed': bool(urban_checkpoint and urban_checkpoint.get('deployment_accepted')),
            'sha256': digest(urban_path) if urban_checkpoint else None,
            'validation': urban_checkpoint.get('validation') if urban_checkpoint else None,
            'scope': 'Derived neighbourhood-use targets, not pixel accuracy or an untouched test'},
        'accuracy_claim': None, 'seconds': time.perf_counter() - started,
        'limitations': ['Dominant cover zones are not cadastral parcels or individual building footprints',
            'Small true objects may be merged; no new pixel accuracy or IoU claim',
            'Chennai has been visually inspected; it is not an untouched blind test',
            'Unknown source/accepted road-domain exclusions stay Unknown; road graph is not navigation validated']}
    cartography.atomic_json(home / 'summary.json', summary)
    (home / 'INCOMPLETE.txt').unlink()
    cartography.atomic_json(directory / POINTER, {'folder': home.name})
    progress(1., 'Neighbourhood-zone model ready; original predictions and accepted roads unchanged')
    return saved_layer(settings, identifier)


def tile(settings, identifier, layer, z, x, y):
    addon = saved_layer(settings, identifier)
    if not addon or layer not in ('classes', 'raw', 'roads', 'edits', 'imagery'):
        raise ValueError('Structured map unavailable')
    maximum = addon['maximum_tile_zoom']
    if not 0 <= z <= maximum or not 0 <= x < 2**z or not 0 <= y < 2**z:
        raise ValueError('Tile outside pyramid')
    directory = result_directory(settings, identifier)
    if layer == 'imagery':
        return imagery_tile(str(directory / 'native_input_stack.vrt'),
            tuple(tuple(row) for row in addon['base_evidence']['limits']),
            z, x, y, maximum, addon['input_fingerprint'])
    path = directory / addon['folder'] / {'classes': 'landcover_native.tif', 'roads': 'roads_native.tif',
                                        'edits': 'edits_native.tif', 'raw': 'unused'}[layer]
    if layer == 'raw':
        path = directory / 'segmentation_labels_georef.tif'
    return cartography._tile(str(path), layer, z, x, y, maximum, addon['input_fingerprint'])


@lru_cache(maxsize=128)
def imagery_tile(path, limits, z, x, y, maximum, signature):
    """Native source imagery for close-up review; no model or texture synthesis."""
    factor = 2 ** (maximum - z)
    window = Window(x * 256 * factor, y * 256 * factor, 256 * factor, 256 * factor)
    with rasterio.open(path) as source:
        raw = source.read([3, 2, 1], window=window, out_shape=(3, 256, 256),
            boundless=True, fill_value=0, resampling=Resampling.average).astype('float32')
    valid = np.any(raw > 0, axis=0)
    limits = np.asarray(limits, 'float32')[[2, 1, 0]]
    rgb = np.clip((raw - limits[:, 0, None, None]) / (limits[:, 1] - limits[:, 0])[:, None, None], 0, 1)
    rgba = np.zeros((256, 256, 4), 'uint8')
    rgba[:, :, :3] = np.moveaxis(np.round(np.power(rgb, .75) * 255).astype('uint8'), 0, -1)
    rgba[:, :, 3] = valid * 255
    output = io.BytesIO(); Image.fromarray(rgba).save(output, format='PNG')
    return output.getvalue()
