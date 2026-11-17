"""Optional OSM main-road priors, never relabelled as autonomous model output."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from xml.etree import ElementTree
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform, transform_bounds, transform_geom
from rasterio.windows import Window
from shapely.geometry import LineString, box, mapping
from shapely.strtree import STRtree

from .saved_results import RESULT_ID


ATTRIBUTION = 'OpenStreetMap contributors; data under ODbL 1.0'
LICENSE_URL = 'https://www.openstreetmap.org/copyright'
ROAD_TYPES = ('motorway', 'trunk', 'primary', 'secondary', 'tertiary',
              'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link')
PROVIDERS = ('https://maps.mail.ru/osm/tools/overpass/api/interpreter',
             'https://overpass-api.de/api/interpreter')
SMALL_MAP_PROVIDER = 'https://www.openstreetmap.org/api/0.6/map'
MAX_BYTES = 12_000_000
_fetch_lock = threading.Lock()
_build_lock = threading.Lock()
_last_failure = {}


def request_bytes(provider, query, parameter='data'):
    # macOS' system curl has a more reliable TLS stack than the CLT Python.
    # URLs are fixed providers and no shell interpolation or certificate bypass is used.
    curl = shutil.which('curl')
    if curl:
        completed = subprocess.run([curl, '--ipv4', '--fail', '--silent', '--show-error', '--location',
                                    '--max-time', '75', '--max-filesize', str(MAX_BYTES),
                                    '--user-agent', 'UrbanAtlas-IITB-BTP-Research/1.0',
                                    '--get', '--data-urlencode', parameter+'='+query, provider],
                                   capture_output=True, timeout=80, check=True)
        return completed.stdout
    request = Request(provider+'?'+urlencode({parameter: query}), headers={
        'User-Agent': 'UrbanAtlas-IITB-BTP-Research/1.0', 'Accept': 'application/json'})
    with urlopen(request, timeout=75) as response:
        return response.read(MAX_BYTES+1)


def parse_small_map(raw):
    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('DTD/entity declarations are not accepted in map data')
    root = ElementTree.fromstring(raw)
    nodes = {node.attrib['id']: {'lon': float(node.attrib['lon']), 'lat': float(node.attrib['lat'])}
             for node in root.findall('node')}
    if len(nodes) > 50_000:
        raise ValueError('Small map response exceeds the node limit')
    elements = []
    for way in root.findall('way'):
        references = [node.attrib['ref'] for node in way.findall('nd')]
        if all(ref in nodes for ref in references):
            elements.append({'type': 'way', 'id': int(way.attrib['id']),
                             'version': int(way.attrib.get('version', 0)), 'timestamp': way.attrib.get('timestamp'),
                             'tags': {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag')},
                             'geometry': [nodes[ref] for ref in references]})
    return {'version': .6, 'elements': elements, 'osm3s': {'copyright': ATTRIBUTION}}


def result_directory(settings, identifier):
    if not RESULT_ID.fullmatch(identifier):
        raise ValueError('Invalid result identifier')
    directory = settings.result_dir / identifier
    if directory.is_symlink() or not (directory / 'prediction_summary.json').is_file():
        raise ValueError('Open a completed analysis first')
    return directory


def road_width(tags):
    text = str(tags.get('width', '')).strip().lower()
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(m|metres|meters)?', text)
    if match and 2 <= float(match[1]) <= 40:
        return float(match[1]), 'OSM width tag; unverified'
    lanes = str(tags.get('lanes', ''))
    if lanes.isdigit() and 1 <= int(lanes) <= 8:
        return min(28., max(3.25, int(lanes)*3.25)), 'Assumed 3.25 m per tagged lane; not measured'
    return 7., 'Assumed 7 m corridor; not measured pavement'


def normalize_ways(data):
    if not isinstance(data, dict) or data.get('remark') or not isinstance(data.get('elements'), list):
        raise ValueError('OSM returned an incomplete or invalid road response; no output was fabricated')
    if len(data['elements']) > 50_000:
        raise ValueError('Road response is too large; use a smaller area or an offline extract')
    ways = []
    for item in data['elements']:
        tags = item.get('tags') or {}
        if (item.get('type') != 'way' or tags.get('highway') not in ROAD_TYPES or tags.get('railway')
                or tags.get('area') == 'yes' or tags.get('access') == 'no'):
            continue
        geometry = item.get('geometry') or []
        points = [(float(p['lon']), float(p['lat'])) for p in geometry if 'lon' in p and 'lat' in p]
        if len(points) != len(geometry) or len(points) < 2 or len(points) > 100_000:
            continue
        if not all(math.isfinite(x) and math.isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90 for x, y in points):
            continue
        width, source = road_width(tags)
        ways.append({'osm_id': str(int(item['id'])), 'road_type': tags['highway'],
                     'name': str(tags.get('name', ''))[:300], 'width_m': width, 'width_source': source,
                     'bridge': str(tags.get('bridge', 'no')), 'tunnel': str(tags.get('tunnel', 'no')),
                     'layer': str(tags.get('layer', '0')), 'osm_version': item.get('version'),
                     'osm_timestamp': item.get('timestamp'), 'coordinates': points})
    return ways


def fetch_roads(bounds, cache_directory):
    west, south, east, north = bounds
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('Invalid geographic extent, including unsupported antimeridian crossings')
    # Bound public-server work rather than partitioning a huge scene into many calls.
    area = (east-west)*111.32*max(.05, math.cos(math.radians((north+south)/2)))*(north-south)*111.32
    if area > 2000:
        raise ValueError('Map-assisted public lookup is limited to 2000 square kilometres; crop the scene')
    query = ('[out:json][timeout:60];way[highway~"^(' + '|'.join(ROAD_TYPES) + ')$"]'
             f'({south:.6f},{west:.6f},{north:.6f},{east:.6f});out meta geom;')
    key = hashlib.sha256(query.encode()).hexdigest()
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    from .osm_offline import read_offline
    local = read_offline(bounds, cache_directory)
    if local is not None:
        normalize_ways(local[0])
        return local
    path = cache_directory / (key + '.json')
    with _fetch_lock:
        candidates = ([path] if path.is_file() else []) + [p for p in cache_directory.glob('*.json') if p != path]
        for candidate in candidates:
            if time.time()-candidate.stat().st_mtime >= 7*86400:
                continue
            cached = json.loads(candidate.read_text())
            extent = cached.get('provenance', {}).get('queried_bounds_wgs84', [])
            if len(extent) != 4 or not (extent[0] <= west and extent[1] <= south and extent[2] >= east and extent[3] >= north):
                continue
            normalize_ways(cached['data'])
            if hashlib.sha256(json.dumps(cached['data'], sort_keys=True).encode()).hexdigest() != cached['data_sha256']:
                raise ValueError('OSM cache checksum mismatch')
            return cached['data'], {**cached['provenance'], 'cache_hit': True,
                                     'requested_bounds_wgs84': list(bounds), 'covering_spatial_cache': candidate != path}
        errors = []
        # The OSM editing API is used only for a single small <=0.5 km2 crop,
        # never to bulk-download or tile an entire city.
        providers = ((SMALL_MAP_PROVIDER,) if area <= .5 else ()) + PROVIDERS
        for provider in providers:
            if time.monotonic()-_last_failure.get(provider, -1e9) < 30:
                continue
            try:
                raw = (request_bytes(provider, ','.join(f'{v:.7f}' for v in bounds), parameter='bbox')
                       if provider == SMALL_MAP_PROVIDER else request_bytes(provider, query))
                if len(raw) > MAX_BYTES:
                    raise ValueError('Road response exceeds the 12 MB safety limit')
                data = parse_small_map(raw) if provider == SMALL_MAP_PROVIDER else json.loads(raw)
                normalize_ways(data)
                provenance = {'provider': provider, 'queried_bounds_wgs84': list(bounds),
                              'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                              'osm_database_timestamp': data.get('osm3s', {}).get('timestamp_osm_base'),
                              'attribution': ATTRIBUTION, 'license_url': LICENSE_URL,
                              'cache_hit': False, 'satellite_pixels_uploaded': False}
                cached = {'data': data, 'provenance': provenance,
                          'data_sha256': hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(cached))
                temporary.replace(path)
                return data, provenance
            except Exception as error:
                _last_failure[provider] = time.monotonic()
                detail = error.stderr.decode(errors='replace')[-300:] if isinstance(error, subprocess.CalledProcessError) else str(error)
                logging.getLogger(__name__).warning('OSM road provider %s failed: %s', provider, detail)
                errors.append(type(error).__name__)
        raise ValueError('Public OSM road lookup is unavailable ('+', '.join(errors)+'). '
                         'Your model-only result is unchanged. Retry later; no network was invented.')


def projected_corridors(ways, crs, extent):
    geometries, features = [], []
    region = box(*extent)
    for way in ways:
        lon, lat = zip(*way['coordinates'])
        xs, ys = transform('EPSG:4326', crs, lon, lat)
        line = LineString(zip(xs, ys))
        clipped = line.intersection(region)
        if clipped.is_empty or clipped.length == 0:
            continue
        geometry = line.buffer(way['width_m']/2, cap_style=2).intersection(region)
        if geometry.is_empty:
            continue
        geometries.append(geometry)
        features.append({'type': 'Feature', 'geometry': transform_geom(crs, 'EPSG:4326', mapping(clipped)),
                         'properties': {key: value for key, value in way.items() if key != 'coordinates'} |
                                       {'source': 'OSM map prior, not model prediction', 'length_inside_scene_m': clipped.length}})
    return geometries, features


def build_network(settings, identifier, progress=None, fetcher=fetch_roads):
    progress = progress or (lambda *_: None)
    with _build_lock:
        directory = result_directory(settings, identifier)
        existing = directory / 'map_assisted_summary.json'
        if existing.is_file():
            return json.loads(existing.read_text())
        summary = json.loads((directory / 'prediction_summary.json').read_text())
        road_class = next((row for row in summary.get('class_percentages', []) if row.get('class_id') == 1), {})
        if 'road' not in str(road_class.get('class_name', '')).lower():
            raise ValueError('This older model has no compatible vehicle-road class; use a current five-class analysis')
        labels_path = directory / 'segmentation_labels_georef.tif'
        if not labels_path.is_file():
            raise ValueError('This saved result has no georeferenced class raster')
        progress(.03, 'Checking georeferencing; model-only output will remain unchanged')
        with rasterio.open(labels_path) as source:
            if not source.crs or not source.crs.is_projected or source.crs.linear_units != 'metre':
                raise ValueError('Map-assisted width estimates need a projected metre CRS')
            bounds = transform_bounds(source.crs, 'EPSG:4326', *source.bounds, densify_pts=21)
            progress(.07, 'Checking local OSM main roads; public lookup only if no local index covers the scene')
            data, provenance = fetcher(bounds, settings.runtime_dir / 'osm_road_cache')
            ways = normalize_ways(data)
            geometries, features = projected_corridors(ways, source.crs, source.bounds)
            if not geometries:
                raise ValueError('No mapped main vehicle roads intersect this scene; no roads were invented')
            tree = STRtree(geometries)
            profile = source.profile.copy()
            profile.update(driver='GTiff', count=1, dtype='uint8', nodata=255, compress='deflate')
            temporary = directory / 'map_assisted_corridors.partial.tif'
            raster_path = directory / 'map_assisted_corridors_georef.tif'
            counts = {1: 0, 2: 0, 3: 0}
            total = math.ceil(source.width/1024)*math.ceil(source.height/1024)
            complete = 0
            try:
                with rasterio.open(temporary, 'w', **profile) as destination:
                    for y in range(0, source.height, 1024):
                        for x in range(0, source.width, 1024):
                            window = Window(x, y, min(1024, source.width-x), min(1024, source.height-y))
                            labels = source.read(1, window=window)
                            indices = tree.query(box(*rasterio.windows.bounds(window, source.transform)))
                            shapes = [(mapping(geometries[int(i)]), 1) for i in indices]
                            mask = rasterize(shapes, out_shape=labels.shape, transform=source.window_transform(window), dtype='uint8') if shapes else np.zeros(labels.shape, np.uint8)
                            output = mask.copy()
                            output[(mask == 1) & (labels == 1)] = 2
                            output[(mask == 1) & (labels == 255)] = 3
                            output[(labels == 255) & (mask == 0)] = 255
                            for code in counts:
                                counts[code] += int((output == code).sum())
                            destination.write(output, 1, window=window)
                            complete += 1
                            progress(.15+.7*complete/total, f'Map-assisted corridors: block {complete}/{total}')
                    destination.set_band_description(1, '0=outside; 1=map-only prior; 2=overlap with model road class; 3=obscured/Unknown map prior; 255=source Unknown')
                    destination.update_tags(source='OpenStreetMap map-assisted prior, NOT autonomous prediction',
                                            attribution=ATTRIBUTION, license=LICENSE_URL,
                                            width_warning='Tagged or estimated widths; not surveyed road boundaries', model_output_changed='false')
                temporary.replace(raster_path)
            finally:
                temporary.unlink(missing_ok=True)
        geojson_path = directory / 'map_assisted_network_wgs84.geojson'
        geojson_path.write_text(json.dumps({'type': 'FeatureCollection', 'name': 'OSM main-road priors, not model predictions',
            'attribution': ATTRIBUTION, 'license': LICENSE_URL, 'features': features}))
        progress(.9, 'Rendering a separate map-assisted overlay')
        with Image.open(directory / 'input_preview.png') as image:
            preview = np.asarray(image.convert('RGB')).copy()
        with rasterio.open(raster_path) as raster:
            codes = raster.read(1, out_shape=preview.shape[:2], resampling=rasterio.enums.Resampling.nearest)
        for code, color in ((1, (245, 158, 11)), (2, (16, 185, 129)), (3, (190, 190, 190))):
            selected = codes == code
            preview[selected] = (.25*preview[selected]+.75*np.array(color)).astype(np.uint8)
        Image.fromarray(preview).save(directory / 'map_assisted_overlay.png')
        corridor_pixels = sum(counts.values())
        url = lambda name: f'/results/{identifier}/{name}'
        result = {'result_id': identifier, 'mode': 'map_assisted_main_roads_v1', 'provenance': provenance,
                  'mapped_road_ways': len(features), 'corridor_pixel_counts': {str(k): v for k, v in counts.items()},
                  'model_road_overlap_percent': 100*counts[2]/max(1, corridor_pixels),
                  'overlap_is_not_accuracy': True, 'model_output_changed': False, 'measured_road_iou': None,
                  'road_scope': 'Main vehicle-road classes only; residential lanes, fields, footpaths and railways excluded',
                  'width_warning': 'OSM width tags or lane/default estimates, not measured pavement edges.',
                  'warning': 'External map-assisted network, not image-only prediction or ground truth. '
                             'OSM dates/alignment/completeness may differ from this TIFF. '
                             'Amber: map-only corridor; green: overlaps model road pixels; grey: Unknown. '
                             'No clouds, water or other classes were relabelled. Roads may be missing from either source.',
                  'artifacts': {'map_assisted_preview_url': url('map_assisted_overlay.png'),
                                'map_assisted_geotiff_url': url(raster_path.name),
                                'map_assisted_geojson_url': url(geojson_path.name),
                                'map_assisted_summary_url': url('map_assisted_summary.json')}}
        existing.write_text(json.dumps(result, indent=2))
        return result


def export_network_shapefile(settings, identifier):
    """Explicit user request only; never reuse the model's class-shapefile names."""
    import geopandas as gpd

    with _build_lock:
        directory = result_directory(settings, identifier)
        summary_path = directory / 'map_assisted_summary.json'
        if not summary_path.is_file():
            raise ValueError('Build the separate OSM-assisted layer first')
        summary = json.loads(summary_path.read_text())
        destination = directory / 'map_assisted_main_roads_shapefile.zip'
        if not destination.exists():
            data = json.loads((directory / 'map_assisted_network_wgs84.geojson').read_text())
            frame = gpd.GeoDataFrame.from_features(data['features'], crs='EPSG:4326')
            with rasterio.open(directory / 'segmentation_labels_georef.tif') as source:
                target_crs = source.crs
            frame = frame.to_crs(target_crs)
            frame = frame.rename(columns={'width_source': 'width_src'})[
                ['osm_id', 'road_type', 'name', 'width_m', 'width_src', 'bridge', 'tunnel', 'layer', 'geometry']]
            frame['map_prior'] = 'OSM, NOT model-only'
            with tempfile.TemporaryDirectory(prefix='map_road_export_', dir=directory) as temporary:
                folder = Path(temporary)
                path = folder / 'map_assisted_main_roads.shp'
                frame.to_file(path, driver='ESRI Shapefile', engine='pyogrio', encoding='UTF-8')
                restored = gpd.read_file(path, engine='pyogrio')
                if len(restored) != len(frame) or restored.crs != frame.crs or restored.geometry.is_empty.any():
                    raise ValueError('Map-network shapefile failed its CRS/geometry round-trip check')
                if not all(path.with_suffix(suffix).is_file() for suffix in ('.shp', '.shx', '.dbf', '.prj')):
                    raise ValueError('Map-network shapefile sidecars are incomplete')
                (folder / 'SOURCE_AND_LIMITATIONS.txt').write_text(
                    ATTRIBUTION+'\n'+LICENSE_URL+'\n'+summary['warning']+'\n'+summary['width_warning']+
                    '\nThis is an OSM-derived map database, not a model prediction or surveyed road surface.\n'+
                    'Data license: https://opendatacommons.org/licenses/odbl/1-0/\n'+
                    'Network coordinates are in the original classification GeoTIFF CRS.\n')
                archive = folder / destination.name
                with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
                    for artifact in folder.iterdir():
                        if artifact != archive:
                            zipped.write(artifact, artifact.name)
                archive.replace(destination)
        url = f'/results/{identifier}/{destination.name}'
        summary['artifacts']['map_assisted_shapefile_url'] = url
        summary['shapefile_validation'] = 'CRS, feature count, nonempty geometry and sidecars verified'
        summary_path.write_text(json.dumps(summary, indent=2))
        return {'map_assisted_shapefile_url': url, 'model_class_shapefiles_changed': False}
