"""Read locally indexed, attributed OSM regional extracts with no network call."""

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from shapely.geometry import box, shape
from shapely.ops import unary_union


@lru_cache(maxsize=8)
def verified(path, size, modified, expected):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('Offline road index checksum mismatch; model output is unchanged')
    return True


def read_offline(bounds, directory):
    manifest_path = Path(directory) / 'regional_networks.json'
    if not manifest_path.is_file():
        return None
    import geopandas as gpd

    manifest = json.loads(manifest_path.read_text())
    region = box(*bounds)
    selected, coverage, sources, seen = [], [], [], set()
    for entry in manifest['extracts']:
        footprint = shape(entry['coverage_wgs84'])
        if not footprint.intersects(region):
            continue
        name = entry['filename']
        if Path(name).name != name:
            raise ValueError('Invalid offline network filename')
        path = Path(directory) / name
        stat = path.stat()
        verified(str(path), stat.st_size, stat.st_mtime_ns, entry['sha256'])
        frame = gpd.read_file(path, layer='main_roads', bbox=bounds, engine='pyogrio')
        coverage.append(footprint)
        sources.append({key: entry[key] for key in ('source_url', 'snapshot_date', 'sha256')})
        for _, row in frame.iterrows():
            geometry = row.geometry
            parts = list(geometry.geoms) if geometry.geom_type == 'MultiLineString' else [geometry]
            for part_index, part in enumerate(parts):
                key = (str(row['osm_id']), part_index)
                if key in seen or part.geom_type != 'LineString':
                    continue
                seen.add(key)
                tags = {key: str(row[key]) for key in ('highway', 'name', 'width', 'lanes', 'bridge', 'tunnel', 'layer')
                        if key in row and row[key] is not None and str(row[key]) not in ('nan', '')}
                selected.append({'type': 'way', 'id': int(row['osm_id']), 'tags': tags,
                                 'geometry': [{'lon': float(x), 'lat': float(y)} for x, y in part.coords]})
    if not selected:
        return None
    data = {'elements': selected, 'osm3s': {'timestamp_osm_base': manifest['snapshot_date'],
                                           'copyright': 'OpenStreetMap contributors, ODbL 1.0'}}
    complete = unary_union(coverage).covers(region)
    return data, {'provider': 'Local Geofabrik regional OSM index', 'sources': sources,
                  'queried_bounds_wgs84': list(bounds), 'requested_bounds_wgs84': list(bounds),
                  'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                  'osm_database_timestamp': manifest['snapshot_date'], 'offline': True,
                  'cache_hit': True, 'satellite_pixels_uploaded': False, 'geographic_bounds_transmitted': False,
                  'extract_footprint_covers_entire_scene': bool(complete),
                  'coverage_warning': None if complete else 'The scene intersects an extract boundary; unmapped area is not verified as road-free.',
                  'attribution': 'OpenStreetMap contributors; data under ODbL 1.0',
                  'license_url': 'https://www.openstreetmap.org/copyright'}
