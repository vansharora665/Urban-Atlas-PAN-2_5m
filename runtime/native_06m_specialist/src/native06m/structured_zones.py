"""Learned neighbourhood context and image-guided region-label inference.

The frozen semantic network supplies evidence, not pre-decided pixel labels.
Roads are an independent, immutable input. This maps dominant land-cover zones,
not individual buildings, and does not establish ground-truth accuracy.
"""
from dataclasses import asdict, dataclass
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn

IDS = np.array([0, 2, 3, 4], dtype=np.uint8)


@dataclass(frozen=True)
class ZoneOptions:
    factor: int = 4
    region_width_m: float = 60.
    compactness: float = .14
    image_sigma_m: float = 6.
    pairwise_weight: float = 1.6
    iterations: int = 24
    head_weight: float = 0.
    minimum_zone_m2: float = 5000.


class ContextHead(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, 48), nn.ReLU(), nn.Dropout(.15),
                                    nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 4))

    def forward(self, x):
        return self.layers(x)


class UrbanZoneHead(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, 24), nn.ReLU(), nn.Dropout(.15), nn.Linear(24, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def slic_regions(image, valid, roads, gsd, width_m=60., compactness=.14):
    deps = Path(__file__).resolve().parents[2] / '.structured_deps'
    if str(deps) not in sys.path:
        sys.path.append(str(deps))
    from skimage.segmentation import slic
    from skimage.measure import label
    domain = valid & ~roads
    if not domain.any():
        return np.zeros(valid.shape, np.int32)
    # Smoothing guides boundaries at block scale; it is not a class-mask filter.
    guide = np.stack([masked_filter(c, valid, 6. / gsd, gaussian=True) for c in image], -1)
    guide = np.nan_to_num(guide).astype(np.float32)
    segments = max(2, round(domain.size * gsd * gsd / width_m ** 2))
    regions = slic(guide, n_segments=segments, compactness=compactness, sigma=0,
                channel_axis=-1, convert2lab=False, enforce_connectivity=True,
                min_size_factor=.35, max_size_factor=2.5, start_label=1).astype(np.int32)
    # Dense seeds avoid the quadratic masked-centroid initialisation on large scenes.
    # Split each region after excluding roads so a label cannot cross that barrier.
    regions[~domain] = 0
    return label(regions, background=0, connectivity=1).astype(np.int32)


def masked_filter(channel, valid, scale, gaussian=False):
    weight = valid.astype(np.float32)
    if gaussian:
        numerator = ndi.gaussian_filter(channel * weight, max(.5, scale), mode='nearest')
        denominator = ndi.gaussian_filter(weight, max(.5, scale), mode='nearest')
    else:
        size = max(1, int(round(scale)) | 1)
        numerator = ndi.uniform_filter(channel * weight, size, mode='nearest')
        denominator = ndi.uniform_filter(weight, size, mode='nearest')
    return numerator / np.maximum(denominator, 1e-6)


def region_features(image, posterior, valid, regions, gsd):
    """Same feature contract for reviewed tiles and full scenes; bounded planes."""
    number = int(regions.max()) + 1
    flat = regions.ravel()
    counts = np.bincount(flat, minlength=number)
    features, names = [], []

    def pool(plane, name):
        features.append(np.bincount(flat, weights=plane.ravel(), minlength=number) / np.maximum(counts, 1))
        names.append(name)

    for b, channel in enumerate(image):
        pool(channel, f'band{b}_mean')
        for metres in (24., 60.):
            pool(masked_filter(channel, valid, metres / gsd), f'band{b}_context{metres:g}')
        mean = masked_filter(channel, valid, 12. / gsd)
        variance = masked_filter(channel ** 2, valid, 12. / gsd) - mean ** 2
        pool(np.sqrt(np.maximum(variance, 0)), f'band{b}_texture12')
    # A learned contrast on normalized DNs, explicitly not physical NDVI.
    pool((image[3] - image[2]) / (image[3] + image[2] + .15), 'normalized_nir_red_contrast')
    for k in range(5):
        pool(posterior[k], f'neural_p{k}')
        pool(masked_filter(posterior[k], valid, 60. / gsd), f'neural_context60_p{k}')
    for k in (0, 2, 4):
        pool((posterior[k] >= .60).astype(np.float32), f'neural_confident_fraction_p{k}')
    return np.stack(features, 1).astype(np.float32), names, counts


def adjacency(regions, image):
    pairs = []
    for a, b in ((regions[:, :-1], regions[:, 1:]), (regions[:-1], regions[1:])):
        keep = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.sort(np.stack([a[keep], b[keep]], 1), axis=1))
    pairs, boundary = np.unique(np.concatenate(pairs), axis=0, return_counts=True)
    n = int(regions.max()) + 1
    counts = np.bincount(regions.ravel(), minlength=n)
    means = np.stack([np.bincount(regions.ravel(), weights=c.ravel(), minlength=n) /
                      np.maximum(counts, 1) for c in image], 1)
    difference = np.linalg.norm(means[pairs[:, 0]] - means[pairs[:, 1]], axis=1)
    scale = max(.04, float(np.median(difference))) if len(difference) else .04
    weight = np.exp(-.5 * (difference / scale) ** 2)
    weight *= np.minimum(1., boundary / np.maximum(1., np.sqrt(np.minimum(counts[pairs[:, 0]], counts[pairs[:, 1]]))))
    return pairs, weight


def graph_labels(unary, pairs, weight, strength=1.6, iterations=24, anchors=None):
    """Sequential ICM: every accepted update decreases the declared Potts energy."""
    n = len(unary)
    neighbours = [[] for _ in range(n)]
    for (a, b), w in zip(pairs, weight * strength):
        neighbours[a].append((b, w)); neighbours[b].append((a, w))
    labels = unary.argmin(1)
    anchors = np.zeros(n, bool) if anchors is None else anchors

    def energy():
        return float(unary[np.arange(n), labels].sum() +
                     strength * (weight * (labels[pairs[:, 0]] != labels[pairs[:, 1]])).sum())

    history = [energy()]
    for iteration in range(iterations):
        changes = 0
        for node in range(1, n):
            if anchors[node]:
                continue
            costs = unary[node].copy()
            for other, w in neighbours[node]:
                costs += w * (np.arange(4) != labels[other])
            best = costs.argmin()
            if costs[best] < costs[labels[node]] - 1e-7:
                labels[node] = best; changes += 1
        history.append(energy())
        if not changes:
            break
    if np.any(np.diff(history) > 1e-4):
        raise RuntimeError('Structured energy increased')
    return labels, history


def infer_regions(image, posterior, valid, roads, gsd, checkpoint, options=ZoneOptions(), urban_checkpoint=None):
    regions = slic_regions(image, valid, roads, gsd, options.region_width_m, options.compactness)
    features, names, counts = region_features(image, posterior, valid, regions, gsd)
    # The rejected four-class candidate remains an audit artifact, not a dependency
    # on its older feature schema when its contribution is explicitly disabled.
    if options.head_weight and names != checkpoint['feature_names']:
        raise ValueError('Structured feature schema mismatch')
    indices = [names.index(f'neural_p{k}') for k in IDS]
    neural = features[:, indices].clip(1e-5)
    neural /= neural.sum(1, keepdims=True)
    probability = neural.copy()
    if options.head_weight:
        model = ContextHead(len(names))
        model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
        normalized = (features - np.array(checkpoint['mean'])) / np.array(checkpoint['std'])
        with torch.inference_mode():
            learned = torch.softmax(model(torch.from_numpy(normalized.astype(np.float32))), 1).numpy()
        probability = (1 - options.head_weight) * neural + options.head_weight * learned
    urban_promoted = 0
    if urban_checkpoint and urban_checkpoint.get('deployment_accepted'):
        if urban_checkpoint['feature_names'] != names:
            raise ValueError('Urban-zone feature schema mismatch')
        head = UrbanZoneHead(len(names)); head.load_state_dict(urban_checkpoint['state_dict']); head.eval()
        xx = (features - np.array(urban_checkpoint['mean'])) / np.array(urban_checkpoint['std'])
        with torch.inference_mode():
            score = torch.sigmoid(head(torch.from_numpy(xx.astype('float32')))).numpy()
        seed_fraction = features[:, names.index('neural_confident_fraction_p2')]
        promote = (score >= .60) & (seed_fraction >= .08) & (neural[:, 0] < .50)
        # Only the urban-context decision is learned here. Other classes retain
        # the frozen network's conditional distribution and major-region anchors.
        urban = np.maximum(probability[promote, 1], score[promote])
        other = probability[promote][:, [0, 2, 3]]
        other *= ((1 - urban) / np.maximum(other.sum(1), 1e-6))[:, None]
        probability[np.ix_(promote, [0, 2, 3])] = other
        probability[promote, 1] = urban
        urban_promoted = int(promote.sum())
    unary = -np.log(probability.clip(1e-6))
    pairs, weight = adjacency(regions, image)
    # Protect strong, substantial water/vegetation evidence, never tiny colour flecks.
    anchors = np.zeros(len(counts), bool)
    for k in (0, 3):
        confident = (neural[:, k] > .90) & (counts * gsd ** 2 >= options.minimum_zone_m2)
        unary[confident, k] = 0
        anchors |= confident
    labels, energies = graph_labels(unary, pairs, weight, options.pairwise_weight, options.iterations, anchors)
    labels, merged = enforce_minimum_zone(labels, counts * gsd ** 2, pairs, unary, anchors,
                                           options.minimum_zone_m2)
    output = IDS[labels[regions]]
    output[~valid] = 255
    output[roads & valid] = 1
    return output, regions, {'options': asdict(options), 'regions': int(regions.max()),
        'energy_history': energies, 'anchored_regions': int(anchors.sum()),
        'minimum_area_region_merges': merged,
        'urban_context_promoted_regions': urban_promoted,
        'mapping_target': 'Dominant neighbourhood/block land-cover zones; not individual roofs or crowns',
        'region_label_consistency': 'One shared non-road label per image-guided region',
        'accuracy_claim': None}, probability


def enforce_minimum_zone(labels, areas, pairs, unary, anchors, minimum):
    """Region-level minimum mapping unit, not a pixel majority/sieve operation.

    Merge entire connected region groups into the least-cost adjacent zone.
    Strong water/vegetation anchors and isolated domain islands are exceptions.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    labels = labels.copy()
    merged = 0
    for _ in range(12):
        same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
        a, b = pairs[same].T
        graph = coo_matrix((np.ones(len(a) * 2), (np.r_[a, b], np.r_[b, a])), shape=(len(labels), len(labels)))
        _, component = connected_components(graph, directed=False)
        size = np.bincount(component, weights=areas)
        protected = np.bincount(component, weights=anchors) > 0
        small = np.flatnonzero((size < minimum) & ~protected & (size > 0))
        changes = 0
        for group in small[np.argsort(size[small])]:
            members = np.flatnonzero(component == group)
            members = members[members != 0]
            if not len(members):
                continue
            left, right = component[pairs[:, 0]] == group, component[pairs[:, 1]] == group
            neighbours = np.r_[pairs[left & ~right, 1], pairs[right & ~left, 0]]
            candidates = np.unique(labels[neighbours])
            candidates = candidates[candidates != labels[members[0]]]
            if not len(candidates):
                continue
            costs = np.average(unary[members], weights=areas[members], axis=0)
            best = candidates[np.argmin(costs[candidates])]
            labels[members] = best; merged += len(members); changes += 1
        if not changes:
            break
    return labels, merged


class CoarsePosteriorWriter:
    """Validity-weighted streaming reduction, including partial final blocks."""
    def __init__(self, output, native_width, factor=4):
        self.output, self.width, self.factor = output, native_width, factor
        self.next_y = self.written = 0
        self.pending_p = self.pending_v = None

    def __call__(self, y, posterior, valid):
        if y != self.next_y or posterior.shape[1:] != valid.shape or valid.shape[1] != self.width:
            raise ValueError('Non-contiguous posterior rows')
        self.next_y += valid.shape[0]
        p = posterior * valid[None]
        v = valid.astype(np.float32)
        if self.pending_p is not None:
            p = np.concatenate([self.pending_p, p], 1)
            v = np.concatenate([self.pending_v, v], 0)
        ready = p.shape[1] // self.factor * self.factor
        if ready:
            self._write(p[:, :ready], v[:ready])
        self.pending_p, self.pending_v = p[:, ready:].copy(), v[ready:].copy()

    def _write(self, p, v):
        from rasterio.windows import Window
        f = self.factor
        pad_h, pad_w = (-v.shape[0]) % f, (-v.shape[1]) % f
        p = np.pad(p, ((0, 0), (0, pad_h), (0, pad_w)))
        v = np.pad(v, ((0, pad_h), (0, pad_w)))
        h, w = v.shape[0] // f, v.shape[1] // f
        denominator = v.reshape(h, f, w, f).sum((1, 3))
        means = p.reshape(5, h, f, w, f).sum((2, 4)) / np.maximum(denominator, 1)[None]
        encoded = np.concatenate([np.round(means * 65535).clip(0, 65535).astype('uint16'),
                                  denominator[None].astype('uint16')])
        self.output.write(encoded, window=Window(0, self.written, w, h))
        self.written += h

    def finish(self, height):
        if self.next_y != height:
            raise ValueError('Incomplete probability coverage')
        if self.pending_v is not None and self.pending_v.size:
            self._write(self.pending_p, self.pending_v)
        if self.written != (height + self.factor - 1) // self.factor:
            raise ValueError('Incorrect coarse probability height')
