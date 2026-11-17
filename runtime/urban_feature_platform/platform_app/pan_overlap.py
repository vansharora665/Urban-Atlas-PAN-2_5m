"""Bounded-memory probability blending on a fixed global PAN patch lattice."""

import numpy as np


def positions(length, window, stride):
    values = list(range(0, max(1, length-window+1), stride))
    last = max(0, length-window)
    if values[-1] != last:
        values.append(last)
    return values


def blend_probabilities(width, height, classes, read_patch, predict, emit,
                        window=256, stride=192, batch_size=32, progress=None):
    """Emit each row exactly once after all overlapping contexts are accumulated."""
    if not 0 < stride <= window:
        raise ValueError("Stride must be positive and no larger than a patch")
    xs, ys = positions(width, window, stride), positions(height, window, stride)
    taper = np.maximum(np.hanning(window), .05).astype(np.float32)
    kernel = np.outer(taper, taper)
    probability_sum = np.zeros((classes, window, width), np.float32)
    weight_sum = np.zeros((window, width), np.float32)
    origin = 0
    for row, y in enumerate(ys):
        delta = y-origin
        if delta:
            emit(origin, probability_sum[:, :delta]/np.maximum(weight_sum[:delta], 1e-8), weight_sum[:delta]>0)
            probability_sum[:, :window-delta] = probability_sum[:, delta:].copy()
            probability_sum[:, window-delta:] = 0
            weight_sum[:window-delta] = weight_sum[delta:].copy()
            weight_sum[window-delta:] = 0
            origin = y
        for start in range(0,len(xs),batch_size):
            local_xs = xs[start:start+batch_size]
            patches, masks = zip(*(read_patch(x,y) for x in local_xs))
            probabilities = predict(np.stack(patches))
            for x, probability, mask in zip(local_xs, probabilities, masks):
                h, w = min(window,height-y), min(window,width-x)
                weights = kernel[:h,:w]*mask[:h,:w]
                probability_sum[:,:h,x:x+w] += probability[:,:h,:w]*weights
                weight_sum[:h,x:x+w] += weights
        if progress:
            progress((row+1)/len(ys))
    remaining = height-origin
    emit(origin, probability_sum[:,:remaining]/np.maximum(weight_sum[:remaining],1e-8), weight_sum[:remaining]>0)
