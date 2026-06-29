"""Poker44 bot detector — uid73 (Ares90125/Poker73).

Model: **gradient-boosted decision trees** over winsorized behavioral features,
scored by **within-batch ranking**. Robust to the benchmark-vs-live distribution
shift: winsorizing bounds out-of-distribution live features, and ranking the
batch is exactly what the validator's AP reward optimizes. Trees in model.json
(reproducible via train_model.py). Pure-Python tree walker at inference.

`score_batch(chunks)` is the real path; it returns one rank-based bot-risk score
in [0,1] per chunk (higher = more bot-like).
"""
from __future__ import annotations

import json
import math
import os

from poker44_model.features import extract_group_features

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        with open(os.path.join(os.path.dirname(__file__), "model.json")) as fh:
            _MODEL = json.load(fh)
    return _MODEL


def _tree_value(tree, x):
    i = 0
    f, thr, left, right = tree["f"], tree["thr"], tree["l"], tree["r"]
    while f[i] >= 0:
        i = left[i] if x[f[i]] <= thr[i] else right[i]
    return tree["v"][i]


def _raw_decision(m, feats):
    """Raw GBM score on winsorized features (ordered)."""
    x = [min(max(feats.get(name, 0.0), m["winsor_lo"][i]), m["winsor_hi"][i])
         for i, name in enumerate(m["features"])]
    return m["F0"] + m["lr"] * sum(_tree_value(t, x) for t in m["trees"])


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, ranked within the batch."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        raws = [_raw_decision(m, extract_group_features(c)) for c in chunks]
        return _rank_normalize(raws)
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback (bounded sigmoid). The batch path is score_batch."""
    try:
        if not chunk:
            return 0.5
        z = _raw_decision(_model(), extract_group_features(chunk))
        return round(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))), 6)
    except Exception:
        return 0.5
