"""Poker44 bot detector — uid73 (Ares90125/Poker73).

Model: **gradient-boosted decision trees** over behavioral features
(see features.py). A nonlinear ensemble that captures feature interactions the
linear model misses. Trees live in model.json and are reproducible with
train_model.py against the public benchmark. Pure-Python tree walker at
inference — no sklearn needed to serve.

`score_chunk(group)` returns P(bot) in [0, 1] for one chunk group.
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


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _tree_value(tree, x):
    """Walk one regression tree to its leaf value (f<0 marks a leaf)."""
    i = 0
    f, thr, left, right = tree["f"], tree["thr"], tree["l"], tree["r"]
    while f[i] >= 0:
        i = left[i] if x[f[i]] <= thr[i] else right[i]
    return tree["v"][i]


def score_chunk(chunk):
    """One bot-risk score in [0, 1] for a chunk group (list of hand dicts)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        feats = extract_group_features(chunk)
        x = [feats.get(name, 0.0) for name in m["features"]]
        raw = m["F0"] + m["lr"] * sum(_tree_value(t, x) for t in m["trees"])
        return round(_sigmoid(raw), 6)
    except Exception:
        # Never crash the miner: a thrown exception = 0 coverage for the cycle.
        return 0.5
