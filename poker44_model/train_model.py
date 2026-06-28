"""Reproducible training for the uid73 gradient-boosted-trees detector.

Writes model.json (the tree ensemble + offset). Requires the public benchmark
and sklearn (`pip install -e .`). Usage from the repo root:

    python3 poker44_model/train_model.py --data /root/ares/Poker/train/raw

Trains on the benchmark `train` split. Evaluate on the `validation` split with
the subnet reward metric (0.75*AP + 0.25*recall@FPR<=0.05).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from poker44_model.features import extract_group_features, FEATURE_NAMES


def load(raw, split):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        for rc in json.load(open(f)).get("chunks", []):
            if rc.get("split") != split:
                continue
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l)))
    return out


def export_tree(est):
    t = est.tree_
    return {
        "f": [int(v) for v in t.feature],
        "thr": [round(float(v), 6) for v in t.threshold],
        "l": [int(v) for v in t.children_left],
        "r": [int(v) for v in t.children_right],
        "v": [round(float(t.value[i][0][0]), 6) for i in range(t.node_count)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to train/raw chunk JSON dir")
    args = ap.parse_args()

    tr = load(args.data, "train")
    X = np.array([[extract_group_features(g)[k] for k in FEATURE_NAMES] for g, _ in tr])
    y = np.array([l for _, l in tr])

    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.85, random_state=0,
    ).fit(X, y)

    trees = [export_tree(est) for est in gbm.estimators_[:, 0]]

    def tree_sum(x):
        s = 0.0
        for tr_ in trees:
            i = 0
            while tr_["f"][i] >= 0:
                i = tr_["l"][i] if x[tr_["f"][i]] <= tr_["thr"][i] else tr_["r"][i]
            s += tr_["v"][i]
        return s

    f0 = float(np.mean(gbm.decision_function(X) - gbm.learning_rate
                       * np.array([tree_sum(row) for row in X])))

    model = {
        "type": "gbm",
        "features": FEATURE_NAMES,
        "lr": float(gbm.learning_rate),
        "F0": round(f0, 6),
        "trees": trees,
    }
    out = os.path.join(os.path.dirname(__file__), "model.json")
    json.dump(model, open(out, "w"))
    print(f"wrote {out} ({len(tr)} train examples, {len(trees)} trees)")


if __name__ == "__main__":
    main()
