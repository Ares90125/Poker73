"""Participant-owned model package for the Poker44 miner (uid73).

Bot detector = gradient-boosted decision trees over behavioral features. See
detector.py (inference), features.py (feature extraction), train_model.py
(reproducible training), and model.json (the tree ensemble).
"""

from poker44_model.detector import score_chunk

__all__ = ["score_chunk"]
