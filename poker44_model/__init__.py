"""Participant-owned model package for the Poker44 miner.

Build your bot detector in `detector.py`. Keep everything model-related inside
this package so upstream merges into the rest of the repo stay conflict-free.
"""

from poker44_model.detector import clamp01, score_chunk, score_hand

__all__ = ["clamp01", "score_chunk", "score_hand"]
