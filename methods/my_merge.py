"""Backward-compatible entry point for LAMP-Merge.

The formal method implementation lives in :mod:`methods.lamp_merge`.  This
module keeps historical ``my_merge`` imports and CLI aliases working.
"""

from .lamp_merge import merge_lamp_merge


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    return merge_lamp_merge(state_dicts, weights, meta=meta, checkpoints=checkpoints, cfg=cfg)
