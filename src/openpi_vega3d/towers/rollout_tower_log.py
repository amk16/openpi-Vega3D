"""Emit generative-tower diagnostics on the ``run_rollout`` logger.

OmniGibson/Kit raises the root logger to WARNING after startup; this module uses
WARNING so lines still appear in rollout logs. Handlers on ``run_rollout`` (see
``scripts/run_rollout.configure_logging``) receive these records even when root
filters INFO.
"""

from __future__ import annotations

import logging
import sys

_rollout = logging.getLogger("run_rollout")


def _flush_handlers(log: logging.Logger) -> None:
    for h in log.handlers:
        flush = getattr(h, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass


def log_tower(msg: str, *args) -> None:
    """Log one tower trace line and flush (best-effort for segfault/teardown)."""
    _rollout.warning("[trace tower] " + msg, *args)
    _flush_handlers(logging.root)
    _flush_handlers(_rollout)
    sys.stdout.flush()
    sys.stderr.flush()
