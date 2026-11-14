"""First-party Hermes observability integrations."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _features() -> tuple[Any, ...]:
    from . import command_telemetry, relay_shared_metrics

    return relay_shared_metrics, command_telemetry


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Dispatch a Hermes lifecycle event to built-in observability features (each one isolated)."""
    for feature in _features():
        try:
            feature.observe_lifecycle(hook_name, **kwargs)
        except Exception:
            logger.warning("Built-in observability hook failed: %s (%s)", hook_name, feature.__name__, exc_info=True)


def handles_hook(hook_name: str) -> bool:
    """Return whether any built-in observability feature handles a hook."""
    return any(feature.handles_hook(hook_name) for feature in _features())
