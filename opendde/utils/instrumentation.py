"""Private, opt-in instrumentation hooks for profiling and qualification.

This module is not a public interface. There is no CLI option, config key or
environment variable that enables it. A harness installs a recorder in-process;
with no recorder installed, every hook is a cheap no-op and returns its input
unchanged, so stock behavior and outputs are unaffected.

Two hooks are provided:

* ``phase(stage_name)`` wraps a model phase. An installed phase recorder can
  synchronize the device at the boundaries so queued GPU work is charged to
  the phase that issued it.
* ``record_draw(site, kind, value, ...)`` observes one random draw at its call
  site, before any scaling is applied. Call sites use stable symbolic names
  (``site``) rather than file and line numbers. The sampler sets the current
  chunk, sample offset and diffusion step with ``set_draw_scope``.
"""

from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Optional

PhaseRecorder = Callable[[str], ContextManager[Any]]
DrawRecorder = Callable[[dict[str, Any], Any], None]

_phase_recorder: Optional[PhaseRecorder] = None
_draw_recorder: Optional[DrawRecorder] = None
_draw_scope: dict[str, Any] = {}


def install_phase_recorder(recorder: Optional[PhaseRecorder]) -> None:
    """Install (or clear, with ``None``) the in-process phase recorder."""
    global _phase_recorder
    _phase_recorder = recorder


def install_draw_recorder(recorder: Optional[DrawRecorder]) -> None:
    """Install (or clear, with ``None``) the in-process random-draw recorder."""
    global _draw_recorder
    _draw_recorder = recorder
    _draw_scope.clear()


def phase(stage_name: str) -> ContextManager[Any]:
    """Return the recorder's context for ``stage_name``, or a null context."""
    if _phase_recorder is None:
        return nullcontext()
    return _phase_recorder(stage_name)


def set_draw_scope(**scope: Any) -> None:
    """Set the sampler scope (chunk, sample offset, step) for later draws."""
    if _draw_recorder is None:
        return
    _draw_scope.clear()
    _draw_scope.update(scope)


def record_draw(site: str, kind: str, value: Any, sample_dim: int, **key: Any) -> Any:
    """Report one raw random draw to the recorder and return it unchanged.

    Args:
        site: stable symbolic name of the RNG call site.
        kind: draw kind, for example ``"gaussian"`` or ``"rotation"``.
        value: the raw draw, captured before any scaling.
        sample_dim: dimension of ``value`` that indexes samples in the chunk.
        **key: extra key fields for draws repeated within one step.
    """
    if _draw_recorder is not None:
        record = dict(_draw_scope)
        record.update(key)
        record.update(site=site, kind=kind, sample_dim=sample_dim)
        _draw_recorder(record, value)
    return value
