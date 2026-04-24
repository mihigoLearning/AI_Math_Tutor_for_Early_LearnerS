"""Root parent report entrypoint required by the assignment brief.

Re-exports report helpers from `src/parent_report.py` without name collisions.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
_SRC_IMPL = _ROOT / "src" / "parent_report.py"

_spec = spec_from_file_location("_src_parent_report", _SRC_IMPL)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load source module at {_SRC_IMPL}")
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_report = _mod.build_report
render_icon_card = _mod.render_icon_card
write_voiced_summary = _mod.write_voiced_summary
SKILLS = _mod.SKILLS
SUMMARY_TEMPLATE = _mod.SUMMARY_TEMPLATE

__all__ = [
    "build_report",
    "render_icon_card",
    "write_voiced_summary",
    "SKILLS",
    "SUMMARY_TEMPLATE",
]
