"""Assignment-aligned adaptive (KT) surface.

This module exposes the adaptive tutoring interfaces from `tutor.engine`.
"""

from tutor.engine import Tutor, grade, select_next_item, update_state

__all__ = ["Tutor", "grade", "select_next_item", "update_state"]
