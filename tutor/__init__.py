"""On-device math-tutor package.

Public surface (lazy-imported so individual modules can be built and tested
in isolation during development):

    from tutor import Tutor                       # engine.Tutor
    from tutor import select_next_item, grade, update_state
"""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["Tutor", "select_next_item", "grade", "update_state"]


def __getattr__(name):
    if name in __all__:
        engine = import_module(".engine", __name__)
        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
