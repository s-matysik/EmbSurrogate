"""Deprecated alias of :mod:`embsurrogate`, kept so that frozen models and
scripts written against the original package name keep working.

The package was renamed from ``klon`` to ``embsurrogate`` when it was prepared
for publication; the numerics were not touched. Importing this module emits a
:class:`DeprecationWarning` and forwards every attribute and every submodule to
``embsurrogate``:

    import klon                      # DeprecationWarning
    klon.features is embsurrogate.features        # True
    python3 -m klon.replay_check model/surrogate.npz --probes probes.jsonl

Nothing here re-implements anything. The two modules that carry the numerics
(``features``, ``dual``) are aliased in ``sys.modules`` eagerly, so every other
module, however it is reached, uses those exact objects, including their hash
cache. The remaining modules resolve lazily through ``__getattr__``, which keeps
a numpy-only installation from importing the parts that need scipy or pandas,
and makes ``klon.predict is embsurrogate.predict`` true.

One detail worth knowing: ``python -m klon.replay_check`` and the dotted form
``import klon.predict`` load a fresh module object under the ``klon.*`` name
(that is how ``runpy`` works). Their relative imports resolve to the eager
aliases above, so the arithmetic is the same objects and the same results; only
the wrapper module is duplicated. Attribute access (``klon.predict``,
``from klon import predict``) always returns the ``embsurrogate`` module itself.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import embsurrogate as _embsurrogate

__version__ = _embsurrogate.__version__

# Submodule search path of the real package: makes `python -m klon.<module>`
# and `import klon.<module>` resolve to the same files.
__path__ = list(_embsurrogate.__path__)

# The numpy-only numeric core. Aliased eagerly so that any module loaded under
# the `klon.*` name shares these objects instead of a second copy of them.
# The command-line modules are deliberately NOT eager: a sys.modules entry whose
# spec carries the `embsurrogate.*` name makes `python -m klon.<module>` fail.
_EAGER = ("features", "dual")

# Everything else is reached through __getattr__ (identity preserved) or, for
# the dotted/`-m` forms, through __path__; listed here so `dir(klon)` is useful.
_LAZY = ("predict", "replay_check", "reference", "surrogate", "evaluate",
         "freeze", "probe_table", "run_experiment", "run_primal", "run_big",
         "teacher_sbert", "jadro_iloczynowe", "nauczyciele", "regularyzacja",
         "rysunek")

warnings.warn(
    "`klon` is a deprecated alias for `embsurrogate`; import `embsurrogate` "
    "instead (the numerics are identical, only the package name changed).",
    DeprecationWarning,
    stacklevel=2,
)

for _name in _EAGER:
    _mod = importlib.import_module(f"embsurrogate.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod


def __getattr__(name: str):
    """Forward `klon.X` to `embsurrogate.X`, registering the alias in sys.modules."""
    try:
        mod = importlib.import_module(f"embsurrogate.{name}")
    except ModuleNotFoundError:
        try:
            return getattr(_embsurrogate, name)
        except AttributeError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    sys.modules[f"{__name__}.{name}"] = mod
    globals()[name] = mod
    return mod


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EAGER) | set(_LAZY))


__all__ = list(_EAGER) + list(_LAZY)
