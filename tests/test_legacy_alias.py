"""`klon` is the pre-publication package name and must keep working.

Frozen models and analysis scripts written before the rename refer to
`klon.features`, `klon.predict` and `python -m klon.replay_check`. The alias
forwards to `embsurrogate`, which is the same module objects, so there is no second
copy of the feature hash cache or of the numerics.
"""

from __future__ import annotations

import subprocess
import sys
import warnings


def test_alias_emits_deprecation_warning_and_forwards_modules():
    for name in ("klon", "klon.features", "klon.dual", "klon.predict",
                 "klon.replay_check"):
        sys.modules.pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import klon
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    import embsurrogate
    assert klon.features is embsurrogate.features
    assert klon.dual is embsurrogate.dual
    assert klon.predict is embsurrogate.predict
    assert klon.replay_check is embsurrogate.replay_check
    assert klon.__version__ == embsurrogate.__version__
    assert klon.features.FEATURE_VERSION == "klon-features-3"


def test_lazy_attribute_forwarding():
    import klon
    from embsurrogate import freeze
    assert klon.freeze is freeze


def test_import_klon_features_from_a_clean_interpreter():
    code = "import klon; print(klon.features.FEATURE_VERSION)"
    out = subprocess.run([sys.executable, "-W", "ignore::DeprecationWarning", "-c", code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "klon-features-3"


def test_run_module_through_the_alias():
    """`python -m klon.replay_check --help` must still start."""
    out = subprocess.run([sys.executable, "-W", "ignore::DeprecationWarning",
                          "-m", "klon.replay_check", "--help"],
                         capture_output=True, text=True, check=True)
    assert "--probes" in out.stdout
