"""Shared fixtures: the small frozen surrogate that ships with the repository."""

from __future__ import annotations

import os as _os
import pathlib as _pathlib

_SRC = str((_pathlib.Path(__file__).resolve().parent.parent / "src"))
# Tests that spawn a clean interpreter (process-independence, PYTHONHASHSEED,
# legacy alias) inherit the environment, so the package has to be on PYTHONPATH
# there as well -- pytest's own `pythonpath` setting only affects this process.
_prev = _os.environ.get("PYTHONPATH", "")
if _SRC not in _prev.split(_os.pathsep):
    _os.environ["PYTHONPATH"] = _SRC + ((_os.pathsep + _prev) if _prev else "")


import json
import pathlib

import numpy as np
import pytest

DATA = pathlib.Path(__file__).parent / "data"
MODEL = DATA / "surrogate_tiny.npz"
PROBES = DATA / "probes_tiny.jsonl"
EXPECTED = DATA / "expected_tiny.npy"
DECIMALS = 6


@pytest.fixture(scope="session")
def model_path() -> pathlib.Path:
    if not MODEL.exists():
        pytest.skip(f"{MODEL} missing, build it with examples/build_tiny_fixture.py")
    return MODEL


@pytest.fixture(scope="session")
def probes() -> list[str]:
    with PROBES.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


@pytest.fixture(scope="session")
def expected() -> np.ndarray:
    return np.load(EXPECTED)


@pytest.fixture(scope="session")
def frozen(model_path):
    from embsurrogate.predict import FrozenSurrogate
    return FrozenSurrogate(model_path)
