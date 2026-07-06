import os as _os
import warnings as _warnings
from importlib import import_module as _import_module

# Quiet third-party import-time deprecation noise (e.g. resampy -> pkg_resources,
# numba/librosa deprecations) so it doesn't clutter benchmark output. Installed
# BEFORE the submodule imports below, which transitively pull in librosa/resampy/
# crepe -- so the filters are in effect by the time those emit at import time.
_warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# TensorFlow (SPICE / crepe) logs a flood through C++ + Python logging, NOT the
# warnings module. The C++ side is gated by this env var, which TF only reads at
# import time -- so set it here, before anything pulls TF in. The Python-side
# logger flood is silenced per-tracker via _quiet_tensorflow() in the benchmarker.
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all .. 3=errors only

_EXPORTS = {
    "PitchBenchmarker": ("benchmarks.modules.pitch.PitchBenchmarker", "PitchBenchmarker"),
    "CocoChoralesBenchmarker": (
        "benchmarks.modules.CocoChoralesBenchmarker",
        "CocoChoralesBenchmarker",
    ),
    "NoteBenchmarker": ("benchmarks.note.NoteBenchmarker", "NoteBenchmarker"),
    "CocoNoteBenchmarker": ("benchmarks.note.CocoNoteBenchmarker", "CocoNoteBenchmarker"),
    "RupturesDetector": (
        "benchmarks.note.algorithms.RupturesDetector",
        "RupturesDetector",
    ),
    "MistakeBenchmarker": ("benchmarks.modules.mistake.MistakeBenchmarker", "MistakeBenchmarker"),
    "MistakeInjector": ("benchmarks.modules.mistake.MistakeInjector", "MistakeInjector"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _EXPORTS[name]
    value = getattr(_import_module(module_name), attr)
    globals()[name] = value
    return value

__all__ = [
    "PitchBenchmarker",
    "CocoChoralesBenchmarker",
    "NoteBenchmarker",
    "CocoNoteBenchmarker",
    "RupturesDetector",
    "MistakeBenchmarker",
    "MistakeInjector",
]
