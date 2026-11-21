"""Wall time and CPU-package energy (Intel RAPL via powercap sysfs) for a block of work.

RAPL counts the whole package (all cores, including other processes), so measurements are taken with the
machine otherwise idle and the idle baseline power is reported next to them. The counter wraps at
max_energy_range_uj; one wrap per block is handled.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

RAPL = Path("/sys/class/powercap/intel-rapl:0")


def _read_uj() -> int | None:
    try:
        return int((RAPL / "energy_uj").read_text())
    except (OSError, ValueError):
        return None


def _max_uj() -> int:
    try:
        return int((RAPL / "max_energy_range_uj").read_text())
    except (OSError, ValueError):
        return 2**32


@dataclass
class Measurement:
    seconds: float = 0.0
    joules: float | None = None


@contextmanager
def measure() -> Iterator[Measurement]:
    m = Measurement()
    e0, t0 = _read_uj(), time.perf_counter()
    try:
        yield m
    finally:
        m.seconds = time.perf_counter() - t0
        e1 = _read_uj()
        if e0 is not None and e1 is not None:
            d = e1 - e0 if e1 >= e0 else e1 + _max_uj() - e0
            m.joules = d / 1e6


def idle_power(seconds: float = 5.0) -> float | None:
    with measure() as m:
        time.sleep(seconds)
    return None if m.joules is None else m.joules / m.seconds
