"""Fitness: four components, one integer, priority strictly enforced.

Priority is accuracy > reliability > density > latency (docs §2). A weighted sum
cannot enforce that — a full swing in a low-priority term outweighs a real drop in
a high-priority one. Banded lexicographic scoring quantizes each component into
bands and gives each tier more headroom than the tiers below it can ever reach, so
a lower-priority term only ever breaks a tie *within* a band.

Band widths come from measurement noise, not taste: with n trials the standard
error on a pass rate is ~0.5/sqrt(n), so two genomes inside one band are
statistically tied on that component and the next tier down should decide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Latency normalization window, from the 3.6-6.9 s range measured in docs/EVAL.md.
L_FLOOR_MS = 2000.0
L_CEIL_MS = 8000.0

# Derived from the dev suite size, not chosen by taste: 29 dev probes x 3 reps = 87
# trials, so 1 s.e. on a pass rate is ~0.054. A band finer than that would let
# measurement noise outrank reliability. Re-derive with suggested_band() if the
# suite or the repetition count changes — scripts/evolve_memory.py --preflight warns.
BAND_A = 0.05
BAND_R = 0.05
BAND_D = 0.05
BAND_L = 0.05

TIER_A, TIER_R, TIER_D, TIER_L = 1_000_000, 10_000, 100, 1


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def band(x: float, width: float) -> int:
    """Quantize [0,1] into bands. Top value lands in the last band, not past it."""
    return min(int(clamp01(x) / width), int(1.0 / width))


def suggested_band(n_trials: int) -> float:
    """~1 standard error on a pass rate, the resolution floor for this suite size."""
    return 0.5 / math.sqrt(max(1, n_trials))


@dataclass
class Components:
    A: float = 0.0            # accuracy   — mean per-probe pass rate
    R: float = 0.0            # reliability— 1 means every probe is deterministic
    D: float = 0.0            # density    — usable facts per briefing token vs control
    L: float = 0.0            # latency    — 1 is fast
    # diagnostics (never scored, always logged)
    n_probes: int = 0
    n_seeds: int = 0
    density_rate: float = 0.0     # raw facts_used / briefing_token
    precision: float = 0.0        # relevant briefing lines / total briefing lines
    median_ms: float = 0.0
    mean_brief_tokens: float = 0.0
    schema_retries: int = 0       # fallback_level==1: recovered, but a reliability smell
    gates: list[str] = field(default_factory=list)

    @property
    def gated(self) -> bool:
        return bool(self.gates)


def accuracy(pass_rates: list[float]) -> float:
    return sum(pass_rates) / len(pass_rates) if pass_rates else 0.0


def reliability(pass_rates: list[float]) -> float:
    """1 - mean flakiness, where flakiness peaks at p=0.5 and is 0 at p in {0,1}.

    Rewarding consistent *failure* is harmless: accuracy dominates lexicographically,
    so nothing can win by failing reliably — but a genome that is steady at 4/5 does
    correctly outrank one that oscillates between 5/5 and 2/5.
    """
    if not pass_rates:
        return 0.0
    return 1.0 - sum(4.0 * p * (1.0 - p) for p in pass_rates) / len(pass_rates)


def density(facts_used: int, briefing_tokens: int, ref_rate: float | None) -> tuple[float, float]:
    """-> (D, raw_rate). Normalized so 1.0 == as dense as production today.

    Counts briefing tokens only, not the whole prompt: the briefing is what G4
    composes, so charging a genome for its G1 header would conflate two genes.
    """
    rate = facts_used / briefing_tokens if briefing_tokens else 0.0
    if not ref_rate:
        return 0.0, rate
    return clamp01(rate / ref_rate), rate


def latency(median_ms: float) -> float:
    return clamp01((L_CEIL_MS - median_ms) / (L_CEIL_MS - L_FLOOR_MS))


def fitness(c: Components) -> int:
    """Single integer. Gated genomes score 0 and are culled regardless of components."""
    if c.gated:
        return 0
    return (TIER_A * band(c.A, BAND_A)
            + TIER_R * band(c.R, BAND_R)
            + TIER_D * band(c.D, BAND_D)
            + TIER_L * band(c.L, BAND_L))


def explain(c: Components) -> str:
    if c.gated:
        return f"GATED[{','.join(c.gates)}] A={c.A:.2f} R={c.R:.2f} D={c.D:.2f} L={c.L:.2f}"
    return (f"{fitness(c):>9,d}  A={c.A:.3f}({band(c.A, BAND_A):>2}) "
            f"R={c.R:.3f}({band(c.R, BAND_R):>2}) "
            f"D={c.D:.3f}({band(c.D, BAND_D):>2}) "
            f"L={c.L:.3f}({band(c.L, BAND_L):>2})  "
            f"{c.median_ms:.0f}ms {c.mean_brief_tokens:.0f}tok")
