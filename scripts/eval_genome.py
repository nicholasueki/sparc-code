#!/usr/bin/env python3
"""Run generation 0 as a designed experiment — no evolution, no mutation.

    scripts/eval_genome.py                       # the 10-cell factorial
    scripts/eval_genome.py --distill compound atomic   # 20 cells, G3 varied too
    scripts/eval_genome.py --genome path/to/x.yaml     # one genome
    scripts/eval_genome.py --holdout                   # score against held-out probes

This is step 4 of the build order and worth doing even if the evolution loop is
never run: gen 0 is a factorial over world closure, epistemic marking, structure,
and persona, so its main effects are readable on their own. The --effects table at
the end is the actual deliverable — it says which axis is doing the work.

Runs on Node C (local Mac python lacks the sqlite extensions the mirror needs).
Isolated temp stores throughout; production world.db is never touched.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _p in ("common", "node_a", "node_c"):
    sys.path.insert(0, str(ROOT / "packages" / _p))

from evolution import fitness as fit  # noqa: E402
from evolution import seeds, suite  # noqa: E402
from evolution.evolve import rescore  # noqa: E402
from evolution.genome import Genome  # noqa: E402
from evolution.llm import LLMClient  # noqa: E402
from evolution.runner import CortexClient, RunResult, run_genome  # noqa: E402
from sparc_common import config  # noqa: E402

log = logging.getLogger("eval_genome")


def axes_of(g: Genome) -> dict[str, str]:
    """Recover the factorial cell from gene values, so main effects can be pooled
    without polluting the Genome schema with experiment metadata."""
    header = {seeds.HEADER_OPEN: "open", seeds.HEADER_CLOSED: "closed",
              seeds.HEADER_CLOSED_DEMO: "closed+demo"}.get(g.briefing_header, "custom")
    persona = {seeds.PERSONA_CONTROL: "control", seeds.PERSONA_HUMBLE: "humble",
               seeds.PERSONA_CONFIDENT: "confident"}.get(g.persona_memory, "custom")
    distill = {seeds.DISTILL_COMPOUND: "compound",
               seeds.DISTILL_ATOMIC: "atomic"}.get(g.distill_system, "custom")
    ff = g.fact_format
    marking = ("provenance" if ff.get("provenance") else
               "confidence" if ff.get("confidence_marks") else "bare")
    if ff.get("provenance") and ff.get("confidence_marks"):
        marking = "both"
    return {
        "closure": header,
        "marking": marking,
        "structure": "delimited" if ff.get("separator") != " " else "prose",
        "persona": persona,
        "distiller": distill,
    }


def effects_table(results: dict[str, RunResult], genomes: list[Genome]) -> str:
    """Per-axis mean accuracy and F3 refusal rate — the readable payoff of a
    factorial design. With one cell per level this is descriptive, not inferential:
    treat it as 'which axis is worth a follow-up', not as a significance test."""
    by_axis: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list))
    for g in genomes:
        r = results.get(g.id)
        if r is None:
            continue
        f3 = r.agg.family_rates.get("F3", 0.0)
        f6 = r.agg.family_rates.get("F6", 0.0)
        for axis, level in axes_of(g).items():
            by_axis[axis][level].append((r.components.A, f3, f6))

    out = ["", "MAIN EFFECTS  (mean over cells at each level)",
           f"  {'axis / level':<26} {'n':>2}  {'acc':>6} {'F3 refuse':>10} {'F6 load':>8}"]
    for axis, levels in by_axis.items():
        if len(levels) < 2:
            continue
        out.append(f"  {axis}")
        for level, vals in sorted(levels.items()):
            A = statistics.mean(v[0] for v in vals)
            f3 = statistics.mean(v[1] for v in vals)
            f6 = statistics.mean(v[2] for v in vals)
            out.append(f"    {level:<24} {len(vals):>2}  {A:>6.3f} {f3:>10.3f} {f6:>8.3f}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", action="append", default=[],
                    help="path to a genome YAML; repeatable. Default: the gen-0 seeds")
    ap.add_argument("--distill", nargs="+", default=["compound"],
                    choices=["compound", "atomic"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--cortex", default=None)
    ap.add_argument("--holdout", action="store_true",
                    help="score against the held-out probes instead of the dev set")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "evolution_runs" / "gen0.json"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-12s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ecfg = config.get("evolution", {})
    genomes = ([Genome.load(p) for p in args.genome] if args.genome
               else seeds.build(tuple(args.distill)))
    reps = args.reps or int(ecfg.get("reps", 3))
    cortex = CortexClient(args.cortex or ecfg.get("cortex_url"))
    judge = None
    if not args.no_judge:
        rc = ecfg.get("researcher") or {}
        if rc.get("base_url") and rc.get("judge_model"):
            judge = LLMClient(model=rc["judge_model"], base_url=rc["base_url"],
                              api_key_env=rc.get("api_key_env", ""),
                              timeout=float(rc.get("timeout_s", 600)),
                              max_concurrency=int(rc.get("max_concurrency", 4)),
                              temperature=0.0)
        else:
            log.warning("no researcher.base_url/judge_model configured; "
                        "grading programmatically only")

    probes = suite.probes(holdout=args.holdout)
    captures = suite.capture_probes(holdout=args.holdout)
    log.info("%s", suite.summary())
    log.info("running %d genome(s) x %d probes x %d reps = %d think calls",
             len(genomes), len(probes), reps, len(genomes) * len(probes) * reps)

    results: dict[str, RunResult] = {}
    try:
        for g in genomes:
            r = run_genome(g, probes, captures, cortex=cortex,
                           embed_model=ecfg.get("embed_model"), reps=reps, judge=judge)
            results[g.id] = r
            log.info("%-24s %s", g.id, fit.explain(r.components))

        ctrl = seeds.control_id(tuple(args.distill))
        ref = results[ctrl].components.density_rate if ctrl in results else None
        if ref:
            for r in results.values():
                rescore(r, ref)
            log.info("density anchored on %s (%.4f facts/token)", ctrl, ref)
    finally:
        cortex.close()
        if judge:
            judge.close()

    order = sorted(results, key=lambda k: results[k].fitness, reverse=True)
    print(f"\n{'genome':<24} {'fitness':>10}  {'A':>5} {'R':>5} {'D':>5} {'L':>5}"
          f"  {'F3':>5} {'F6':>5}  gates")
    for gid in order:
        r = results[gid]
        c = r.components
        print(f"{gid:<24} {r.fitness:>10,}  {c.A:>5.3f} {c.R:>5.3f} {c.D:>5.3f} "
              f"{c.L:>5.3f}  {r.agg.family_rates.get('F3', 0):>5.2f} "
              f"{r.agg.family_rates.get('F6', 0):>5.2f}  {','.join(c.gates) or '-'}")

    if len(genomes) > 1:
        print(effects_table(results, genomes))
        neg = next((g for g in genomes if axes_of(g)["persona"] == "confident"), None)
        ctl = next((g for g in genomes if g.id == seeds.control_id(tuple(args.distill))), None)
        if neg and ctl and neg.id in results and ctl.id in results:
            d = (results[ctl.id].agg.family_rates.get("F3", 0)
                 - results[neg.id].agg.family_rates.get("F3", 0))
            print(f"\nnegative control: memory-confident persona changes F3 refusal by "
                  f"{-d:+.3f}")
            if abs(d) < 0.05:
                print("  -> G0 is NOT load-bearing on refusal. Freeze the persona gene "
                      "and spend the search budget on G1/G3/G4 instead.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        gid: {"fitness": r.fitness, "axes": axes_of(next(g for g in genomes if g.id == gid)),
              "components": {k: v for k, v in vars(r.components).items()},
              "family_rates": r.agg.family_rates, "pass_rates": r.agg.pass_rates,
              "capture": r.capture, "proposals": r.proposals,
              "failures": [{"probe": o.probe_id, "family": o.family, "gates": o.gates,
                            "said": o.text, "briefing": o.briefing, "why": o.reasons}
                           for rep in r.outcomes for o in rep if not o.passed]}
        for gid, r in results.items()}, indent=1))
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
