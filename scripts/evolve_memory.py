#!/usr/bin/env python3
"""Evolve SPARC's memory prompt stack. See docs/MEMORY_PROMPT_EVOLUTION.md.

    scripts/evolve_memory.py --preflight              # check the rig, spend nothing
    scripts/evolve_memory.py --generations 20

Run this on Node C: the local Mac's python lacks the sqlite extensions the vector
mirror needs (the same constraint scripts/eval_memory.py documents).

All stores are isolated temp copies. The production world.db is never touched.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _p in ("common", "node_a", "node_c"):
    sys.path.insert(0, str(ROOT / "packages" / _p))

from evolution import seeds, suite  # noqa: E402
from evolution.evolve import Evolver  # noqa: E402
from evolution.genome import Genome  # noqa: E402
from evolution.llm import LLMClient  # noqa: E402
from evolution.researcher import Researcher  # noqa: E402
from evolution.runner import CortexClient  # noqa: E402
from sparc_common import config  # noqa: E402


def _cfg(key: str, default=None):
    return config.get(f"evolution.{key}", default)


def make_client(role: str, override: str | None = None, **kw) -> LLMClient:
    """Build one researcher/grader client from the evolution.researcher config.

    All three roles share base_url and auth; only the model slug differs, so a
    single local endpoint can serve all of them while leaving room to move one
    role elsewhere later.
    """
    rc = _cfg("researcher", {}) or {}
    # kw wins over config so callers (e.g. probe_client) can override any field
    # without colliding on a keyword already passed explicitly.
    fields = {
        "model": override or rc.get(f"{role}_model"),
        "base_url": rc.get("base_url", ""),
        "api_key_env": rc.get("api_key_env", ""),
        "timeout": float(rc.get("timeout_s", 600)),
        "max_concurrency": int(rc.get("max_concurrency", 4)),
    }
    fields.update(kw)
    return LLMClient(**fields)


def probe_client(role: str, override: str | None = None) -> LLMClient:
    """Same client, but diagnosing rather than working: one attempt, short ceiling.
    A preflight that retries with backoff turns 'the endpoint is down' into minutes
    of silence, which is the opposite of what a preflight is for."""
    # 240s, not 90s: LM Studio JIT-loads a model on first request, so the very
    # first probe after an idle period pays a cold 27B load. Observed 2026-08-17 —
    # postmortem timed out at 90s while synthesis/judge, hitting the now-warm model,
    # answered in 1.8s.
    return make_client(role, override, max_retries=1, timeout=240.0)


def preflight(args) -> int:
    """Check every dependency before spending a single model call."""
    ok = True

    def check(label: str, fn):
        nonlocal ok
        try:
            print(f"  {label:.<46} {fn()}")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:.<46} FAIL: {e}")
            ok = False

    print("\nsuite")
    check("probe ids unique", lambda: (
        "ok" if len({p.id for p in suite.PROBES}) == len(suite.PROBES)
        else (_ for _ in ()).throw(AssertionError("duplicate probe id"))))
    check("scenarios resolve", lambda: (
        "ok" if all(p.scenario in suite.SCENARIOS for p in suite.PROBES)
        else (_ for _ in ()).throw(AssertionError("unknown scenario"))))
    check("composition", suite.summary)

    def band_check():
        from evolution import fitness as F
        n = len(suite.probes()) * (args.reps or int(_cfg("reps", 3)))
        want = F.suggested_band(n)
        verdict = "ok" if F.BAND_A >= want * 0.9 else (
            f"WARN: BAND_A={F.BAND_A} is finer than 1 s.e. ({want:.3f}) — "
            "measurement noise could outrank reliability")
        return f"n={n} trials, 1 s.e.={want:.3f}, BAND_A={F.BAND_A} -> {verdict}"
    check("fitness band vs noise floor", band_check)

    print("\ngenomes")
    check("gen-0 seeds validate", lambda: f"{len(seeds.build(tuple(args.distill)))} cells")

    print("\nlocal deps (vector mirror)")

    def vec_check():
        import sqlite3
        __import__("sqlite_vec"), __import__("fastembed")
        if not hasattr(sqlite3.Connection, "enable_load_extension"):
            # Importing sqlite_vec is not enough — the interpreter's own sqlite3 must
            # be built with extension loading. Stock macOS python is not.
            raise RuntimeError(
                f"{sys.executable} has no sqlite3 extension loading. Run on Node C "
                "(e.g. ~/mlx312/bin/python), as scripts/eval_memory.py notes.")
        return "ok"
    check("sqlite_vec + fastembed + extension loading", vec_check)

    print("\ncortexd (the frozen subject model)")
    cortex_url = args.cortex or _cfg("cortex_url")
    check(f"health @ {cortex_url}", lambda: CortexClient(cortex_url, timeout=15).health())

    def override_check():
        """Does the DEPLOYED cortexd honour the gene overrides?

        The live service accepts unknown request fields silently (pydantic drops
        them), so a pre-refactor build would run the whole experiment with every
        genome sharing one prompt — 36 h of identical scores and no error anywhere.
        Compares reported prompt_chars for a short vs padded header, which is
        deterministic and does not depend on how the model chooses to reply.
        """
        import httpx
        base = {"deliberation_id": "preflight", "scene": "SPARC is in the apartment.",
                "memory": "Maya's flight is Friday.", "conversation": [],
                "event": 'they said: "hello"', "max_options": 3}
        sizes = []
        for pad in ("", "X" * 600):
            r = httpx.post(f"{cortex_url}/think", timeout=300,
                           json={**base, "memory_header_override": f"MEM {pad}: {{memory}}"})
            r.raise_for_status()
            sizes.append(r.json()["timing_ms"].get("prompt_chars", 0))
        delta = sizes[1] - sizes[0]
        if delta < 500:
            raise RuntimeError(
                f"overrides IGNORED (prompt_chars {sizes[0]} -> {sizes[1]}, delta {delta}). "
                "cortexd on Node C is running a pre-refactor build: redeploy "
                "packages/{common,node_c} and restart it, or every genome will score "
                "identically.")
        return f"honoured (prompt_chars {sizes[0]} -> {sizes[1]})"

    if not args.quick:
        check("gene overrides reach the model", override_check)

    def latency_check():
        import httpx
        import time as _t
        t0 = _t.time()
        r = httpx.post(f"{cortex_url}/think", timeout=300, json={
            "deliberation_id": "preflight-lat", "scene": "SPARC is in the apartment.",
            "memory": "Maya's flight is Friday.", "conversation": [],
            "event": 'they said: "when is my flight?"', "max_options": 3})
        wall = _t.time() - t0
        gen = r.json()["timing_ms"].get("generate", 0) / 1000
        overhead = wall - gen
        # Per generation: population x reps x (dev probes + one distill per scenario
        # world), plus the one-rep smoke cascade over the cheap scenarios.
        pop = args.population or int(_cfg("population", 10))
        reps = args.reps or int(_cfg("reps", 3))
        n_probes, n_worlds = len(suite.probes()), len(suite.SCENARIOS)
        smoke = suite.probes(smoke_only=True)
        calls = (pop * reps * (n_probes + n_worlds)
                 + pop * (len(smoke) + len(suite.scenarios_for(smoke, []))))
        hours = calls * wall / 3600
        verdict = "ok" if overhead < 5 else (
            f"WARN: {overhead:.0f}s of non-generation overhead per call — use the IP, "
            "not a .local hostname (see docs §5)")
        return (f"wall={wall:.1f}s generate={gen:.1f}s overhead={overhead:.1f}s "
                f"-> ~{calls} calls = ~{hours:.1f}h/generation "
                f"(~{hours * 20:.0f}h for 20 gens). {verdict}")
    if not args.quick:
        check("think latency + projected runtime", latency_check)

    rc = _cfg("researcher", {}) or {}
    print(f"\nresearcher + grader @ {rc.get('base_url', '(unset)')}")
    if not rc.get("base_url"):
        check("base_url", lambda: (_ for _ in ()).throw(RuntimeError(
            "not set — put the DGX's OpenAI-compatible endpoint in "
            "config/sparc.yaml evolution.researcher.base_url "
            "(e.g. http://100.x.y.z:1234/v1)")))
        print("  (skipping model probes until base_url is set)")
        reachable = False
    else:
        reachable = True

        def endpoint():
            nonlocal reachable
            try:
                return make_client("synthesis").health()
            except Exception:
                reachable = False
                raise
        check("endpoint reachable", endpoint)
        if reachable and not args.quick:
            try:    # absorb the JIT model load here rather than in a role probe
                w = probe_client("postmortem")
                w.complete("Reply with ok.", "ok", max_tokens=8)
                w.close()
            except Exception:
                pass
        if not reachable:
            print("  (skipping model probes — endpoint unreachable, fix that first)")
    for role, override in ((("postmortem", args.postmortem_model),
                            ("synthesis", args.synthesis_model),
                            ("judge", args.judge_model)) if reachable else ()):
        def probe(r=role, o=override):
            # One real round trip per role: a wrong slug or a server that rejects
            # response_format must fail here, cheaply, not hours into a run.
            c = probe_client(r, o)
            t0 = _time.time()
            # Probe with a schema: that is what every real call does, and on
            # LM Studio json_schema is the only structured mode the server accepts.
            out = c.complete_json(
                'Reply ONLY with JSON: {"ok":true}', "Reply with ok.", max_tokens=256,
                schema={"type": "object", "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"], "additionalProperties": False})
            dt = _time.time() - t0
            c.close()
            mode = ("json_schema" if c.schema_mode else
                    "json_object" if c.json_object_mode else "text+salvage")
            return f"{c.model} -> {out} ({dt:.1f}s, {mode})"
        check(f"{role} model", probe)

    def context_check():
        """Does the loaded model actually have room for a postmortem?

        This is the LM Studio trap: context length is set per model *load*, and the
        default is far below the ~20k tokens a postmortem carries. Nothing errors —
        the front of the prompt is silently dropped, which is exactly where the
        traces live, so every postmortem would reason about nothing and the whole
        26 h run would produce confident garbage.

        Functional test rather than a config read: hide a token at the very start of
        a realistically-sized prompt and ask for it back. If the window truncates,
        the needle is the first thing to go.
        """
        needle = "GRAPEFRUIT-7731"
        # ~20k tokens of filler, matching a real postmortem payload.
        filler = ("The apartment is quiet and nothing of note is happening. " * 1400)
        c = probe_client("postmortem")
        c.timeout = float(_cfg("researcher", {}).get("timeout_s", 600))
        c._client.timeout = c.timeout
        out = c.complete(
            "You answer with exactly one word, copied verbatim from the user text.",
            f"REMEMBER THIS CODE: {needle}\n\n{filler}\n\n"
            "What was the code at the very top of this message? Reply with the code only.",
            max_tokens=512, temperature=0.0)
        c.close()
        approx = (len(filler) + 200) // 4
        if needle.split("-")[0].lower() not in (out or "").lower():
            raise RuntimeError(
                f"model lost a needle placed ~{approx} tokens back (replied {out.strip()[:60]!r}). "
                "The context window is truncating the traces. In LM Studio, reload the "
                "model with a larger context length (32k+); the default is far too small.")
        return f"needle survived ~{approx} tokens of context"

    if reachable and not args.quick:
        check("usable context window", context_check)

    print("\n" + ("PREFLIGHT OK — safe to run." if ok else
                  "PREFLIGHT FAILED — fix the above before running."))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true",
                    help="validate rig and models, then exit without running the loop")
    ap.add_argument("--quick", action="store_true",
                    help="with --preflight: skip the checks that make real think calls")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--population", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="repetitions per probe (drives the reliability term)")
    ap.add_argument("--workdir", default=str(ROOT / "evolution_runs" / "phase1"))
    ap.add_argument("--cortex", default=None)
    ap.add_argument("--distill", nargs="+", default=["compound"],
                    choices=["compound", "atomic"],
                    help="G3 variants in the gen-0 factorial")
    ap.add_argument("--postmortem-model", default=None)
    ap.add_argument("--synthesis-model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--no-judge", action="store_true",
                    help="programmatic grading only; skips the ~5 ambiguous-prose probes")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-14s %(message)s",
        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.preflight:
        return preflight(args)

    cortex = CortexClient(args.cortex or _cfg("cortex_url"))
    researcher = Researcher(
        postmortem_client=make_client("postmortem", args.postmortem_model),
        synthesis_client=make_client("synthesis", args.synthesis_model),
    )
    judge = None if args.no_judge else make_client("judge", args.judge_model,
                                                   temperature=0.0)

    evo = Evolver(
        workdir=Path(args.workdir), cortex=cortex, researcher=researcher, judge=judge,
        embed_model=_cfg("embed_model"),
        reps=args.reps or int(_cfg("reps", 3)),
        population=args.population or int(_cfg("population", 10)),
        survivors=int(_cfg("survivors", 4)),
        holdout_every=int(_cfg("holdout_every", 3)),
        distill_variants=tuple(args.distill),
    )
    try:
        evo.run(args.generations)
    finally:
        cortex.close()
        clients = [researcher.postmortem_client, researcher.synthesis_client]
        if judge:
            clients.append(judge)
        calls = sum(c.usage.calls for c in clients)
        secs = sum(c.usage.seconds for c in clients)
        spent = sum(c.usage.cost for c in clients)
        print(f"\nresearcher tier: {calls} calls, {secs / 60:.0f} min"
              + (f", ${spent:.2f}" if spent else " (local, no cost)"))
        print(f"artifacts: {args.workdir}")
        champ = Path(args.workdir) / "champion.yaml"
        if champ.exists():
            g = Genome.load(champ)
            print(f"champion: {g.id}  (gen {g.generation}, from {g.parent})")
            print("  promote by copying its genes into config/sparc.yaml, then re-run "
                  "the full EVAL.md suite as a regression before the rover boots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
