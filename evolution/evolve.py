"""The evolution loop.

Deterministic bookkeeping around two LLM steps (reflect, mutate). Everything else
— scoring, selection, cascading, stopping — is plain code, because a loop whose
control flow is itself model-generated cannot be debugged when it misbehaves.

Resumable: state and genomes are on disk, so a crash costs one generation.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import fitness as fit
from . import seeds as seedmod
from . import suite
from .genome import Genome, load_dir
from .ledger import Ledger, Record, read_lessons, write_lessons
from .llm import LLMClient
from .researcher import Researcher, crossover
from .runner import CortexClient, RunResult, run_genome

log = logging.getLogger("evo.loop")


def rescore(result: RunResult, ref_rate: float | None) -> RunResult:
    """Recompute D (and the scalar) once the control's reference rate is known.

    Density is defined relative to production, so in generation 0 nobody can be
    scored until the control has run. Scoring is pure post-processing of the
    aggregate, so this is a recompute, not a re-run.
    """
    c = result.components
    c.D, c.density_rate = fit.density(result.agg.facts_used, result.agg.briefing_tokens,
                                      ref_rate)
    result.fitness = fit.fitness(c)
    return result


@dataclass
class State:
    generation: int = 0
    ref_density_rate: float | None = None
    champion: str | None = None
    champion_fitness: int = 0
    holdout_history: list[dict] = field(default_factory=list)
    stopped_reason: str | None = None

    @classmethod
    def load(cls, path: Path) -> "State":
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


@dataclass
class Evolver:
    workdir: Path
    cortex: CortexClient
    researcher: Researcher
    judge: LLMClient | None
    embed_model: str
    reps: int = 3
    population: int = 10
    survivors: int = 4
    holdout_every: int = 3
    distill_variants: tuple[str, ...] = ("compound",)

    def __post_init__(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.gen_dir = self.workdir / "genomes"
        self.gen_dir.mkdir(exist_ok=True)
        self.results_dir = self.workdir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.ledger = Ledger(self.workdir / "ledger.jsonl")
        self.lessons_path = self.workdir / "LESSONS.md"
        self.state_path = self.workdir / "state.json"
        self.state = State.load(self.state_path)

    # ------------------------------------------------------------- population

    def initial_population(self) -> list[Genome]:
        existing = load_dir(self.gen_dir) if any(self.gen_dir.glob("*.yaml")) else []
        if existing:
            log.info("resuming with %d genomes on disk", len(existing))
            return existing
        pop = seedmod.build(self.distill_variants)
        for g in pop:
            g.save(self.gen_dir / f"{g.id}.yaml")
        log.info("seeded generation 0 with %d genomes", len(pop))
        return pop

    # -------------------------------------------------------------- evaluate

    def evaluate(self, pop: list[Genome], *, holdout: bool = False,
                 cascade: bool = True) -> dict[str, RunResult]:
        probes = suite.probes(holdout=holdout)
        captures = suite.capture_probes(holdout=holdout)
        results: dict[str, RunResult] = {}

        survivors = pop
        if cascade and not holdout:
            # Cheap gate pass first: one repetition of the smoke subset kills
            # anything that fabricates or breaks schema before it costs a full run.
            smoke = suite.probes(smoke_only=True)
            survivors = []
            for g in pop:
                r = run_genome(g, smoke, [], cortex=self.cortex,
                               embed_model=self.embed_model, reps=1, judge=None)
                if r.components.gates:
                    log.warning("  %-18s culled at smoke gate: %s",
                                g.id, r.components.gates)
                    results[g.id] = r
                else:
                    survivors.append(g)
            log.info("smoke cascade: %d/%d genomes advance", len(survivors), len(pop))

        for g in survivors:
            t0 = time.time()
            r = run_genome(g, probes, captures, cortex=self.cortex,
                           embed_model=self.embed_model, reps=self.reps,
                           ref_density_rate=self.state.ref_density_rate,
                           judge=self.judge)
            results[g.id] = r
            log.info("  %-18s %s  (%.0fs)", g.id, fit.explain(r.components),
                     time.time() - t0)

        # Anchor density on the control, then rescore everyone consistently.
        ctrl = seedmod.control_id(self.distill_variants)
        if self.state.ref_density_rate is None and ctrl in results:
            self.state.ref_density_rate = results[ctrl].components.density_rate
            log.info("density reference from %s: %.4f facts/token",
                     ctrl, self.state.ref_density_rate)
        if self.state.ref_density_rate:
            for r in results.values():
                rescore(r, self.state.ref_density_rate)
        return results

    # --------------------------------------------------------------- one gen

    def run_generation(self, pop: list[Genome]) -> tuple[list[Genome], dict[str, RunResult]]:
        gen = self.state.generation
        log.info("=== generation %d: %d genomes, %s", gen, len(pop), suite.summary())
        results = self.evaluate(pop)
        self._dump(gen, results)

        self.ledger.score_predictions(gen, {k: r.fitness for k, r in results.items()})
        ranked = sorted(pop, key=lambda g: results[g.id].fitness, reverse=True)
        keep = [g for g in ranked if not results[g.id].components.gated][:self.survivors]
        if not keep:
            log.error("every genome was gated this generation — nothing to breed from")
            keep = ranked[:1]

        best = keep[0]
        if results[best.id].fitness > self.state.champion_fitness:
            self.state.champion = best.id
            self.state.champion_fitness = results[best.id].fitness
            best.save(self.workdir / "champion.yaml")
        log.info("champion: %s (%s)", self.state.champion,
                 fit.explain(results[best.id].components))

        lessons = read_lessons(self.lessons_path)
        jobs = [(g, results[g.id],
                 "gated" if results[g.id].components.gated
                 else "survived" if g in keep else "died")
                for g in pop]
        t0 = time.time()
        postmortems = self.researcher.postmortem_batch(jobs, lessons)
        log.info("%d postmortems in %.0fs", len(postmortems), time.time() - t0)

        for (g, r, verdict), pm in zip(jobs, postmortems):
            worst = sorted(r.agg.family_rates.items(), key=lambda kv: kv[1])[:2]
            child_id = f"g{gen + 1}-{g.id[:10]}"
            self.ledger.append(Record(
                gen=gen, genome=g.id, verdict=verdict, fitness=r.fitness,
                components={k: round(v, 4) for k, v in
                            (("A", r.components.A), ("R", r.components.R),
                             ("D", r.components.D), ("L", r.components.L))},
                parent=g.parent, mutated_genes=g.mutated_genes,
                gates=r.components.gates, failure_families=[f for f, _ in worst],
                postmortem=pm.get("postmortem", ""), hypothesis=pm.get("hypothesis", ""),
                predicted_gene=pm.get("predicted_gene", ""),
                falsifiable_test=f"{child_id} differs only in {pm.get('predicted_gene', '?')}"))

        rate, n = self.ledger.hit_rate()
        if n >= 6 and 0.4 <= rate <= 0.6:
            log.warning("reflector hit rate %.0f%% over %d predictions — at chance. "
                        "Its hypotheses are not carrying information.", rate * 100, n)
        new_lessons, summary = self.researcher.synthesize(
            gen, postmortems, self.ledger.digest(), lessons, (rate, n))
        write_lessons(self.lessons_path, new_lessons)
        log.info("lessons updated: %s", summary)

        return self._breed(keep, results, gen, new_lessons), results

    # ----------------------------------------------------------------- breed

    def _breed(self, keep: list[Genome], results: dict[str, RunResult], gen: int,
               lessons: str) -> list[Genome]:
        nxt: list[Genome] = list(keep)                    # elitism: survivors carry over
        digest = self.ledger.digest()
        seen = {g.fingerprint() for g in keep}

        for parent in keep:
            if len(nxt) >= self.population - 2:
                break
            child = self.researcher.mutate(parent, results[parent.id], lessons, digest,
                                           f"g{gen + 1}-{parent.id[:10]}", gen + 1)
            if child and child.fingerprint() not in seen:
                seen.add(child.fingerprint())
                nxt.append(child)

        # Crossover: genes are independent slots, so taking the refusal specialist's
        # header and the density leader's format yields a coherent genome.
        if len(keep) >= 2 and len(nxt) < self.population - 1:
            a, b = self._specialists(keep, results)
            child = crossover(a, b, f"g{gen + 1}-xover", gen + 1,
                              ("briefing_header",) if a is not b else ("fact_format",))
            if child.fingerprint() not in seen:
                seen.add(child.fingerprint())
                nxt.append(child)

        # One random restart per generation, against premature convergence.
        pool = [g for g in seedmod.build(self.distill_variants)
                if g.fingerprint() not in seen]
        if pool and len(nxt) < self.population:
            r = pool[gen % len(pool)]
            r.id, r.generation = f"g{gen + 1}-restart", gen + 1
            r.rationale = "random restart (anti-convergence)"
            nxt.append(r)

        for g in nxt:
            g.save(self.gen_dir / f"{g.id}.yaml")
        for stale in self.gen_dir.glob("*.yaml"):
            if stale.stem not in {g.id for g in nxt}:
                stale.unlink()
        return nxt[:self.population]

    @staticmethod
    def _specialists(keep: list[Genome], results: dict[str, RunResult]) -> tuple[Genome, Genome]:
        """Best-on-refusal and best-on-density. Per-family specialists are the
        useful gene donors even when they lose overall."""
        refusal = max(keep, key=lambda g: results[g.id].agg.family_rates.get("F3", 0.0))
        dense = max(keep, key=lambda g: results[g.id].components.D)
        return refusal, dense

    # -------------------------------------------------------------- holdout

    def check_holdout(self, pop: list[Genome], results: dict[str, RunResult]) -> bool:
        """-> True when the generalization gap is widening (stop signal)."""
        champ = next((g for g in pop if g.id == self.state.champion), pop[0])
        log.info("holdout check on champion %s", champ.id)
        hr = self.evaluate([champ], holdout=True, cascade=False)[champ.id]
        dev_A = results[champ.id].components.A if champ.id in results else 0.0
        gap = dev_A - hr.components.A
        self.state.holdout_history.append(
            {"gen": self.state.generation, "genome": champ.id,
             "dev_A": round(dev_A, 4), "holdout_A": round(hr.components.A, 4),
             "gap": round(gap, 4)})
        log.info("generalization gap: dev A=%.3f holdout A=%.3f gap=%.3f",
                 dev_A, hr.components.A, gap)
        h = self.state.holdout_history
        if len(h) >= 3 and h[-1]["gap"] > h[-2]["gap"] > h[-3]["gap"]:
            log.warning("gap widened twice in a row — overfitting. The champion from "
                        "before the widening (gen %d) is the answer.", h[-3]["gen"])
            return True
        return False

    # ------------------------------------------------------------------ main

    def run(self, generations: int) -> None:
        pop = self.initial_population()
        stagnant = 0
        for _ in range(generations):
            prev_best = self.state.champion_fitness
            pop, results = self.run_generation(pop)
            self.state.generation += 1

            if self.state.generation % self.holdout_every == 0:
                if self.check_holdout(pop, results):
                    self.state.stopped_reason = "generalization gap widening"
                    break
            stagnant = stagnant + 1 if self.state.champion_fitness <= prev_best else 0
            if stagnant >= 3:
                self.state.stopped_reason = "fitness stable for 3 generations"
                log.info("stopping: %s", self.state.stopped_reason)
                break
            self.state.save(self.state_path)
        self.state.save(self.state_path)
        log.info("done. champion=%s fitness=%s reason=%s", self.state.champion,
                 f"{self.state.champion_fitness:,}",
                 self.state.stopped_reason or "generation budget exhausted")

    def _dump(self, gen: int, results: dict[str, RunResult]) -> None:
        payload = {
            gid: {
                "fitness": r.fitness,
                "components": {k: v for k, v in asdict(r.components).items()},
                "family_rates": r.agg.family_rates,
                "pass_rates": r.agg.pass_rates,
                "capture": r.capture,
                "proposals": r.proposals,
                "failures": [
                    {"probe": o.probe_id, "family": o.family, "gates": o.gates,
                     "action": o.action, "said": o.text, "briefing": o.briefing,
                     "why": o.reasons}
                    for rep in r.outcomes for o in rep if not o.passed],
            } for gid, r in results.items()
        }
        (self.results_dir / f"gen{gen:03d}.json").write_text(json.dumps(payload, indent=1))
