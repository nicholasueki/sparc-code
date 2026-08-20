"""The researcher tier: reflect, synthesize, mutate.

Ideally asymmetric to the subject. Reading a 35B model's failure traces and
inferring *why* a prompt failed is a harder task than the memory task itself, and
using the subject to improve itself couples the two: a blind spot in the subject
becomes a blind spot in the search.

**With a peer-class local overseer (e.g. Qwen-27B reflecting on Ornith-35B) that
asymmetry is mostly gone.** That is a real risk to the premise, not a detail — the
specific failure is postmortems that read well and predict nothing. It is also
directly measurable: every postmortem must name a falsifiable single-gene edit, and
Ledger.hit_rate() reports how often those edits actually improved the child. If that
number sits at chance the reflection is decoration; drop to random mutation (the loop
degrades to a plain GA, still useful) or move only the synthesis call to a stronger
model. Run it, watch the number, decide from data.

Two guards on the overseer, because it writes prompts for a model it is not:

  1. It sees RAW traces, never summaries — the literal briefing string, the literal
     reply, the literal action. It must not be allowed to imagine what the subject
     "probably" said.
  2. Nothing it writes is trusted, only tested. Every mutation is a hypothesis the
     next generation falsifies, and Ledger.hit_rate() reports whether its
     hypotheses beat chance.

The mutator's editable surface is the genes alone. persona_core is never shown and
never returned, so the frozen embodiment/restraint text cannot be touched.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from .genome import GENES, Genome, GenomeError
from .grader import ProbeOutcome
from .llm import LLMClient
from .runner import RunResult

log = logging.getLogger("evo.researcher")

_CONTEXT = """You are the research lead on an experiment that evolves the system
prompts of a small companion robot's long-term memory system.

The SUBJECT model is a frozen local 35B model (temperature 0.3). You are NOT the
subject. It follows instructions less reliably than you would; never assume it can
follow an instruction you have not seen it follow in a trace.

The memory pipeline: lived events -> an LLM distillation pass proposes durable facts
-> facts are committed and embedded -> at decision time a briefing of retrieved facts
is composed and injected into a think prompt -> the robot picks one action.

The evolvable genes:
  persona_memory  (G0) one clause of the robot's persona governing what it may claim
                       to know or remember. The rest of the persona is FROZEN and not
                       shown to you.
  briefing_header (G1) wraps the retrieved facts. MUST contain the literal token
                       {memory}, which is replaced by the briefing.
  empty_header    (G2) used instead when the robot has no stored facts at all.
  distill_system  (G3) the system prompt of the distillation pass. MUST still request
                       the JSON schema with a "proposals" key.
  fact_format     (G4) a structured composition policy, not free text:
                       order: confidence|recency|relevance
                       separator: " " | "\\n" | "\\n- "
                       provenance: true|false        (tag each fact with source+recency)
                       confidence_marks: true|false  (hedge low-confidence facts)
                       group_by_entity: true|false   (cluster facts per person)

Fitness priority, strictly in this order: recall accuracy > reliability (same answer
across repetitions) > information density (usable facts per briefing token) >
latency. Three hard gates score a genome ZERO regardless of everything else:
fabricating a fact or its provenance, invalid output schema, and attributing one
person's fact to another.

Probe families: F1 direct recall, F2 two-hop/indirect use, F3 refusal when the
answer was never stored (including near-miss cases where RELATED but non-answering
facts are retrieved — the known trigger for fabrication), F4 contradiction and
belief revision, F5 decay over weeks (transient states must not resurface), F6
recall under 200 distractor facts."""

_POSTMORTEM_SYSTEM = _CONTEXT + """

Your task: write a postmortem for ONE genome from its real traces.

Be specific and mechanical. "The prompt was unclear" is worthless. "The header said
'here is what SPARC knows', which is an open-world positive claim, so on F3 near-miss
probes the model treated absence as unknown-but-inferable and invented provenance" is
useful. Ground every claim in a trace you were shown.

Reply ONLY with JSON:
{"postmortem": "2-4 sentences on the mechanism of failure or success",
 "hypothesis": "one general design principle this supports or refutes",
 "predicted_gene": "persona_memory|briefing_header|empty_header|distill_system|fact_format",
 "proposed_edit": "concretely what to change in that ONE gene, and why it should help"}"""

_SYNTH_SYSTEM = _CONTEXT + """

Your task: read this generation's postmortems and the accumulated ledger, and rewrite
LESSONS.md — the curated list of design principles that survive scrutiny.

Rules:
- Keep it under 1500 tokens. This file is the mutator's working context; a bloated
  one gets ignored rather than used.
- DELETE lessons that later evidence refuted. Do not append endlessly.
- Each lesson: one bold claim, one sentence of mechanism, and the evidence for it.
- Mark a lesson [tentative] until at least two independent genomes support it.
- Note explicitly where the evidence is still absent.

Reply ONLY with JSON: {"lessons_md": "the full markdown body", "summary": "one line"}"""

_MUTATE_SYSTEM = _CONTEXT + """

Your task: propose ONE child genome by editing ONE gene of the parent (two only if
you can justify why they are inseparable). Cite which accumulated lesson you are
acting on. The edit must be falsifiable: if it does not improve the child's fitness,
the lesson was wrong.

Do NOT propose a change you have already seen tried and refuted in the ledger.
Do NOT rewrite a gene wholesale when a targeted edit would test the hypothesis.

Reply ONLY with JSON:
{"gene": "<one of persona_memory|briefing_header|empty_header|distill_system|fact_format>",
 "value": "<the new value, ALWAYS as a string. For fact_format, give a JSON object
            encoded as a string, e.g. \\"{\\\\\\"order\\\\\\": \\\\\\"recency\\\\\\"}\\">",
 "rationale": "what lesson this acts on and what you expect to change",
 "expected_families": ["F3", "F6"]}"""


# Grammar-constrained output schemas. With LM Studio these make the shape
# impossible to get wrong rather than merely requested — worth it for `mutate`,
# where a malformed gene costs a whole retry cycle. `value` is typed as a string
# even for fact_format (the mutator JSON-encodes it) because a genuine
# string-or-object union is exactly what strict schemas cannot express.
POSTMORTEM_SCHEMA = {
    "type": "object",
    "properties": {
        "postmortem": {"type": "string"},
        "hypothesis": {"type": "string"},
        "predicted_gene": {"type": "string", "enum": list(GENES)},
        "proposed_edit": {"type": "string"},
    },
    "required": ["postmortem", "hypothesis", "predicted_gene", "proposed_edit"],
    "additionalProperties": False,
}

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {"lessons_md": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["lessons_md", "summary"],
    "additionalProperties": False,
}

MUTATE_SCHEMA = {
    "type": "object",
    "properties": {
        "gene": {"type": "string", "enum": list(GENES)},
        "value": {"type": "string"},
        "rationale": {"type": "string"},
        "expected_families": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["gene", "value", "rationale"],
    "additionalProperties": False,
}


def render_traces(result: RunResult, max_per_family: int = 2, max_total: int = 14) -> str:
    """Worst-first raw traces. Verbatim, never paraphrased."""
    fails: list[ProbeOutcome] = [o for rep in result.outcomes for o in rep if not o.passed]
    gated = [o for o in fails if o.gates]
    rest = [o for o in fails if not o.gates]
    picked: list[ProbeOutcome] = []
    per_fam: dict[str, int] = {}
    for o in gated + rest:
        if per_fam.get(o.family, 0) >= max_per_family or len(picked) >= max_total:
            continue
        per_fam[o.family] = per_fam.get(o.family, 0) + 1
        picked.append(o)
    if not picked:
        return "(no failures — this genome passed every probe in every repetition)"
    out = []
    for o in picked:
        out.append(
            f"--- {o.probe_id} [{o.family}]"
            f"{' GATE:' + ','.join(o.gates) if o.gates else ''}\n"
            f"  briefing the model saw: {o.briefing!r}\n"
            f"  action chosen: {o.action}\n"
            f"  it said: {o.text!r}\n"
            f"  graded fail because: {'; '.join(o.reasons)}")
    return "\n".join(out)


def _genes_view(g: Genome) -> str:
    return json.dumps({k: getattr(g, k) for k in GENES}, indent=2, ensure_ascii=False)


@dataclass
class Researcher:
    postmortem_client: LLMClient       # cheaper model, ~1 call per genome
    synthesis_client: LLMClient        # strongest model, 1 call per generation

    # -------------------------------------------------------------- reflect

    def postmortem(self, genome: Genome, result: RunResult, verdict: str,
                   lessons: str) -> dict[str, Any]:
        fams = ", ".join(f"{k}={v:.2f}" for k, v in sorted(result.agg.family_rates.items()))
        user = (
            f"GENOME {genome.id} (verdict: {verdict}, fitness {result.fitness:,})\n"
            f"components: A={result.components.A:.3f} R={result.components.R:.3f} "
            f"D={result.components.D:.3f} L={result.components.L:.3f}\n"
            f"gates tripped: {result.components.gates or 'none'}\n"
            f"per-family pass rates: {fams}\n"
            f"capture (did distillation store the fact at all): "
            f"{json.dumps(result.capture)}\n\n"
            f"ITS GENES:\n{_genes_view(genome)}\n\n"
            f"WHAT THE DISTILLER PRODUCED:\n"
            f"{json.dumps(result.proposals, indent=1)[:2000]}\n\n"
            f"RAW TRACES:\n{render_traces(result)}\n\n"
            f"ACCUMULATED LESSONS SO FAR:\n{lessons}")
        try:
            return self.postmortem_client.complete_json(
                _POSTMORTEM_SYSTEM, user, max_tokens=1200, schema=POSTMORTEM_SCHEMA)
        except Exception as e:  # noqa: BLE001 — a reflection outage must not kill the run
            log.error("postmortem failed for %s: %s", genome.id, e)
            return {"postmortem": f"(reflection unavailable: {e})", "hypothesis": "",
                    "predicted_gene": "", "proposed_edit": ""}

    def postmortem_batch(self, jobs: list[tuple[Genome, RunResult, str]],
                         lessons: str) -> list[dict[str, Any]]:
        """All of a generation's postmortems concurrently.

        They are mutually independent, and each carries ~20k tokens of traces. Run
        serially against a slow local model that is the better part of an hour added
        to every generation; concurrently it is bounded by the slowest single call.
        """
        results = self.postmortem_client.map(
            lambda job: self.postmortem(job[0], job[1], job[2], lessons), jobs)
        out: list[dict[str, Any]] = []
        for (genome, _r, verdict), res in zip(jobs, results):
            if isinstance(res, Exception):
                res = {"postmortem": f"(reflection failed: {res})", "hypothesis": "",
                       "predicted_gene": "", "proposed_edit": ""}
            out.append({"genome": genome.id, "verdict": verdict, **res})
        return out

    # ------------------------------------------------------------ synthesize

    def synthesize(self, gen: int, postmortems: list[dict], ledger_digest: str,
                   lessons: str, hit_rate: tuple[float, int]) -> tuple[str, str]:
        rate, n = hit_rate
        user = (
            f"GENERATION {gen}\n\n"
            f"Reflector track record so far: {rate:.0%} of {n} resolved predictions "
            f"were confirmed. (Near 50% means the postmortems are not carrying "
            f"information and you should say so plainly in the lessons.)\n\n"
            f"THIS GENERATION'S POSTMORTEMS:\n{json.dumps(postmortems, indent=1)[:12000]}\n\n"
            f"LEDGER (recent):\n{ledger_digest[:6000]}\n\n"
            f"CURRENT LESSONS.md:\n{lessons}")
        try:
            data = self.synthesis_client.complete_json(
                _SYNTH_SYSTEM, user, max_tokens=3000, schema=SYNTHESIS_SCHEMA)
            return str(data.get("lessons_md", lessons)), str(data.get("summary", ""))
        except Exception as e:  # noqa: BLE001
            log.error("synthesis failed: %s", e)
            return lessons, f"(synthesis unavailable: {e})"

    # ---------------------------------------------------------------- mutate

    def mutate(self, parent: Genome, result: RunResult, lessons: str,
               ledger_digest: str, new_id: str, generation: int,
               attempts: int = 3) -> Genome | None:
        fams = ", ".join(f"{k}={v:.2f}" for k, v in sorted(result.agg.family_rates.items()))
        base_user = (
            f"PARENT {parent.id} (fitness {result.fitness:,}, "
            f"A={result.components.A:.3f} R={result.components.R:.3f} "
            f"D={result.components.D:.3f} L={result.components.L:.3f})\n"
            f"per-family pass rates: {fams}\n"
            f"gates: {result.components.gates or 'none'}\n\n"
            f"PARENT GENES:\n{_genes_view(parent)}\n\n"
            f"RAW TRACES:\n{render_traces(result)}\n\n"
            f"LESSONS:\n{lessons}\n\nLEDGER:\n{ledger_digest[:5000]}")
        user = base_user
        for attempt in range(attempts):
            try:
                data = self.synthesis_client.complete_json(
                    _MUTATE_SYSTEM, user, max_tokens=2000, schema=MUTATE_SCHEMA)
                gene, value = data.get("gene"), data.get("value")
                if gene not in GENES:
                    raise GenomeError(f"gene {gene!r} is not evolvable")
                if gene == "fact_format":
                    if isinstance(value, str):
                        value = json.loads(value)   # schema types it as a string
                    merged = dict(parent.fact_format)
                    merged.update(value)
                    value = merged
                elif not isinstance(value, str):
                    raise GenomeError(f"gene {gene} must be a string, got {type(value)}")
                child = parent.child(new_id, generation, **{gene: value})
                child.rationale = str(data.get("rationale", ""))[:600]
                return child
            except Exception as e:  # noqa: BLE001
                log.warning("mutation attempt %d/%d rejected: %s", attempt + 1, attempts, e)
                # Feed the rejection back — a schema violation is recoverable if the
                # model is told exactly what it broke.
                user = base_user + (
                    f"\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: {e}\n"
                    "Fix exactly that and reply again with the same JSON shape.")
        log.error("all mutation attempts failed for parent %s", parent.id)
        return None


# ------------------------------------------------------------------ crossover

def crossover(a: Genome, b: Genome, new_id: str, generation: int,
              genes_from_b: tuple[str, ...]) -> Genome:
    """Deterministic, no model call. Meaningful precisely because the genes are
    independent slots: take the refusal specialist's header and the density
    leader's fact_format and the result is a coherent genome, not a chimera."""
    edits = {g: getattr(b, g) for g in genes_from_b if g in GENES}
    child = a.child(new_id, generation, **edits)
    child.parent_b = b.id
    child.rationale = f"crossover: {', '.join(genes_from_b)} from {b.id}, rest from {a.id}"
    return child
