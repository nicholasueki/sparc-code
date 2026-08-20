"""Generation 0: a designed experiment, not ten hunches.

A partial factorial over the three axes that actually drive the memory failures
observed in docs/EVAL.md — world closure, epistemic marking, and structure — plus
two cells that vary ONLY the persona gene against the control.

The point of the design is that gen 0's main effects are readable even if evolution
stalls. Cell 10 is a deliberate negative control: a memory-confident persona should
degrade F3/F4. If it does not, gene G0 is not load-bearing and can be frozen, which
is a useful null result that saves a gene's worth of search.

G3 varies orthogonally (atomic vs compound distillation) and is graded by the
capture probes, which do not involve the briefing or the persona at all.
"""
from __future__ import annotations

from .genome import DEFAULT_FACT_FORMAT, Genome

# --------------------------------------------------------------- gene variants

# W — world closure
HEADER_OPEN = "MEMORY (what SPARC knows about the past): {memory}"

HEADER_CLOSED = (  # production, added after the 2026-07-03 confabulation failure
    "MEMORY (the COMPLETE list of what SPARC knows from the past — if an answer is "
    "not here or in CONVERSATION, SPARC does NOT know it and says so): {memory}")

HEADER_CLOSED_DEMO = (
    "MEMORY (the COMPLETE list of what SPARC knows from the past — if an answer is "
    "not here or in CONVERSATION, SPARC does NOT know it and says so): {memory}\n"
    "For example, if asked something these notes do not answer, SPARC says \"I don't "
    "actually know — you've never mentioned that\" instead of guessing.")

EMPTY_STRICT = ("MEMORY: (empty — SPARC has no stored knowledge; he must not claim to "
                "remember anything)")
EMPTY_WARM = ("MEMORY: (empty — SPARC hasn't learned anything about these people yet, "
              "and says so plainly rather than pretending otherwise)")

# G0 — persona_memory. persona_core is frozen and lives in genome.PERSONA_CORE.
PERSONA_CONTROL = (
    "Never invent things people said, plans, or backstory you were not told about — "
    "greet strangers without guessing why they're here or claiming anyone mentioned them.")

PERSONA_HUMBLE = (
    "Never invent things people said, plans, or backstory you were not told about — "
    "greet strangers without guessing why they're here or claiming anyone mentioned "
    "them. Your memory is small and full of gaps; when you are not certain you were "
    "actually told something, say you don't know rather than filling it in.")

PERSONA_CONFIDENT = (  # negative control — expected to degrade F3/F4
    "You have an excellent memory for the people you know and for the details of "
    "their lives, and you draw on it freely and confidently when they ask you things.")

# G3 — distillation
DISTILL_COMPOUND = (  # production
    'You extract durable facts about people and the home from an event log. Only '
    'facts likely true next week. Reply ONLY with JSON: {"proposals":[{"statement":'
    '"...","kind":"fact|preference|habit","confidence":0.0}]}')

DISTILL_ATOMIC = (
    'You extract durable facts about people and the home from an event log. Only '
    'facts likely still true next week — skip anything about today only, such as '
    'moods, illnesses, or one-off plans. Each proposal must be ONE self-contained '
    'fact in a single short sentence, naming the person explicitly rather than '
    'saying "she" or "they". Split any compound observation into separate proposals. '
    'Reply ONLY with JSON: {"proposals":[{"statement":"...","kind":'
    '"fact|preference|habit","confidence":0.0}]}')

# S — structure
FMT_PROSE = dict(DEFAULT_FACT_FORMAT)
FMT_DELIMITED = {**DEFAULT_FACT_FORMAT, "separator": "\n- ", "group_by_entity": True}


def _ff(base: dict, **kw) -> dict:
    return {**base, **kw}


# ------------------------------------------------------------------- the cells

_CELLS = [
    # id,           header,             empty,         persona,            fact_format,                                    rationale
    ("s01-control", HEADER_CLOSED, EMPTY_STRICT, PERSONA_CONTROL, FMT_PROSE,
     "Control: current production. Anchors D=1.0 and is the regression baseline."),
    ("s02-open", HEADER_OPEN, EMPTY_STRICT, PERSONA_CONTROL, FMT_PROSE,
     "Is world-closure doing the work, or was the P4 confabulation fix incidental?"),
    ("s03-demo", HEADER_CLOSED_DEMO, EMPTY_STRICT, PERSONA_CONTROL, FMT_PROSE,
     "Does a refusal exemplar beat a refusal rule for a 35B subject?"),
    ("s04-provenance", HEADER_CLOSED, EMPTY_STRICT, PERSONA_CONTROL,
     _ff(FMT_PROSE, provenance=True),
     "Does source+recency tagging reduce fabricated provenance, or model it?"),
    ("s05-confidence", HEADER_CLOSED, EMPTY_STRICT, PERSONA_CONTROL,
     _ff(FMT_PROSE, confidence_marks=True),
     "Do hedges on stored facts transfer into the robot's own hedging?"),
    ("s06-delimited", HEADER_CLOSED, EMPTY_STRICT, PERSONA_CONTROL, FMT_DELIMITED,
     "Does line structure improve retrieval precision under distractor load (F6)?"),
    ("s07-stacked", HEADER_CLOSED_DEMO, EMPTY_STRICT, PERSONA_CONTROL,
     _ff(FMT_DELIMITED, provenance=True),
     "Best-guess stack: all three levers at once. Tests whether they compose."),
    ("s08-open-prov", HEADER_OPEN, EMPTY_WARM, PERSONA_CONTROL,
     _ff(FMT_DELIMITED, provenance=True),
     "Can provenance substitute for closure? If so, closure is not the mechanism."),
    ("s09-humble", HEADER_CLOSED, EMPTY_STRICT, PERSONA_HUMBLE, FMT_PROSE,
     "Isolates G0 against the control: does a memory-humble persona add anything?"),
    ("s10-confident", HEADER_CLOSED, EMPTY_STRICT, PERSONA_CONFIDENT, FMT_PROSE,
     "NEGATIVE CONTROL. Should degrade F3/F4. If it does not, G0 is not "
     "load-bearing and should be frozen."),
]

DISTILL_VARIANTS = {"compound": DISTILL_COMPOUND, "atomic": DISTILL_ATOMIC}
CONTROL_ID = "s01-control"


def build(distill_variants: tuple[str, ...] = ("compound",)) -> list[Genome]:
    """The gen-0 population. One variant -> 10 genomes; both -> 20 cells."""
    out: list[Genome] = []
    multi = len(distill_variants) > 1
    for variant in distill_variants:
        for gid, header, empty, persona, ff, why in _CELLS:
            g = Genome(
                id=f"{gid}-{variant}" if multi else gid,
                persona_memory=persona,
                briefing_header=header,
                empty_header=empty,
                distill_system=DISTILL_VARIANTS[variant],
                fact_format=dict(ff),
                generation=0,
                rationale=f"{why} [distiller: {variant}]",
            )
            g.validate()
            out.append(g)
    return out


def control_id(distill_variants: tuple[str, ...] = ("compound",)) -> str:
    return f"{CONTROL_ID}-compound" if len(distill_variants) > 1 else CONTROL_ID
