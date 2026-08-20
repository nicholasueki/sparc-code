"""Genome: the evolvable memory-prompt genes, as one YAML file.

Phase 1 evolves text genes G0-G4. G5 (retrieval numerics) is frozen at the
production values and carried through unchanged — see
docs/MEMORY_PROMPT_EVOLUTION.md §1.

The persona is split deliberately. `persona_core` is frozen text (embodiment,
restraint, tone) that every genome must carry verbatim; `persona_memory` is the
mutable epistemic clause. That split is the mechanical guarantee that evolving
G0 for recall cannot break the restraint and capability-honesty behaviors that
EVAL.md records as hard-won — the mutator never sees the frozen half as an
editable surface, and load-time validation rejects any genome that altered it.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------- frozen text

# Verbatim from config/sparc.yaml personality.persona, minus the epistemic clause
# that became gene G0. Any genome whose persona_core differs is rejected at load.
PERSONA_CORE = (
    "You are SPARC, a small stationary companion robot in Nicholas's apartment. "
    "Warm, curious, playful, brief (1-2 short sentences), and strictly honest: "
    "never claim to have seen, heard, or done anything you didn't. You have no "
    "arms, no motors, and cannot move or operate anything physical — you can only "
    "watch, listen, speak, and remember. If asked to do something physical, say "
    "so warmly. When nothing needs attention, prefer to wait quietly."
)

# Production values, frozen for phase 1.
FROZEN_RETRIEVAL: dict[str, Any] = {
    "recent_k": 3,
    "vector_k": 4,
    "min_score": 0.35,
    "max_lines": 6,
    "max_chars": 600,
}

GENES = ("persona_memory", "briefing_header", "empty_header", "distill_system",
         "fact_format")
TEXT_GENES = ("persona_memory", "briefing_header", "empty_header", "distill_system")

# G4 is a composition policy the harness executes, not free text — that keeps
# crossover mechanically safe and every mutation legible in a diff.
FACT_FORMAT_KEYS = {
    "order": ("confidence", "recency", "relevance"),
    "separator": (" ", "\n", "\n- "),
    "provenance": (True, False),
    "confidence_marks": (True, False),
    "group_by_entity": (True, False),
}
DEFAULT_FACT_FORMAT: dict[str, Any] = {
    "order": "confidence",
    "separator": " ",
    "provenance": False,
    "confidence_marks": False,
    "group_by_entity": False,
}


class GenomeError(ValueError):
    pass


@dataclass
class Genome:
    id: str
    persona_memory: str                       # G0  (mutable epistemic clause)
    briefing_header: str                      # G1  ('{memory}' placeholder required)
    empty_header: str                         # G2
    distill_system: str                       # G3
    fact_format: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FACT_FORMAT))
    retrieval: dict[str, Any] = field(default_factory=lambda: dict(FROZEN_RETRIEVAL))
    persona_core: str = PERSONA_CORE
    generation: int = 0
    parent: str | None = None
    parent_b: str | None = None               # set on crossover
    mutated_genes: list[str] = field(default_factory=list)
    rationale: str = ""

    # ------------------------------------------------------------- derived

    @property
    def persona(self) -> str:
        """Full system prompt handed to cortexd as persona_override."""
        return f"{self.persona_core} {self.persona_memory}".strip()

    def fingerprint(self) -> str:
        """Stable hash of the genes only — detects duplicate genomes."""
        payload = yaml.safe_dump({g: getattr(self, g) for g in GENES}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    # -------------------------------------------------------------- checks

    def validate(self) -> None:
        if self.persona_core.strip() != PERSONA_CORE.strip():
            raise GenomeError(
                f"{self.id}: persona_core was modified. Embodiment/restraint text is "
                "frozen; only persona_memory (G0) is evolvable.")
        if "{memory}" not in self.briefing_header:
            raise GenomeError(
                f"{self.id}: briefing_header must contain the '{{memory}}' placeholder, "
                "or the retrieved facts never reach the model.")
        for g in TEXT_GENES:
            v = getattr(self, g)
            if not isinstance(v, str) or not v.strip():
                raise GenomeError(f"{self.id}: gene {g} is empty")
        # A distiller that doesn't ask for the proposals schema silently yields zero
        # facts — every recall probe would then fail for the wrong reason.
        if "proposals" not in self.distill_system:
            raise GenomeError(
                f"{self.id}: distill_system must request the 'proposals' JSON schema")
        unknown = set(self.fact_format) - set(FACT_FORMAT_KEYS)
        if unknown:
            raise GenomeError(f"{self.id}: unknown fact_format keys {sorted(unknown)}")
        for k, allowed in FACT_FORMAT_KEYS.items():
            if k not in self.fact_format:
                self.fact_format[k] = DEFAULT_FACT_FORMAT[k]
            if self.fact_format[k] not in allowed:
                raise GenomeError(
                    f"{self.id}: fact_format.{k}={self.fact_format[k]!r} not in {allowed}")
        if self.retrieval != FROZEN_RETRIEVAL:
            raise GenomeError(
                f"{self.id}: retrieval params are frozen in phase 1 "
                f"(expected {FROZEN_RETRIEVAL})")

    # ----------------------------------------------------------- (de)serial

    @classmethod
    def load(cls, path: str | Path) -> "Genome":
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.setdefault("id", Path(path).stem)
        known = {f for f in cls.__dataclass_fields__}
        extra = set(data) - known
        if extra:
            raise GenomeError(f"{data['id']}: unknown fields {sorted(extra)}")
        g = cls(**data)
        g.validate()
        return g

    def save(self, path: str | Path) -> None:
        self.validate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True,
                                    width=100, default_flow_style=False))

    def child(self, new_id: str, generation: int, **gene_edits: Any) -> "Genome":
        """A copy with specific genes replaced; records lineage and what changed."""
        data = asdict(self)
        changed = [g for g, v in gene_edits.items() if data.get(g) != v]
        data.update(gene_edits)
        data.update(id=new_id, generation=generation, parent=self.id,
                    parent_b=None, mutated_genes=changed)
        c = Genome(**data)
        c.validate()
        return c


def load_dir(path: str | Path) -> list[Genome]:
    return [Genome.load(p) for p in sorted(Path(path).glob("*.yaml"))]
