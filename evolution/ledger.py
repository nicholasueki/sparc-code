"""The ledger: what makes this research rather than search.

Every death and every promotion writes a record with a postmortem and a
*falsifiable* hypothesis naming the single gene edit that would test it. The next
generation either confirms or refutes it, and `hit_rate()` reports how often the
reflector's predictions actually panned out.

That number is the guard against reflection theater. If it sits near chance, the
postmortems are decoration and mutation should fall back to random perturbation —
better to know that than to keep paying for plausible-sounding narration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Record:
    gen: int
    genome: str
    verdict: str                     # survived | died | gated | champion
    fitness: int
    components: dict[str, float]
    parent: str | None = None
    mutated_genes: list[str] = field(default_factory=list)
    gates: list[str] = field(default_factory=list)
    failure_families: list[str] = field(default_factory=list)
    postmortem: str = ""
    hypothesis: str = ""
    falsifiable_test: str = ""       # "child X differs only in gene G"
    predicted_gene: str = ""         # gene the reflector expects to matter
    # filled in once the child has run
    prediction_outcome: str = ""     # confirmed | refuted | untested


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, rec: Record) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def all(self) -> list[dict[str, Any]]:
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def score_predictions(self, gen: int, results: dict[str, int]) -> None:
        """Resolve last generation's predictions against this generation's fitness.

        A prediction is confirmed when the child that carries the proposed
        single-gene edit outscores the parent it was derived from.
        """
        rows = self.all()
        fitness_by_id = dict(results)
        changed = False
        for r in rows:
            if r.get("prediction_outcome") or not r.get("falsifiable_test"):
                continue
            child = r.get("falsifiable_test", "").split()[0] if r.get("falsifiable_test") else ""
            if child in fitness_by_id:
                r["prediction_outcome"] = (
                    "confirmed" if fitness_by_id[child] > r["fitness"] else "refuted")
                changed = True
        if changed:
            with self.path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def hit_rate(self) -> tuple[float, int]:
        """-> (fraction confirmed, n resolved). Near 0.5 means the reflector is
        no better than chance and its hypotheses are not carrying information."""
        res = [r for r in self.all() if r.get("prediction_outcome") in
               ("confirmed", "refuted")]
        if not res:
            return 0.0, 0
        return sum(r["prediction_outcome"] == "confirmed" for r in res) / len(res), len(res)

    def digest(self, limit: int = 25) -> str:
        """Compact recent history for the mutator's context."""
        rows = self.all()[-limit:]
        out = []
        for r in rows:
            out.append(
                f"[gen{r['gen']} {r['genome']} {r['verdict']} fit={r['fitness']:,}"
                f"{' gates=' + ','.join(r['gates']) if r.get('gates') else ''}] "
                f"{r.get('postmortem', '')[:220]}")
        return "\n".join(out)


LESSONS_HEADER = """# LESSONS — memory prompt design

Curated by the synthesis pass each generation. This file is the mutator's working
context, so it must stay short (<1500 tokens): a bloated lessons file gets ignored
rather than used. Superseded lessons are deleted, not appended to.
"""


def read_lessons(path: str | Path) -> str:
    p = Path(path)
    return p.read_text() if p.exists() else LESSONS_HEADER


def write_lessons(path: str | Path, body: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = body if body.lstrip().startswith("#") else LESSONS_HEADER + "\n" + body
    p.write_text(text.rstrip() + "\n")
