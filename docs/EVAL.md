# Ornith-1.0-35B Capability Eval — 2026-07-03

Harness: `scripts/eval_ornith.py` → cortexd `/think` with deployment-shaped inputs
(prose snapshot + memory briefing + event line, exactly as the orchestrator sends them).
Programmatic grading: schema-valid first try, action kind in allowed set, required/
forbidden strings in chosen text, latency, prompt size. Raw data: `/tmp/ornith_eval.json`.

## Results

| Config | Pass | Median gen | Notes |
|---|---|---|---|
| Verbose persona, T=0.7, 5 opts (original) | 6/8 | 5.1 s | failed restraint (chatty in empty room) + capability honesty (claimed it would preheat an oven) |
| Compact persona, T=0.7, 5 opts | 4/4* | 6.9 s | restraint fixed |
| Compact persona, T=0.3, 5 opts | 4/4* | 5.1 s | same quality, tighter variance |
| **Compact persona, T=0.3, 3 opts** | **4/4*** | **3.6 s** | **winner — ~30% faster** |
| Final persona (+embodiment line), T=0.3 (5 opts in harness) | **8/8** | 6.6 s | both failures fixed; production runs 3 opts → ~4 s expected |

*variant runs used the 4 most diagnostic scenarios (greet-known, grounded recall, honesty, restraint).

## Key findings

1. **Schema reliability is 100%** — ~28 calls, zero parse retries, zero fallbacks. The
   parse+retry+clamp harness has yet to fire in anger. Ornith's structured output is
   dependable at T≤0.7 with the JSON-in-prompt approach (no grammar needed so far).
2. **Grounding works** — memory briefing facts (jasmine tea) surfaced correctly every time;
   no cross-contamination (never called the stranger "Nicholas").
3. **Persona wording does real behavioral work:**
   - Verbose/sociable persona → narrates empty rooms. Adding "prefer to wait quietly when
     nothing needs attention" → clean `wait`.
   - The dangerous failure was **capability fabrication** ("On it! I'll get that preheated").
     Fixed with an explicit embodiment paragraph (no arms/motors). Lesson: honesty
     instructions don't generalize to capabilities the model doesn't know it lacks —
     enumerate the physical limits.
4. **Token efficiency:** 3 options ≈ same decision quality as 5, ~30% latency cut
   (decode-bound). Prompt ~1.3 KB (~350 tok) per think. Next big win is prefix/prompt
   caching of the persona+task blocks (prefill-bound TTFT).

## Adopted production config (committed)

- Compact persona + embodiment constraints + "prefer wait" (config/lucas.yaml)
- `temperature: 0.3`, `max_options: 3` (orchestrator)

## Gaps / next eval iterations

- Multi-turn conversation depth (only 2-turn tested); barge-in/merge scenarios.
- Vision-in-the-loop scenarios (image + event) once Node B camera frames wire in.
- Selector shadow-agreement measurement (10H letter-pick vs Mac choice) on live traffic.
- Post-say deterministic capability checker (validator can't catch fabricated actions from
  wording alone; persona holds for now).
