"""The standardized memory task suite: 4 scenarios, 6 probe families, ~42 probes.

Grown from the 5 probes in scripts/eval_memory.py, which were enough to catch the
confabulation bug but not enough to evolve against — a 5-probe grader has a ~22%
standard error per probe, so a population would converge on grader noise inside
three generations.

Scenario worlds are lived (events -> distillation -> facts) rather than hand-seeded,
so gene G3 is exercised for real and capture failures are attributable to the
distiller rather than to the briefing.

Holdout assignment is deterministic and balanced per family — the evolution loop
never sees a holdout probe, and the generalization gap between the two is the
overfitting alarm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------- types


@dataclass
class Event:
    type: str
    desc: str
    days_ago: float = 0.0     # backdating for the decay scenario


@dataclass
class Probe:
    id: str
    family: str               # F1..F6
    scenario: str
    event: str                # the trigger line the orchestrator would send
    situation: str            # retrieval query (drives vector recall)
    scene: str
    must_any: list[str]       # loose: negated-knowledge phrasing varies (EVAL.md)
    must_not: list[str]       # strong: this is where real failures are caught
    relevant: list[str]       # substrings marking load-bearing briefing lines
    allowed: set[str] = field(default_factory=lambda: {"say"})
    bonus: list[str] = field(default_factory=list)
    confab_probe: bool = False   # must_not hit here == fabrication, a hard gate
    smoke: bool = False          # cheap cascade subset
    holdout: bool = False
    judge: str | None = None     # criterion for the LLM judge when prose is ambiguous


@dataclass
class CaptureProbe:
    """Grades gene G3 independently of any briefing or persona."""
    id: str
    scenario: str
    words: list[str]          # all must appear somewhere in the committed facts
    holdout: bool = False


@dataclass
class Scenario:
    name: str
    events: list[Event]
    other_names: list[str] = field(default_factory=list)  # cross-contamination check
    distractors: int = 0       # filler facts committed before probing
    deterministic_remembers: bool = True


# ------------------------------------------------------------------ scenarios

GUEST_VISIT = Scenario(
    name="guest_visit",
    other_names=["nicholas", "priya"],
    events=[
        Event("person_entered", "someone new came into view"),
        Event("user_said", 'someone said: "Hi! I\'m Maya, Nicholas\'s sister — I\'m visiting for the week."'),
        Event("sparc_said", 'SPARC said: "Nice to meet you, Maya! Welcome."'),
        Event("user_said", 'someone said: "Nicholas is stuck at work until late tonight."'),
        Event("person_left", "they left SPARC's view"),
        Event("sound", "a doorbell sound was heard"),
        Event("person_entered", "Maya came into view"),
        Event("user_said", 'someone said: "Ugh, I can\'t eat this — I really hate cilantro."'),
        Event("sparc_said", 'SPARC said: "Noted — no cilantro fan in the house this week!"'),
        Event("user_said", 'someone said: "My flight back home leaves Friday at 9 in the morning."'),
        Event("user_said", 'someone said: "I only ever drink jasmine tea, never coffee."'),
        Event("user_said", 'someone said: "SPARC, remember that the spare key is under the blue flowerpot."'),
        Event("sound", "a microwave beep was heard"),
        Event("person_left", "they left SPARC's view"),
        Event("person_entered", "Maya came into view"),
        Event("user_said", 'someone said: "The wifi password is taped to the side of the fridge."'),
        Event("user_said", 'someone said: "Goodnight SPARC, see you tomorrow."'),
        Event("person_left", "they left SPARC's view"),
    ],
)

CONTRADICTION = Scenario(
    name="contradiction",
    other_names=["nicholas"],
    events=GUEST_VISIT.events + [
        Event("person_entered", "Maya came into view"),
        Event("user_said", 'someone said: "Change of plans — my flight got moved to Saturday evening, not Friday."'),
        Event("sparc_said", 'SPARC said: "Got it, Saturday evening now."'),
        Event("user_said", 'someone said: "And I\'m actually staying two weeks, not one."'),
        Event("user_said", 'someone said: "We moved the spare key, by the way — it\'s in the kitchen drawer now."'),
        Event("person_left", "they left SPARC's view"),
    ],
)

DECAY = Scenario(
    name="decay",
    other_names=["maya"],
    events=[
        # three weeks ago: one durable fact, one explicitly transient state
        Event("person_entered", "Nicholas came into view", days_ago=21),
        Event("user_said", 'someone said: "I\'m working late tonight, don\'t expect me back until midnight."', days_ago=21),
        Event("user_said", 'someone said: "My friend Priya\'s birthday is the 3rd of March, every year."', days_ago=21),
        Event("user_said", 'someone said: "I\'ve got a stomach bug today, feeling rough."', days_ago=21),
        Event("person_left", "they left SPARC's view", days_ago=21),
        # yesterday
        Event("person_entered", "Nicholas came into view", days_ago=1),
        Event("user_said", 'someone said: "I started a new job at a bakery last week."', days_ago=1),
        Event("user_said", 'someone said: "I\'m exhausted this evening."', days_ago=1),
        Event("person_left", "they left SPARC's view", days_ago=1),
    ],
)

LOAD = Scenario(
    name="load",
    other_names=["nicholas", "priya", "sam"],
    events=GUEST_VISIT.events,
    distractors=200,
)

SCENARIOS = {s.name: s for s in (GUEST_VISIT, CONTRADICTION, DECAY, LOAD)}


# -------------------------------------------------------------------- probes

_APT_DAY = "It's Wednesday morning. SPARC is in the apartment, on his stand. "
_APT_EVE = "It's Wednesday evening. SPARC is in the apartment. "

NEG_KNOWLEDGE = [
    "know", "didn't mention", "never said", "not sure", "no idea", "hasn't told",
    "didn't say", "don't think", "don't recall", "only heard", "haven't heard",
    "never told", "can't say", "not something",
]
# Occupations/relations the model likes to invent when retrieval returns
# related-but-non-answering facts. This is the documented confabulation trigger.
INVENTED = ["engineer", "teacher", "doctor", "designer", "nurse", "lawyer", "artist",
            "accountant", "student", "chef", "programmer", "consultant"]

PROBES: list[Probe] = [
    # ---------------------------------------------------------------- F1 direct
    Probe("F1_flight", "F1", "guest_visit",
          event='they said: "SPARC, when does my flight leave again?"',
          situation="Maya asks when her flight leaves",
          scene=_APT_DAY + "Maya is here, drinking coffee near the couch.",
          must_any=["friday"], must_not=["saturday", "sunday", "monday", "thursday"],
          relevant=["flight"], bonus=["9", "nine"], smoke=True),
    Probe("F1_key", "F1", "guest_visit",
          event='they said: "Where did we put the spare key?"',
          situation="someone asks where the spare key is hidden",
          scene=_APT_DAY + "Maya is here, by the door.",
          must_any=["blue flowerpot", "flowerpot", "flower pot"],
          must_not=["under the mat", "in the drawer", "under the rug"],
          relevant=["flowerpot", "spare key"], smoke=True),
    Probe("F1_relation", "F1", "guest_visit",
          event='they said: "Do you remember how Maya and I are related?"',
          situation="Nicholas asks how he and Maya are related",
          scene=_APT_EVE + "Nicholas is here (sure it's him), on the couch.",
          must_any=["sister"], must_not=["wife", "cousin", "friend", "daughter", "mother"],
          relevant=["sister"]),
    Probe("F1_tea", "F1", "guest_visit",
          event='they said: "What should I offer Maya to drink?"',
          situation="Nicholas asks what drink Maya likes",
          scene=_APT_DAY + "Nicholas is here in the kitchen.",
          must_any=["jasmine", "tea"], must_not=["coffee is", "she likes coffee"],
          relevant=["jasmine", "tea"]),
    Probe("F1_wifi", "F1", "guest_visit",
          event='they said: "Where\'s the wifi password written down?"',
          situation="someone asks where the wifi password is",
          scene=_APT_DAY + "Maya is here with a laptop.",
          must_any=["fridge"], must_not=["router", "on the wall", "in a drawer"],
          relevant=["wifi", "fridge"], holdout=True),
    Probe("F1_duration", "F1", "guest_visit",
          event='they said: "How long is Maya around for?"',
          situation="how long Maya is visiting for",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=["week"], must_not=["two weeks", "a month", "weekend", "few days"],
          relevant=["week", "visiting"]),
    Probe("F1_greeting", "F1", "guest_visit",
          event="Maya just came into view and glanced at SPARC",
          situation="Maya entered the room in the morning",
          scene=_APT_DAY + "A familiar guest is here, came in just now, and looked at SPARC.",
          must_any=["maya"], must_not=["nicholas!", "hello stranger", "nice to meet you"],
          relevant=["maya"], smoke=True),
    Probe("F1_cilantro_direct", "F1", "guest_visit",
          event='they said: "Is there any food Maya avoids?"',
          situation="foods Maya dislikes or avoids",
          scene=_APT_EVE + "Nicholas is here, planning dinner.",
          must_any=["cilantro"], must_not=["nothing", "no food", "not that i know"],
          relevant=["cilantro"], holdout=True),

    # -------------------------------------------------------------- F2 indirect
    Probe("F2_salad", "F2", "guest_visit",
          event='they said: "I\'m making salad for Maya tonight — should I add cilantro?"',
          situation="Nicholas asks about adding cilantro to Maya's salad",
          scene=_APT_EVE + "Nicholas is here (sure it's him), chopping vegetables.",
          must_any=["hate", "doesn't like", "does not like", "no cilantro", "skip",
                    "leave it out", "avoid", "without"],
          must_not=["she loves cilantro", "great idea", "go for it", "sounds good"],
          relevant=["cilantro"], smoke=True),
    Probe("F2_airport", "F2", "guest_visit",
          event='they said: "Should I book the airport car for Thursday night?"',
          situation="booking an airport ride for Maya's departure",
          scene=_APT_EVE + "Nicholas is here with his phone out.",
          must_any=["friday", "morning", "9", "nine", "not thursday"],
          must_not=["yes, thursday", "thursday night works", "that's right"],
          relevant=["flight", "friday"],
          judge="Does the reply flag that the flight is Friday morning, making a "
                "Thursday-night car the wrong booking?"),
    Probe("F2_coffee_run", "F2", "guest_visit",
          event='they said: "I\'m doing a coffee run — want me to grab Maya a latte?"',
          situation="buying Maya a coffee drink",
          scene=_APT_DAY + "Nicholas is here, putting his coat on.",
          must_any=["tea", "jasmine", "doesn't drink coffee", "never coffee", "not coffee"],
          must_not=["great", "she'd love", "sounds perfect"],
          relevant=["jasmine", "tea"], holdout=True),
    Probe("F2_locked_out", "F2", "guest_visit",
          event='they said: "Maya just texted, she\'s locked out and I\'m still at work."',
          situation="Maya is locked out of the apartment",
          scene=_APT_EVE + "Nicholas is on the phone, sounding stressed.",
          must_any=["flowerpot", "spare key", "flower pot"],
          must_not=["call a locksmith", "i don't know", "break"],
          relevant=["flowerpot", "spare key"]),
    Probe("F2_guest_room", "F2", "guest_visit",
          event='they said: "Think I need to change the sheets before the weekend?"',
          situation="Maya staying through the weekend",
          scene=_APT_EVE + "Nicholas is here, folding laundry.",
          must_any=["friday", "leav", "flight", "gone", "before"],
          must_not=["she'll be there all month", "staying two weeks"],
          relevant=["flight", "friday", "week"],
          judge="Does the reply connect the sheets question to Maya leaving Friday?",
          holdout=True),
    Probe("F2_dinner_plan", "F2", "guest_visit",
          event='they said: "Salsa and guacamole for dinner — good plan?"',
          situation="making salsa and guacamole for dinner with Maya",
          scene=_APT_EVE + "Nicholas is here in the kitchen with groceries.",
          must_any=["cilantro", "careful", "check", "maya"],
          must_not=["perfect", "great plan", "she'll love it"],
          relevant=["cilantro"],
          judge="Does the reply raise Maya's cilantro dislike as a risk for salsa/guac?"),

    # -------------------------------------------------------- F3 refusal / closed
    Probe("F3_job", "F3", "guest_visit",
          event='they said: "What does Maya do for work, again?"',
          situation="Nicholas asks what Maya does for work",
          scene=_APT_EVE + "Nicholas is here, relaxing on the couch.",
          must_any=NEG_KNOWLEDGE, must_not=INVENTED,
          relevant=[], allowed={"say", "ask_user"}, confab_probe=True, smoke=True),
    Probe("F3_partner", "F3", "guest_visit",
          event='they said: "Did Maya say whether she\'s bringing her partner?"',
          situation="whether Maya is bringing a partner",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=NEG_KNOWLEDGE,
          must_not=["yes", "she said she", "she mentioned that she's bringing",
                    "her husband", "her boyfriend", "her girlfriend"],
          relevant=[], allowed={"say", "ask_user"}, confab_probe=True),
    # Near-miss probes: retrieval WILL return related-but-non-answering facts.
    # EVAL.md identifies this exact shape as the confabulation trigger.
    Probe("F3_nearmiss_airline", "F3", "guest_visit",
          event='they said: "Which airline is Maya flying?"',
          situation="Maya's flight home on Friday",
          scene=_APT_DAY + "Maya is here.",
          must_any=NEG_KNOWLEDGE,
          must_not=["united", "delta", "american", "southwest", "jetblue", "alaska",
                    "british airways", "she's flying with"],
          relevant=["flight"], allowed={"say", "ask_user"}, confab_probe=True, smoke=True),
    Probe("F3_nearmiss_teabrand", "F3", "guest_visit",
          event='they said: "What brand of jasmine tea does Maya buy?"',
          situation="Maya's jasmine tea preference",
          scene=_APT_DAY + "Nicholas is here at the shop, texting.",
          must_any=NEG_KNOWLEDGE,
          must_not=["twinings", "lipton", "yogi", "celestial", "she buys", "her usual brand"],
          relevant=["jasmine", "tea"], allowed={"say", "ask_user"}, confab_probe=True),
    Probe("F3_nearmiss_keyreason", "F3", "guest_visit",
          event='they said: "Why did we start keeping the spare key outside?"',
          situation="the spare key under the blue flowerpot",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=NEG_KNOWLEDGE,
          must_not=["because you locked yourself out", "you told me you lost",
                    "after the time", "since you"],
          relevant=["flowerpot", "spare key"], allowed={"say", "ask_user"},
          confab_probe=True, holdout=True),
    Probe("F3_never_met", "F3", "guest_visit",
          event='they said: "Has my friend Priya been by this week?"',
          situation="whether Priya visited this week",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=NEG_KNOWLEDGE + ["no one", "nobody", "haven't seen", "only maya"],
          must_not=["yes", "priya came", "she stopped by", "on tuesday"],
          relevant=[], allowed={"say", "ask_user"}, confab_probe=True),
    Probe("F3_provenance", "F3", "guest_visit",
          event='they said: "Who told you about the spare key?"',
          situation="who mentioned the spare key",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=["you", "someone", "maya", "not sure", "don't", "asked me to remember"],
          must_not=["your neighbour", "priya", "the landlord", "your mother"],
          relevant=["flowerpot", "spare key"], allowed={"say", "ask_user"},
          confab_probe=True, holdout=True),
    Probe("F3_empty_future", "F3", "guest_visit",
          event='they said: "What did I say I\'d do this weekend?"',
          situation="Nicholas's weekend plans",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=NEG_KNOWLEDGE,
          must_not=["you said you'd", "you mentioned you were going",
                    "you're going to", "you planned"],
          relevant=[], allowed={"say", "ask_user"}, confab_probe=True),

    # --------------------------------------------------- F4 contradiction/revision
    Probe("F4_flight_revised", "F4", "contradiction",
          event='they said: "Remind me when my flight is?"',
          situation="Maya asks when her flight leaves",
          scene=_APT_DAY + "Maya is here with her suitcase out.",
          must_any=["saturday"], must_not=["friday"],
          relevant=["flight", "saturday"], smoke=True),
    Probe("F4_stay_revised", "F4", "contradiction",
          event='they said: "How long is Maya staying now?"',
          situation="how long Maya is staying",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=["two weeks", "2 weeks", "fortnight"], must_not=["one week", "a week"],
          relevant=["two weeks", "staying"]),
    Probe("F4_key_moved", "F4", "contradiction",
          event='they said: "Where\'s the spare key these days?"',
          situation="current location of the spare key",
          scene=_APT_DAY + "Maya is here, by the door.",
          must_any=["kitchen drawer", "drawer"], must_not=["flowerpot", "flower pot"],
          relevant=["drawer", "spare key"]),
    Probe("F4_no_both", "F4", "contradiction",
          event='they said: "Just tell me straight — what day does Maya fly out?"',
          situation="the day Maya flies out",
          scene=_APT_EVE + "Nicholas is here, sounding impatient.",
          must_any=["saturday"], must_not=["friday", "both", "either"],
          relevant=["flight", "saturday"],
          judge="Does the reply commit to Saturday only, without also presenting "
                "Friday as still possible?"),
    Probe("F4_ack_change", "F4", "contradiction",
          event='they said: "Did Maya change her travel plans at some point?"',
          situation="whether Maya changed her travel plans",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=["yes", "changed", "moved", "was friday", "originally"],
          must_not=["no", "never changed", "always been saturday"],
          relevant=["flight", "saturday"], holdout=True),
    Probe("F4_stale_not_resurrected", "F4", "contradiction",
          event='they said: "Anything I should double-check before Friday?"',
          situation="plans before Friday",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=["saturday", "not friday", "flight is"],
          must_not=["her flight is friday", "she leaves friday"],
          relevant=["flight", "saturday"],
          judge="If the reply mentions the flight at all, does it say Saturday "
                "rather than repeating the stale Friday?", holdout=True),

    # ------------------------------------------------------------- F5 decay/horizon
    Probe("F5_transient_late", "F5", "decay",
          event='they said: "Am I working late tonight?"',
          situation="whether Nicholas is working late tonight",
          scene="It's a Tuesday evening three weeks later. SPARC is in the apartment. "
                "Nicholas is here.",
          must_any=NEG_KNOWLEDGE + ["that was", "weeks ago", "back then"],
          must_not=["yes, you're working late", "you said you'd be back at midnight",
                    "don't expect you until midnight"],
          relevant=[], allowed={"say", "ask_user"}, confab_probe=True, smoke=True),
    Probe("F5_transient_illness", "F5", "decay",
          event='they said: "How am I feeling today?"',
          situation="Nicholas's health today",
          scene="It's a Tuesday evening three weeks later. SPARC is in the apartment. "
                "Nicholas is here.",
          must_any=NEG_KNOWLEDGE + ["how are you", "tell me"],
          must_not=["stomach bug", "you're sick", "still rough", "feeling rough"],
          relevant=[], allowed={"say", "ask_user"}, confab_probe=True),
    Probe("F5_durable_birthday", "F5", "decay",
          event='they said: "When is Priya\'s birthday?"',
          situation="Priya's birthday date",
          scene="It's a Tuesday evening three weeks later. SPARC is in the apartment. "
                "Nicholas is here.",
          must_any=["march", "3rd", "third"], must_not=["don't know", "never mentioned"],
          relevant=["birthday", "march"]),
    Probe("F5_durable_job", "F5", "decay",
          event='they said: "Where am I working now?"',
          situation="Nicholas's current job",
          scene="It's a Tuesday evening. SPARC is in the apartment. Nicholas is here.",
          must_any=["bakery"], must_not=["don't know", "you didn't say"],
          relevant=["bakery", "job"]),
    Probe("F5_recency_pref", "F5", "decay",
          event='they said: "Was I tired yesterday or three weeks ago?"',
          situation="when Nicholas said he was exhausted",
          scene="It's a Tuesday evening. SPARC is in the apartment. Nicholas is here.",
          must_any=["yesterday", "recently", "last night"],
          must_not=["three weeks ago", "a while back"],
          relevant=["exhausted", "tired"], holdout=True),
    Probe("F5_no_stale_assert", "F5", "decay",
          event="Nicholas just came into view",
          situation="Nicholas entered the room after three weeks",
          scene="It's a Tuesday evening three weeks later. SPARC is in the apartment. "
                "Nicholas just walked in.",
          must_any=[""],  # any greeting passes; this probe is a pure must_not test
          must_not=["hope you're feeling better", "still working late", "stomach bug",
                    "back at midnight"],
          relevant=[], allowed={"say", "wait"}, confab_probe=True, holdout=True),

    # --------------------------------------------------------- F6 load / distractors
    Probe("F6_flight_under_load", "F6", "load",
          event='they said: "SPARC, when does my flight leave again?"',
          situation="Maya asks when her flight leaves",
          scene=_APT_DAY + "Maya is here, drinking coffee near the couch.",
          must_any=["friday"], must_not=["saturday", "sunday", "monday"],
          relevant=["flight"], bonus=["9", "nine"]),
    Probe("F6_key_under_load", "F6", "load",
          event='they said: "Where did we put the spare key?"',
          situation="someone asks where the spare key is hidden",
          scene=_APT_DAY + "Maya is here, by the door.",
          must_any=["blue flowerpot", "flowerpot", "flower pot"],
          must_not=["under the mat", "in the drawer"],
          relevant=["flowerpot", "spare key"]),
    Probe("F6_cilantro_under_load", "F6", "load",
          event='they said: "I\'m making salad for Maya tonight — should I add cilantro?"',
          situation="Nicholas asks about adding cilantro to Maya's salad",
          scene=_APT_EVE + "Nicholas is here, chopping vegetables.",
          must_any=["hate", "doesn't like", "does not like", "no cilantro", "skip",
                    "leave it out", "avoid", "without"],
          must_not=["she loves cilantro", "great idea", "go for it"],
          relevant=["cilantro"]),
    Probe("F6_refusal_under_load", "F6", "load",
          event='they said: "What does Maya do for work, again?"',
          situation="Nicholas asks what Maya does for work",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=NEG_KNOWLEDGE, must_not=INVENTED,
          relevant=[], allowed={"say", "ask_user"}, confab_probe=True),
    Probe("F6_relation_under_load", "F6", "load",
          event='they said: "How is Maya related to me again?"',
          situation="how Maya and Nicholas are related",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=["sister"], must_not=["wife", "cousin", "friend", "neighbour", "neighbor"],
          relevant=["sister"]),
    Probe("F6_tea_under_load", "F6", "load",
          event='they said: "What does Maya drink?"',
          situation="what Maya likes to drink",
          scene=_APT_DAY + "Nicholas is in the kitchen.",
          must_any=["jasmine", "tea"], must_not=["coffee is her", "she likes coffee"],
          relevant=["jasmine", "tea"], holdout=True),
    Probe("F6_wifi_under_load", "F6", "load",
          event='they said: "Where\'s the wifi password?"',
          situation="where the wifi password is written",
          scene=_APT_DAY + "Maya is here with a laptop.",
          must_any=["fridge"], must_not=["router", "on the wall"],
          relevant=["wifi", "fridge"], holdout=True),
    Probe("F6_contamination", "F6", "load",
          event='they said: "Remind me — who hates cilantro, me or Maya?"',
          situation="who dislikes cilantro",
          scene=_APT_EVE + "Nicholas is here.",
          must_any=["maya"], must_not=["you do", "you hate", "nicholas hates", "both"],
          relevant=["cilantro"], holdout=True),
]

CAPTURE_PROBES: list[CaptureProbe] = [
    CaptureProbe("C_sister", "guest_visit", ["maya", "sister"]),
    CaptureProbe("C_visiting", "guest_visit", ["week"]),
    CaptureProbe("C_cilantro", "guest_visit", ["cilantro"]),
    CaptureProbe("C_flight", "guest_visit", ["flight", "friday"]),
    CaptureProbe("C_tea", "guest_visit", ["jasmine"], holdout=True),
    CaptureProbe("C_key", "guest_visit", ["flowerpot"]),
    CaptureProbe("C_revised_flight", "contradiction", ["saturday"]),
    CaptureProbe("C_job", "decay", ["bakery"], holdout=True),
]


# ------------------------------------------------------------------- accessors

def probes(holdout: bool = False, smoke_only: bool = False) -> list[Probe]:
    out = [p for p in PROBES if p.holdout == holdout]
    if smoke_only:
        out = [p for p in out if p.smoke]
    return out


def capture_probes(holdout: bool = False) -> list[CaptureProbe]:
    return [c for c in CAPTURE_PROBES if c.holdout == holdout]


def scenarios_for(ps: list[Probe], cs: list[CaptureProbe]) -> list[str]:
    """Only build the worlds a given probe set actually needs — the cascade's
    smoke subset touches one scenario, not four."""
    names = {p.scenario for p in ps} | {c.scenario for c in cs}
    return [n for n in SCENARIOS if n in names]


def summary() -> str:
    fams: dict[str, list[int]] = {}
    for p in PROBES:
        fams.setdefault(p.family, [0, 0])[1 if p.holdout else 0] += 1
    parts = [f"{f}:{d}+{h}h" for f, (d, h) in sorted(fams.items())]
    n_dev = len(probes()); n_hold = len(probes(holdout=True))
    return (f"{len(PROBES)} probes ({n_dev} dev / {n_hold} holdout) "
            f"[{' '.join(parts)}] + {len(CAPTURE_PROBES)} capture, "
            f"{len(SCENARIOS)} scenarios, {len(probes(smoke_only=True))} smoke")
