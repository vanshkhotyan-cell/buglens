"""Synthetic, paired policy comparison. Run from the repository root with -m."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from functools import cache, lru_cache
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from buglens.modeling.fusion import shannon_entropy_bits
from buglens.pedagogy.bkt_tracer import SCHEMA, MisconceptionTracer, Parameters
from buglens.pedagogy.remediation import FAMILIES, build_lesson, worked
from buglens.pedagogy.review_queue import classify, review_queue

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "simulation"
GROUPS = ("BugLens Targeted", "Random Same-Topic Practice", "No-Practice Control")
IDS = tuple(f.replay_id for f in FAMILIES)


@dataclass(frozen=True)
class Population:
    initial: float = 0.5
    learn: float = 0.32
    relapse: float = 0.04
    concentration: float = 15
    slip: float = 0.10
    accidental_correct: float = 0.12

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(name)
            if name == "concentration":
                if value <= 0:
                    raise ValueError(name)
            elif not 0 <= value <= 1:
                raise ValueError(name)


@dataclass
class Config:
    cohort: int = 100
    sessions: int = 10
    budget: int = 3
    seeds: tuple = (17, 41, 83)
    tutor: Parameters = field(default_factory=Parameters)
    scenarios: dict = field(
        default_factory=lambda: {
            "baseline": Population(),
            "high_relapse": Population(relapse=0.12),
            "noisy_responses": Population(slip=0.22, accidental_correct=0.25),
        }
    )

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.cohort, self.sessions, self.budget)):
            raise ValueError("Cohort, sessions and active budget must be positive integers")
        if not self.seeds or any(type(v) is not int or v < 0 for v in self.seeds):
            raise ValueError("Provide nonnegative integer seeds")
        if len(set(self.seeds)) != len(self.seeds) or not self.scenarios:
            raise ValueError("Provide distinct seeds and at least one scenario")

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text())
        data["tutor"] = Parameters(**data["tutor"])
        data["scenarios"] = {k: Population(**v) for k, v in data["scenarios"].items()}
        return cls(**data)


class Memory(MisconceptionTracer):
    """Private SQLite connection with a deterministic study clock."""

    def __init__(self, params):
        self.parameters = dict.fromkeys(IDS, params)
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.now = datetime(2000, 1, 1, tzinfo=UTC)
        self.db.create_function(
            "current_timestamp", 0, lambda: self.now.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.db.executescript(SCHEMA)

    @contextmanager
    def connection(self, *, write=False):
        with self.db:
            yield self.db


@lru_cache(maxsize=4096)
def entropy(p):
    return shannon_entropy_bits(np.array([p, 1 - p]))


def binary_gain(p, params):
    a, b = params.g_bug, 1 - params.s_bug
    return entropy(p * a + (1 - p) * b) - p * entropy(a) - (1 - p) * entropy(b)


@cache
def lesson_example(i, version):
    return build_lesson(FAMILIES[i].practice[version]).practice


class Targeted:
    def __init__(self, parameters):
        self.memory = Memory(parameters)
        self.learner = self.memory.create_learner()
        self.params = parameters

    def start(self, session):
        self.session = str(session)
        self.memory.now += timedelta(days=1)
        self.memory.start_session(self.learner, self.session)
        self.due = review_queue(self.memory, self.learner, self.session, now=self.memory.now)

    def choose(self, slot, rng):
        p = self.memory.prior(self.learner).active
        if slot == 0:
            i = max(rng.permutation(len(IDS)), key=lambda j: binary_gain(p[IDS[j]], self.params))
            return i, "diagnostic", bank_example(i, FAMILIES[i].anchor)
        if self.due:
            item = self.due.pop(0)
            return IDS.index(item.family), "review", item.example
        i = max(rng.permutation(len(IDS)), key=lambda j: p[IDS[j]])
        return i, "practice", lesson_example(i, int(rng.integers(2)))

    def observe(self, slot, i, kind, correct):
        self.memory.observe(
            self.learner,
            self.session,
            IDS[i],
            f"{self.session}-{slot}",
            "correct" if correct else "bug",
            kind=kind,
            completed=kind == "practice",
        )


FORMS = (
    ("2/7+3/8", "(x+7)^2", "7+4×3", "17-(8-2)", "x+9=15"),
    ("3/7+2/9", "(x+8)^2", "6+5×4", "19-(7-4)", "x+8=14"),
)


@cache
def assessment_form(index):
    form = FORMS[index]
    for family, prompt in zip(FAMILIES, form, strict=True):
        example = worked(family, prompt)
        if (
            prompt in (*family.practice, family.anchor)
            or not example
            or example.value == example.wrong_value
        ):
            raise ValueError("Held-out assessment must distinguish the two response categories")
    return form


@cache
def bank_example(i, prompt):
    return worked(FAMILIES[i], prompt)


@cache
def observed_answer(example, correct):
    answer = example.value if correct else example.wrong_value
    return classify(example, str(answer))


def individual_rate(mean, concentration, rng):
    return mean if mean in (0, 1) else rng.beta(mean * concentration, (1 - mean) * concentration)


def gain(pre, post):
    return (post - pre) / (1 - pre) if pre < 1 else None


def participant(config, pop, seed, identity, group):
    # Fixed exogenous draws couple the counterfactual copies independently of policy choices.
    init, natural, assessment, responses, learning, policy = [
        np.random.default_rng(s) for s in np.random.SeedSequence([seed, identity]).spawn(6)
    ]
    active = init.random(len(IDS)) < pop.initial
    initial = active.copy()
    learn = individual_rate(pop.learn, pop.concentration, init)
    relapse = individual_rate(pop.relapse, pop.concentration, init)

    def assess(index):
        # One unseen, family-matched binary response per family on each parallel form.
        return assessment.random(len(assessment_form(index))) < np.where(
            active, pop.accidental_correct, 1 - pop.slip
        )

    before = assess(0)
    resolved = np.zeros(len(IDS), dtype=int)
    counts = dict.fromkeys(("practice", "diagnostic", "review"), 0)
    tutor = Targeted(config.tutor) if group == GROUPS[0] else None
    try:
        for session in range(1, config.sessions + 1):
            if session > 1:
                active |= natural.random(len(IDS)) < relapse
            if tutor:
                tutor.start(session)
            for slot in range(config.budget if group != GROUPS[2] else 0):
                if tutor:
                    i, kind, example = tutor.choose(slot, policy)
                else:
                    i = policy.integers(len(IDS))
                    kind, example = "practice", bank_example(i, policy.choice(FAMILIES[i].practice))
                counts[kind] += 1
                correct = responses.random() < (
                    pop.accidental_correct if active[i] else 1 - pop.slip
                )
                draw = learning.random()
                if kind != "diagnostic" and draw < learn:
                    active[i] = False
                evidence = observed_answer(example, bool(correct))
                if tutor:
                    tutor.observe(slot, i, kind, evidence == "correct")
            resolved[initial & ~active & (resolved == 0)] = session
    finally:
        if tutor:
            tutor.memory.db.close()
    after = assess(1)
    times = resolved[initial]
    return dict(
        seed=seed,
        learner=identity,
        group=group,
        pre=before.mean(),
        post=after.mean(),
        gain=gain(before.mean(), after.mean()),
        baseline_errors=int((~before).sum()),
        repeated_errors=int((~before & ~after).sum()),
        initial=int(initial.sum()),
        censored=int((times == 0).sum()),
        active_at_end=int((initial & active).sum()),
        observed_time=int(np.where(times, times, config.sessions).sum()),
        **{f"resolved_{s}": int((times == s).sum()) for s in range(1, config.sessions + 1)},
        **counts,
    )


def ratio(a, b):
    return a / b if b else None


def summarize(frame, sessions):
    metrics = {}
    for key in ("pre", "post", "gain"):
        v = frame[key].dropna()
        for stat, value in (("mean", v.mean()), ("median", v.median()), ("variance", v.var())):
            metrics[f"{key}_{stat}"] = value if pd.notna(value) else None
    totals = frame.select_dtypes("number").sum()
    initial = int(totals["initial"])
    remaining = initial
    curve = [ratio(remaining, initial)]
    for s in range(1, sessions + 1):
        remaining -= int(totals[f"resolved_{s}"])
        curve.append(ratio(remaining, initial))
    metrics.update(
        recurrence=ratio(totals.repeated_errors, totals.baseline_errors),
        restricted_mean_sessions=ratio(totals.observed_time, initial),
        median_resolution_session=next(
            (i for i, v in enumerate(curve) if v is not None and v <= 0.5), None
        ),
        unresolved_fraction=ratio(remaining, initial),
        curve=curve,
        learners=len(frame),
        gain_defined=int(frame.gain.count()),
        initial=initial,
        censored=remaining,
        active_at_end=int(totals.active_at_end),
    )
    for key in ("diagnostic", "practice", "review", "repeated_errors", "baseline_errors"):
        metrics[key] = int(totals[key])
    return metrics


def intervals(frame, draws):
    def means(key):
        v = frame[key].to_numpy(dtype=float)[draws]
        return np.divide(
            np.nansum(v, axis=1),
            np.isfinite(v).sum(axis=1),
            out=np.full(len(v), np.nan),
            where=np.isfinite(v).any(axis=1),
        )

    def ratios(a, b):
        x, y = [frame[k].to_numpy()[draws].sum(axis=1) for k in (a, b)]
        return np.divide(x, y, out=np.full(len(x), np.nan), where=y > 0)

    return {
        "pre_mean": means("pre"),
        "post_mean": means("post"),
        "gain_mean": means("gain"),
        "recurrence": ratios("repeated_errors", "baseline_errors"),
        "restricted_mean_sessions": ratios("observed_time", "initial"),
        "unresolved_fraction": ratios("censored", "initial"),
    }


def bounds(values):
    values = values[np.isfinite(values)]
    return np.quantile(values, [0.025, 0.975]).tolist() if len(values) else [None, None]


def run(config, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, contrasts = [], [], []
    for scenario, pop in config.scenarios.items():
        n = config.cohort * len(config.seeds)
        draws = np.random.default_rng(504).integers(n, size=(1000, n))
        bootstrap = {}
        for group in GROUPS:
            rows = [
                dict(scenario=scenario, **participant(config, pop, seed, i, group))
                for seed in config.seeds
                for i in range(config.cohort)
            ]
            all_rows.extend(rows)
            frame = pd.DataFrame(rows)
            summary = dict(scenario=scenario, group=group, **summarize(frame, config.sessions))
            bootstrap[group] = intervals(frame, draws)
            summary["ci95"] = {k: bounds(v) for k, v in bootstrap[group].items()}
            summaries.append(summary)
        for other in GROUPS[1:]:
            for key, values in bootstrap[GROUPS[0]].items():
                a, b = [
                    next(
                        r[key]
                        for r in reversed(summaries)
                        if r["group"] == g and r["scenario"] == scenario
                    )
                    for g in (GROUPS[0], other)
                ]
                contrasts.append(
                    {
                        "scenario": scenario,
                        "comparison": f"Targeted minus {other}",
                        "metric": key,
                        "difference": a - b if a is not None and b is not None else None,
                        "ci95": bounds(values - bootstrap[other][key]),
                    }
                )
    pd.DataFrame(all_rows).to_csv(out / "learners.csv", index=False)
    pd.json_normalize([{k: v for k, v in r.items() if k != "curve"} for r in summaries]).to_csv(
        out / "aggregate.csv", index=False
    )
    pd.DataFrame(contrasts).to_csv(out / "sensitivity.csv", index=False)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    for key, label in (
        ("recurrence", "Repeated errors / baseline errors"),
        ("gain_mean", "Mean normalized gain"),
        ("curve", "Fraction awaiting first resolution"),
    ):
        fig, axes = plt.subplots(
            1,
            len(config.scenarios),
            figsize=(4 * len(config.scenarios), 4),
            sharey=True,
            squeeze=False,
        )
        for ax, scenario in zip(axes.flat, config.scenarios, strict=True):
            for i, r in enumerate(x for x in summaries if x["scenario"] == scenario):
                color = ("#18675f", "#b46021", "#566482")[i]
                if key == "curve":
                    ax.step(
                        range(config.sessions + 1),
                        r[key],
                        where="post",
                        label=GROUPS[i],
                        color=color,
                    )
                    ax.set_xlabel("Session")
                elif r[key] is not None:
                    ax.scatter(i, r[key], color=color)
                    ax.vlines(i, *r["ci95"][key], color=color)
            ax.set_title(scenario.replace("_", " "))
            if key != "curve":
                ax.set_xticks(range(3), ["Targeted", "Random", "Control"])
        axes[0, 0].set_ylabel(label)
        if key == "curve":
            axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{key}.png", dpi=135)
        plt.close(fig)
    (out / "summary.json").write_text(
        json.dumps(
            {"config": asdict(config), "groups": summaries, "contrasts": contrasts},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(Config.load(args.config) if args.config else Config(), args.out)
