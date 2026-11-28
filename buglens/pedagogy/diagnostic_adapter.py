"""Compose the frozen diagnostic contracts with the symbolic-only fusion backend."""

from dataclasses import dataclass
import json
import logging

import numpy as np

from buglens.modeling import fusion
from buglens.schemas import DiagnosisDecision, ProbeCandidate, ReplayOutput


@dataclass(frozen=True)
class BinaryProbe:
    """A supplied two-outcome model in fusion.ALL_CLASSES order."""

    template_id: str
    question_latex: str
    family: str
    predicted_answer: str
    match_likelihood: tuple[float, ...]

    verified_distinct: bool

    def score(self, posterior: np.ndarray) -> ProbeCandidate:
        # The existing EIG implementation assigns uniform odds to unmapped families.
        # Accept only likelihoods that this two-outcome parameterization represents.
        if self.family not in fusion.BUG_FAMILIES or len(self.match_likelihood) != len(
            fusion.ALL_CLASSES
        ):
            raise ValueError("Invalid probe family or likelihood width")
        index = fusion.CLASS_TO_IDX[self.family]
        if not all(0 < p < 1 for p in self.match_likelihood) or any(
            p != 0.5 for i, p in enumerate(self.match_likelihood[:-1]) if i != index
        ):
            raise ValueError("Unsupported binary outcome likelihoods")
        slip = self.match_likelihood[-1]
        gain = fusion.expected_info_gain_bits(
            posterior,
            [{index}, set()],
            eps=1 - self.match_likelihood[index],
            slip_bug_prob=slip,
            slip_other_prob=1 - slip,
        )
        return ProbeCandidate(
            template_id=self.template_id,
            question_latex=self.question_latex,
            worded_prompt="Choose the response category.",
            expected_answers={self.family: self.predicted_answer},
            eig_bits=gain,
            verified_distinct=self.verified_distinct,
        )


@dataclass(frozen=True)
class Observation:
    template_id: str
    matched: bool
    before: dict[str, float]
    after: dict[str, float]


@dataclass(frozen=True)
class Assessment:
    decision: DiagnosisDecision
    candidates: tuple[ProbeCandidate, ...]
    history: tuple[Observation, ...]

    bank: tuple[BinaryProbe, ...]

    @property
    def selected(self) -> ProbeCandidate | None:
        return self.candidates[0] if self.decision.action == "PROBE" and self.candidates else None


def _assess(posterior, replays, probes, history):
    used = {item.template_id for item in history}
    candidates = sorted(
        (p.score(posterior) for p in probes if p.verified_distinct and p.template_id not in used),
        key=lambda p: (-p.eig_bits, p.template_id),
    )
    decision = fusion.selective_decision_policy(
        posterior,
        replays,
        probes_used=len(history),
        max_eig_bits=candidates[0].eig_bits if candidates else 0.0,
    )
    return Assessment(decision, tuple(candidates), history, probes)


def diagnose(
    replays: list[ReplayOutput] | None, probes: tuple[BinaryProbe, ...] = ()
) -> Assessment | None:
    """None means missing evidence. An empty list is a completed replay with no matches."""
    if replays is None:
        return None
    if len({p.template_id for p in probes}) != len(probes):
        raise ValueError("Duplicate probe ID")
    prior = np.full(len(fusion.ALL_CLASSES), 1 / len(fusion.ALL_CLASSES))
    posterior = fusion.log_linear_pool(
        prior, fusion.smoothed_symbolic_likelihood(replays), alphas=(0.1, 1.0, 0.0)
    )
    return _assess(posterior, replays, probes, ())


def observe(current: Assessment, template_id: str, matched: bool) -> Assessment:
    """Consume only the outstanding probe. Repeat delivery is idempotent."""
    if any(item.template_id == template_id for item in current.history):
        return current
    if current.selected is None or current.selected.template_id != template_id:
        raise ValueError("Observation does not match the selected probe")
    probe = next(p for p in current.bank if p.template_id == template_id)
    before = current.decision.posterior
    prior = np.array([before[c] for c in fusion.ALL_CLASSES])
    likelihood = np.array(probe.match_likelihood)
    # Unit log-pool weights give normalized prior times the supplied likelihood.
    posterior = fusion.log_linear_pool(
        prior, likelihood if matched else 1 - likelihood, alphas=(1, 1, 0), delta=0
    )
    record = Observation(
        template_id, matched, before, dict(zip(fusion.ALL_CLASSES, posterior, strict=True))
    )
    return _assess(
        posterior, current.decision.evidence_traces, current.bank, (*current.history, record)
    )


def log_coverage_gap():
    """Only a bounded event and reason leave the browser session."""
    logging.getLogger(__name__).warning(
        json.dumps(
            {
                "event": "diagnostic_coverage_gap",
                "reason": "unsupported_remediation_shape",
            }
        )
    )
