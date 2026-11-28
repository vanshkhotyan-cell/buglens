from dataclasses import dataclass
from enum import Enum


class BugID(str, Enum):
    B1_FRAC_ADD = "B1_FRAC_ADD"
    B2_BINOM_SQ = "B2_BINOM_SQ"
    B3_NO_PREC = "B3_NO_PREC"
    B4_NEG_BRACKET = "B4_NEG_BRACKET"
    B5_TRANSP_SIGN = "B5_TRANSP_SIGN"


@dataclass(frozen=True)
class ScopeResult:
    in_scope: bool
    eligible_families: tuple[BugID, ...]
    reason: str


class EquivalenceStatus(str, Enum):
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class VerificationResult:
    status: EquivalenceStatus
    tier_used: int | None
    reason: str


@dataclass(frozen=True)
class ProbeTemplate:
    template_id: str
    family_split_id: str
    expression_template: str
    applicable_families: tuple[BugID, ...]
    parameter_names: tuple[str, ...]
    is_composite: bool = False


@dataclass(frozen=True)
class VerifiedCandidate:
    template_id: str
    family_split_id: str
    question: str
    correct_answer: str
    bug_answers: tuple[tuple[BugID, str], ...]
    applicable_families: tuple[BugID, ...]


@dataclass(frozen=True)
class EIGScore:
    question: str
    expected_information_gain: float
    response_classes: tuple[str, ...]


class ScopeOutException(ValueError):
    """Raised when an expression is outside the supported scope."""


from typing import Any, Literal
from pydantic import BaseModel


class ParsedSubmission(BaseModel):
    """Stores parsed input data, AST representation, and scope eligibility."""

    raw_input: str
    normalized_str: str
    ast_json: dict[str, Any]
    sympy_latex: str
    in_scope: bool
    eligible_families: list[str]
    decline_reason: str | None = None


class ReplayOutput(BaseModel):
    """Stores the execution results of replaying a specific bug rule."""

    bug_id: str
    applicable: bool
    matches_student: bool
    matches_correct: bool
    predicted_latex: str
    trace_steps: list[str]


class DiagnosisDecision(BaseModel):
    """Stores the diagnostic probability distribution and next pedagogical action."""

    posterior: dict[str, float]
    entropy_bits: float
    action: Literal["TEACH", "PROBE", "ABSTAIN"]
    target_bug: str | None = None
    confidence: float
    evidence_traces: list[ReplayOutput]


class ProbeCandidate(BaseModel):
    """Stores candidate follow-up questions and their expected information gain."""

    template_id: str
    question_latex: str
    worded_prompt: str
    expected_answers: dict[str, str]
    eig_bits: float
    verified_distinct: bool
