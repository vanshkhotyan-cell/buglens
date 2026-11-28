from pydantic import BaseModel, ConfigDict, Field
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


_BUG_ENUM_TO_CANONICAL: dict[str, str] = {
    "B1_FRAC_ADD": "fraction_add",
    "B2_BINOM_SQ": "binomial_expand",
    "B3_NO_PREC": "operator_precedence",
    "B4_NEG_BRACKET": "unary_minus_distrib",
    "B5_TRANSP_SIGN": "transposition_sign",
}


class ReplayOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    bug_id: str
    applicable: bool
    matches_student: bool = False
    matches_correct: bool = False
    predicted_latex: str | None = None
    predicted_expression: str | None = None
    trace_steps: list[str] = Field(default_factory=list)

    def __init__(self, *args, **kwargs) -> None:
        if args:
            bug_raw = args[0]
            bug_str = bug_raw.value if hasattr(bug_raw, "value") else str(bug_raw)
            bug_str = _BUG_ENUM_TO_CANONICAL.get(bug_str, bug_str)
            applicable = bool(args[1]) if len(args) > 1 else False
            pred = args[2] if len(args) > 2 else None
            traces = list(args[3]) if len(args) > 3 and args[3] is not None else []
            super().__init__(
                bug_id=bug_str,
                applicable=applicable,
                predicted_latex=str(pred) if pred is not None else None,
                predicted_expression=str(pred) if pred is not None else None,
                trace_steps=traces,
            )
        else:
            if "bug_id" in kwargs:
                b = kwargs["bug_id"]
                b_str = b.value if hasattr(b, "value") else str(b)
                kwargs["bug_id"] = _BUG_ENUM_TO_CANONICAL.get(b_str, b_str)
            if "trace_steps" in kwargs and isinstance(kwargs["trace_steps"], tuple):
                kwargs["trace_steps"] = list(kwargs["trace_steps"])
            if "predicted_expression" in kwargs and "predicted_latex" not in kwargs:
                kwargs["predicted_latex"] = kwargs["predicted_expression"]
            super().__init__(**kwargs)


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
