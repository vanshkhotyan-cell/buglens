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


@dataclass(frozen=True)
class ReplayOutput:
    bug_id: BugID
    applicable: bool
    predicted_expression: str | None
    trace_steps: tuple[str, ...]


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