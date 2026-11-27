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
