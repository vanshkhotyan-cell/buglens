"""Authored judge demonstrations. These inputs are independent of student submissions."""

from buglens.pedagogy.diagnostic_adapter import BinaryProbe
from buglens.schemas import ReplayOutput


def match(family, answer, step):
    return ReplayOutput(
        bug_id=family,
        applicable=True,
        matches_student=True,
        matches_correct=False,
        predicted_latex=answer,
        trace_steps=[step],
    )


PROBES = (
    BinaryProbe(
        "brackets",
        "8-(4-2)",
        "unary_minus_distrib",
        "2",
        (0.5, 0.5, 0.5, 0.92, 0.5, 0.02),
        verified_distinct=True,
    ),
    BinaryProbe(
        "precedence",
        r"1+2\times3",
        "operator_precedence",
        "9",
        (0.5, 0.5, 0.92, 0.5, 0.5, 0.02),
        verified_distinct=True,
    ),
    BinaryProbe(
        "fractions",
        r"1/2+1/2",
        "fraction_add",
        "1/2",
        (0.92, 0.5, 0.5, 0.5, 0.5, 0.02),
        verified_distinct=True,
    ),
)

# (question, scripted answer, completed replay outputs, supplied probe bank)
CASES = {
    "Fraction error": ("2/3 + 1/4", "3/7", [match("fraction_add", "3/7", "(2+1)/(3+4) = 3/7")], ()),
    "Ambiguous working": (
        "2+3×0-(1-1)",
        "0",
        [
            match("operator_precedence", "0", "(2+3)×0-(1-1) = 0"),
            match("unary_minus_distrib", "0", "2+3×0-1-1 = 0"),
        ],
        PROBES,
    ),
    "Unmatched response": ("2/3 + 1/4", "17", [], ()),
}
