"""Verified worked examples for the five workspace question shapes.

Only bounded integer captures and the fixed symbol x reach SymPy. The authored
layout is retained separately so simplification cannot erase a teaching step.
"""

from dataclasses import dataclass
import re

import sympy as sp

from buglens.schemas import DiagnosisDecision, ReplayOutput


@dataclass(frozen=True)
class Family:
    id: str
    replay_id: str
    shape: str
    anchor: str
    practice: tuple[str, str]
    mistake: str
    rule: str
    reflection: str


N = r"\s*([0-9]{1,3})\s*"
X = sp.Symbol("x")
FAMILIES = (
    Family(
        "B1_FRAC_ADD",
        "fraction_add",
        rf"{N}/{N}\+{N}/{N}",
        "1/2+1/2",
        ("1/2+2/5", "3/4+1/8"),
        "Add the numerators and also the denominators.",
        "A denominator names the size of one piece. Rewrite both fractions in pieces of the same size, then count those pieces.",
        "Which common unit did you use, and how many of those pieces did you count?",
    ),
    Family(
        "B2_BINOM_SQ",
        "binomial_expand",
        rf"\(\s*(x|[0-9]{{1,3}})\s*\+{N}\)\s*\^\s*2",
        "(1+1)^2",
        ("(x+4)^2", "(x+6)^2"),
        "Square each term separately and add the squares.",
        "Squaring a sum multiplies two copies of it. Each term multiplies both terms in the other copy, giving two cross products.",
        "Where do both cross products appear in your expansion?",
    ),
    Family(
        "B3_NO_PREC",
        "operator_precedence",
        rf"{N}\+{N}[×*]{N}",
        "1+2×0",
        ("4+3×2", "2+4×3"),
        "Add first, then multiply the result.",
        "Multiplication forms a single term. Find its value before adding the other term.",
        "Which two numbers form the product, and what gets added afterward?",
    ),
    Family(
        "B4_NEG_BRACKET",
        "unary_minus_distrib",
        rf"{N}-\s*\({N}-{N}\)",
        "10-(5-5)",
        ("12-(7-3)", "9-(6-2)"),
        "Remove the brackets and subtract both numbers.",
        "Subtract the value of the entire bracket. Distributing the outside minus reverses the sign of each term inside.",
        "How does the outside minus affect the second term inside the brackets?",
    ),
    Family(
        "B5_TRANSP_SIGN",
        "transposition_sign",
        rf"x\s*\+{N}={N}",
        "x+5=5",
        ("x+7=12", "x+4=9"),
        "Add the left-hand constant to the right-hand side.",
        "An equation stays balanced when you subtract the same amount from both sides. This undoes the addition to x.",
        "What did you do to both sides, and how does substitution check your answer?",
    ),
)


@dataclass(frozen=True)
class Example:
    question: str
    wrong: str
    wrong_value: sp.Expr
    steps: str
    value: sp.Expr


def fraction(a, b):
    return rf"\frac{{{a}}}{{{b}}}"


def worked(family: Family, raw: str) -> Example | None:
    if len(raw) > 300:
        return None
    # Permit whitespace around tokens without joining separate digits.
    match = re.fullmatch(family.shape, raw.strip().replace("−", "-"))
    if not match:
        return None
    args = [X if token == "x" else sp.Integer(int(token)) for token in match.groups()]
    a, b = args[:2]
    if family.id == "B1_FRAC_ADD":
        _, _, c, d = args
        if b == 0 or d == 0:
            return None
        q = fraction(a, b) + "+" + fraction(c, d)
        unit = sp.ilcm(b, d)
        left, right = a * unit / b, c * unit / d
        value, wrong = a / b + c / d, (a + c) / (b + d)
        steps = fraction(left, unit) + "+" + fraction(right, unit)
        wrong_tex = fraction(a + c, b + d)
        check = (left + right) / unit
    elif family.id == "B2_BINOM_SQ":
        q = rf"({a}+{b})^2"
        steps = rf"{a}^2+2\cdot {a}\cdot {b}+{b}^2"
        wrong_tex = rf"{a}^2+{b}^2"
        value, wrong = sp.expand((a + b) ** 2), a**2 + b**2
        check = a**2 + 2 * a * b + b**2
    elif family.id == "B3_NO_PREC":
        c = args[2]
        q = rf"{a}+{b}\times {c}"
        steps, wrong_tex = rf"{a}+({b * c})", rf"({a}+{b})\times {c}"
        value, wrong = a + b * c, (a + b) * c
        check = sp.Add(a, sp.Mul(b, c))
    elif family.id == "B4_NEG_BRACKET":
        c = args[2]
        q = rf"{a}-({b}-{c})"
        steps, wrong_tex = rf"{a}-{b}+{c}", rf"{a}-{b}-{c}"
        value, wrong = a - (b - c), a - b - c
        check = a - b + c
    else:
        q = rf"x+{a}={b}"
        value, wrong = b - a, b + a
        if value + a != b:
            raise ArithmeticError("Equation verification failed")
        return Example(
            q,
            rf"x={b}+{a}={wrong}",
            wrong,
            rf"{q}\quad\Rightarrow\quad x+{a}-{a}={b}-{a}\quad\Rightarrow\quad x={value}",
            value,
        )
    if sp.expand(value - check) != 0:
        raise ArithmeticError("Worked transformation failed verification")
    wrong_result = sp.latex(wrong)
    if wrong_tex != wrong_result and family.id != "B1_FRAC_ADD":
        wrong_tex += "=" + wrong_result
    return Example(q, wrong_tex, wrong, q + "=" + steps + "=" + sp.latex(value), value)


@dataclass(frozen=True)
class Lesson:
    family: Family
    original: Example
    anchor: Example
    practice: Example
    replay: ReplayOutput | None


def build_lesson(raw: str, decision: DiagnosisDecision | None = None) -> Lesson | None:
    """Consume the confirmed question and its own decision, or use a labeled example.

    A live lesson requires a TEACH policy and discriminating, matching evidence.
    A supplied decision with insufficient evidence never falls back to a fixture.
    """
    for family in FAMILIES:
        ids = {family.id, family.replay_id}
        if decision and (decision.action != "TEACH" or decision.target_bug not in ids):
            continue
        original = worked(family, raw)
        if original is None:
            continue
        replay = None
        if decision:
            replay = next(
                (
                    item
                    for item in decision.evidence_traces
                    if item.bug_id in ids
                    and item.applicable
                    and item.matches_student
                    and not item.matches_correct
                ),
                None,
            )
            if replay is None or original.value == original.wrong_value:
                return None
        anchor = worked(family, family.anchor)
        if anchor is None or anchor.value == anchor.wrong_value:
            raise ArithmeticError("Counterexample must distinguish the two rules")
        for prompt in family.practice:
            practice = worked(family, prompt)
            if (
                practice
                and practice.question != original.question
                and practice.value != practice.wrong_value
            ):
                return Lesson(family, original, anchor, practice, replay)
        raise ArithmeticError("No distinct practice question")
    return None
