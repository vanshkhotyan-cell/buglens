from fractions import Fraction
from .ast import Num, Var, Neg, BinOp, Frac, EqNode
from .render import render_ast_latex

from ..schemas import BugID, ReplayOutput

# ---------------------------------
# Helper 1: Display AST
# ---------------------------------


def display(node):
    return render_ast_latex(node)


# ---------------------------------
# Helper 2: Exact Numeric Evaluation
# ---------------------------------


def evaluate_numeric(node):

    if isinstance(node, Num):
        if isinstance(node.value, int):
            return Fraction(node.value)

        return Fraction(str(node.value))

    elif isinstance(node, Neg):
        return -evaluate_numeric(node.child)

    elif isinstance(node, Frac):
        numerator = evaluate_numeric(node.num)
        denominator = evaluate_numeric(node.den)

        return numerator / denominator

    elif isinstance(node, BinOp):
        left = evaluate_numeric(node.left)
        right = evaluate_numeric(node.right)

        if node.op == "+":
            return left + right

        elif node.op == "-":
            return left - right

        elif node.op == "*":
            return left * right

        elif node.op == "**":
            if right.denominator != 1:
                raise ValueError("Invalid exponent")

            return left**right.numerator

    raise ValueError("Expression contains variables")


def predicted_value(node):

    try:
        return str(evaluate_numeric(node))

    except (ValueError, ZeroDivisionError):
        return display(node)


# ---------------------------------
# B1: Naive Fraction Addition
# ---------------------------------


def apply_frac_add(node):

    bug_id = BugID.B1_FRAC_ADD

    if not (
        isinstance(node, BinOp)
        and node.op == "+"
        and isinstance(node.left, Frac)
        and isinstance(node.right, Frac)
    ):
        return ReplayOutput(bug_id, False, None, ())

    left = node.left
    right = node.right

    a = left.num
    b = left.den
    c = right.num
    d = right.den

    faulty_result = Frac(BinOp("+", a, c), BinOp("+", b, d))

    result = predicted_value(faulty_result)

    traces = (
        f"Detected fraction addition: {display(left)} + {display(right)}",
        f"Added numerators: {display(a)} + {display(c)}",
        f"Added denominators: {display(b)} + {display(d)}",
        f"Predicted faulty answer: {result}",
    )

    return ReplayOutput(bug_id, True, result, traces)


# ---------------------------------
# B2: Binomial Exponent Distribution
# ---------------------------------


def apply_binom_sq(node):

    bug_id = BugID.B2_BINOM_SQ

    if not (
        isinstance(node, BinOp)
        and node.op == "**"
        and isinstance(node.left, BinOp)
        and node.left.op == "+"
        and isinstance(node.right, Num)
        and node.right.value == 2
    ):
        return ReplayOutput(bug_id, False, None, ())

    a = node.left.left
    b = node.left.right

    faulty_result = BinOp("+", BinOp("**", a, Num(2)), BinOp("**", b, Num(2)))

    result = predicted_value(faulty_result)

    traces = (
        f"Detected binomial square: {display(node.left)} squared",
        f"Applied square to first term: {display(a)} squared",
        f"Applied square to second term: {display(b)} squared",
        f"Predicted faulty answer: {result}",
    )

    return ReplayOutput(bug_id, True, result, traces)


# ---------------------------------
# B3: Left-to-Right Order of Operations
# ---------------------------------


def operator_precedence(node):

    if isinstance(node, Frac):
        return 2

    if isinstance(node, BinOp):
        if node.op in ("+", "-"):
            return 1

        if node.op == "*":
            return 2

        if node.op == "**":
            return 4

    return 10


def flatten_left_to_right(node, parent_op=None, side=None):

    current_precedence = operator_precedence(node)

    if parent_op is not None:
        parent_precedence = operator_precedence(BinOp(parent_op, Num(1), Num(1)))

        if current_precedence < parent_precedence:
            return [node]

        if current_precedence == parent_precedence and side == "right":
            return [node]

    if isinstance(node, Frac):
        return (
            flatten_left_to_right(node.num, "/", "left")
            + ["/"]
            + flatten_left_to_right(node.den, "/", "right")
        )

    if isinstance(node, BinOp) and node.op in ("+", "-", "*"):
        return (
            flatten_left_to_right(node.left, node.op, "left")
            + [node.op]
            + flatten_left_to_right(node.right, node.op, "right")
        )

    return [node]


def combine(left, operator, right):

    if operator == "/":
        return Frac(left, right)

    return BinOp(operator, left, right)


def apply_no_prec(node):

    bug_id = BugID.B3_NO_PREC

    if isinstance(node, EqNode):
        return ReplayOutput(bug_id, False, None, ())

    if not isinstance(node, (BinOp, Frac)):
        return ReplayOutput(bug_id, False, None, ())

    if (
        isinstance(node, BinOp)
        and node.op == "+"
        and isinstance(node.left, Frac)
        and isinstance(node.right, Frac)
    ):
        return ReplayOutput(bug_id, False, None, ())

    parts = flatten_left_to_right(node)

    operators = [parts[i] for i in range(1, len(parts), 2)]

    has_low_priority = any(op in ("+", "-") for op in operators)

    has_high_priority_after = False
    low_seen = False

    for op in operators:
        if op in ("+", "-"):
            low_seen = True

        elif op in ("*", "/") and low_seen:
            has_high_priority_after = True

    if not (has_low_priority and has_high_priority_after):
        return ReplayOutput(bug_id, False, None, ())

    result_node = parts[0]
    traces = [
        f"Detected mixed operations: {display(node)}",
        "Applied operations strictly from left to right.",
    ]

    for i in range(1, len(parts), 2):
        operator = parts[i]
        right = parts[i + 1]

        previous = result_node

        result_node = combine(result_node, operator, right)

        traces.append(f"{display(previous)} {operator} {display(right)}")

    result = predicted_value(result_node)

    traces.append(f"Predicted faulty answer: {result}")

    return ReplayOutput(bug_id, True, result, tuple(traces))


# ---------------------------------
# B4: Incomplete Negation Distribution
# ---------------------------------


def apply_neg_bracket(node):

    bug_id = BugID.B4_NEG_BRACKET

    faulty_result = None
    traces = []

    # Example: -(a-b)

    if isinstance(node, Neg) and isinstance(node.child, BinOp) and node.child.op in ("+", "-"):
        inside = node.child

        a = inside.left
        b = inside.right

        faulty_result = BinOp(inside.op, Neg(a), b)

        traces = [
            f"Detected -({display(inside)})",
            f"Applied unary minus to first term only: -{display(a)}",
            f"Kept second term sign unchanged: {display(b)}",
        ]

    # Example: 10-(4-1)

    elif (
        isinstance(node, BinOp)
        and node.op == "-"
        and isinstance(node.right, BinOp)
        and node.right.op in ("+", "-")
    ):
        a = node.right.left
        b = node.right.right

        first_step = BinOp("-", node.left, a)

        faulty_result = BinOp(node.right.op, first_step, b)

        traces = [
            f"Detected {display(node)}",
            f"Applied minus to first bracket term: {display(node.left)} - {display(a)}",
            f"Kept second term sign unchanged: {display(b)}",
        ]

    if faulty_result is None:
        return ReplayOutput(bug_id, False, None, ())

    result = predicted_value(faulty_result)

    traces.append(f"Predicted faulty answer: {result}")

    return ReplayOutput(bug_id, True, result, tuple(traces))


# ---------------------------------
# B5: Sign-Drop Transposition
# ---------------------------------


def apply_transp_sign(node):

    bug_id = BugID.B5_TRANSP_SIGN

    if not isinstance(node, EqNode):
        return ReplayOutput(bug_id, False, None, ())

    left = node.lhs
    right = node.rhs

    if not isinstance(left, BinOp):
        return ReplayOutput(bug_id, False, None, ())

    if left.op not in ("+", "-"):
        return ReplayOutput(bug_id, False, None, ())

    if not isinstance(left.left, Var):
        return ReplayOutput(bug_id, False, None, ())

    try:
        evaluate_numeric(left.right)
        evaluate_numeric(right)

    except (ValueError, ZeroDivisionError):
        return ReplayOutput(bug_id, False, None, ())

    variable = left.left
    constant = left.right

    faulty_result = BinOp(left.op, right, constant)

    result = predicted_value(faulty_result)

    traces = (
        f"Detected equation: {display(node)}",
        f"Moved constant without changing its sign: {display(right)} {left.op} {display(constant)}",
        f"Predicted faulty value for {variable.name}: {result}",
    )

    return ReplayOutput(bug_id, True, result, traces)


# ---------------------------------
# Operator Registry
# ---------------------------------

OPERATORS = {
    BugID.B1_FRAC_ADD: apply_frac_add,
    BugID.B2_BINOM_SQ: apply_binom_sq,
    BugID.B3_NO_PREC: apply_no_prec,
    BugID.B4_NEG_BRACKET: apply_neg_bracket,
    BugID.B5_TRANSP_SIGN: apply_transp_sign,
}


def run_all_operators(node):

    return tuple(operator(node) for operator in OPERATORS.values())
