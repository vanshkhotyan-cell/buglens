from .ast import Num, Var, Neg, BinOp, Frac, EqNode

from ..schemas import BugID, ScopeResult


def walk(node):
    """Visit every node in the AST."""

    yield node

    if isinstance(node, BinOp):
        yield from walk(node.left)
        yield from walk(node.right)

    elif isinstance(node, Neg):
        yield from walk(node.child)

    elif isinstance(node, Frac):
        yield from walk(node.num)
        yield from walk(node.den)

    elif isinstance(node, EqNode):
        yield from walk(node.lhs)
        yield from walk(node.rhs)


def check_fraction_add(node):

    return (
        isinstance(node, BinOp)
        and node.op == "+"
        and isinstance(node.left, Frac)
        and isinstance(node.right, Frac)
    )


def check_binomial_square(node):

    return (
        isinstance(node, BinOp)
        and node.op == "**"
        and isinstance(node.right, Num)
        and node.right.value == 2
        and isinstance(node.left, BinOp)
        and node.left.op in ("+", "-")
    )


def check_precedence(nodes):

    operators = {node.op for node in nodes if isinstance(node, BinOp)}

    has_addition = bool(operators & {"+", "-"})
    has_multiplication = bool(operators & {"*", "/"})

    return has_addition and has_multiplication


def check_negative_distribution(nodes):

    for node in nodes:
        # Example: -(a-b)
        if isinstance(node, Neg):
            if isinstance(node.child, BinOp) and node.child.op in ("+", "-"):
                return True

        # Example: 10-(4-1)
        if (
            isinstance(node, BinOp)
            and node.op == "-"
            and isinstance(node.right, BinOp)
            and node.right.op in ("+", "-")
        ):
            return True

    return False


def is_constant(node):

    if isinstance(node, Num):
        return True

    if isinstance(node, Frac):
        return isinstance(node.num, Num) and isinstance(node.den, Num)

    return False


def check_transposition(node):

    if not isinstance(node, EqNode):
        return False

    for side, other_side in ((node.lhs, node.rhs), (node.rhs, node.lhs)):
        if isinstance(side, BinOp) and side.op in ("+", "-") and is_constant(other_side):
            variable_constant = isinstance(side.left, Var) and is_constant(side.right)

            constant_variable = is_constant(side.left) and isinstance(side.right, Var)

            if variable_constant or constant_variable:
                return True

    return False


def scope_gate(expression):

    nodes = list(walk(expression))

    eligible = []

    # B1: Fraction Addition
    if any(check_fraction_add(n) for n in nodes):
        eligible.append(BugID.B1_FRAC_ADD)

    # B2: Binomial Square
    if any(check_binomial_square(n) for n in nodes):
        eligible.append(BugID.B2_BINOM_SQ)

    # B3: Operator Precedence
    if check_precedence(nodes):
        eligible.append(BugID.B3_NO_PREC)

    # B4: Negative Distribution
    if check_negative_distribution(nodes):
        eligible.append(BugID.B4_NEG_BRACKET)

    # B5: Transposition
    if any(check_transposition(n) for n in nodes):
        eligible.append(BugID.B5_TRANSP_SIGN)

    if eligible:
        return ScopeResult(
            in_scope=True,
            eligible_families=tuple(eligible),
            reason="Supported misconception pattern detected.",
        )

    return ScopeResult(
        in_scope=False, eligible_families=(), reason="No supported misconception pattern detected."
    )


if __name__ == "__main__":
    from intake.parse import parse

    examples = [r"\frac{1}{2}+\frac{1}{3}", "(a+b)^2", "2+3*4", "10-(4-1)", "x+3=7", "x+5"]

    for example in examples:
        result = scope_gate(parse(example))

        print("Input:", example)
        print("In Scope:", result.in_scope)
        print("Eligible:", [bug.value for bug in result.eligible_families])
        print("Reason:", result.reason)
        print()
