from .ast import (
    Num, Var, Neg, BinOp, Frac, EqNode
)

def get_precedence(node):

    if isinstance(node, BinOp):

        if node.op in ("+", "-"):
            return 1

        if node.op == "*":
            return 2

        if node.op == "**":
            return 4

    if isinstance(node, Neg):
        return 3

    return 5


def render_number(value):

    if isinstance(value, int):
        return str(value)

    return str(value)


def render_child(node, parent_op, side):

    result = _render(node)

    child_precedence = get_precedence(node)

    parent_precedence = {
        "+": 1,
        "-": 1,
        "*": 2,
        "**": 4,
        "neg": 3
    }[parent_op]

    need_brackets = child_precedence < parent_precedence

    if child_precedence == parent_precedence:

        if parent_op == "-" and side == "right":
            need_brackets = True

        elif parent_op == "**" and side == "left":
            need_brackets = True

        elif parent_op in ("+", "-") and side == "right":
            need_brackets = isinstance(node, BinOp)

    if need_brackets:
        return r"\left(" + result + r"\right)"

    return result


def _render(node):

    if isinstance(node, Num):
        return render_number(node.value)

    elif isinstance(node, Var):
        return node.name

    elif isinstance(node, Neg):

        child = render_child(node.child, "neg", "right")

        return "-" + child

    elif isinstance(node, Frac):

        numerator = _render(node.num)
        denominator = _render(node.den)

        return (
            r"\frac{" + numerator + "}{" +
            denominator + "}"
        )

    elif isinstance(node, BinOp):

        left = render_child(node.left, node.op, "left")
        right = render_child(node.right, node.op, "right")

        if node.op == "+":
            return left + " + " + right

        elif node.op == "-":
            return left + " - " + right

        elif node.op == "*":
            return left + r" \cdot " + right

        elif node.op == "**":
            return left + "^{" + _render(node.right) + "}"

    elif isinstance(node, EqNode):

        left = _render(node.lhs)
        right = _render(node.rhs)

        return left + " = " + right

    raise TypeError("Unsupported AST node")


def render_ast_latex(node):

    return _render(node)


if __name__ == "__main__":

    from intake.parse import parse

    tests = [
        r"\frac{1}{2}",
        "2 + 3 * 4",
        "(x+1)^2",
        "x + 5 = 12",
        "-(a+b)"
    ]

    for expression in tests:

        node = parse(expression)

        print("Input:", expression)
        print("LaTeX:", render_ast_latex(node))
        print()