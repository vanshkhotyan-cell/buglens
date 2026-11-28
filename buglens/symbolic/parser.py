import ast as py_ast
import math
import re

from .ast import Num, Var, Neg, BinOp, Frac, EqNode
from ..schemas import ScopeOutException


ALLOWED_NAMES = {
    "x", "y", "z", "a", "b", "c",
    "m", "n", "p", "q", "r", "t"
}

MAX_INPUT_LENGTH = 120
MAX_EXPONENT = 4


# Step 1: Check balanced brackets and braces

def validate_balanced(text):

    pairs = {
        ")": "(",
        "}": "{"
    }

    stack = []

    for char in text:

        if char in "({":
            stack.append(char)

        elif char in ")}":

            if not stack or stack.pop() != pairs[char]:
                raise ScopeOutException(
                    "Unbalanced parentheses or braces"
                )

    if stack:
        raise ScopeOutException(
            "Unbalanced parentheses or braces"
        )


# Step 2: Read balanced LaTeX groups

def read_braced_group(text, start):

    if start >= len(text) or text[start] != "{":
        raise ScopeOutException("Invalid LaTeX fraction")

    depth = 0

    for i in range(start, len(text)):

        if text[i] == "{":
            depth += 1

        elif text[i] == "}":

            depth -= 1

            if depth == 0:
                return text[start + 1:i], i + 1

    raise ScopeOutException("Unclosed fraction brace")


# Step 3: Normalize nested fractions

def normalize_fractions(text):

    output = []
    i = 0

    while i < len(text):

        if text.startswith(r"\frac", i):

            j = i + 5

            while j < len(text) and text[j].isspace():
                j += 1

            numerator, j = read_braced_group(text, j)

            while j < len(text) and text[j].isspace():
                j += 1

            denominator, j = read_braced_group(text, j)

            numerator = normalize_fractions(numerator)
            denominator = normalize_fractions(denominator)

            output.append(
                f"(({numerator})/({denominator}))"
            )

            i = j

        else:

            output.append(text[i])
            i += 1

    return "".join(output)


# Step 4: Normalize mathematical symbols

def normalize_symbols(text):

    text = text.replace("×", "*")
    text = text.replace("÷", "/")
    text = text.replace("−", "-")
    text = text.replace("–", "-")

    text = text.replace(r"\cdot", "*")
    text = text.replace(r"\times", "*")

    text = text.replace("^", "**")

    text = normalize_fractions(text)

    return text


# Step 5: Insert implicit multiplication

TOKEN_PATTERN = re.compile(
    r"(?P<NUMBER>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<NAME>[A-Za-z_]\w*)"
    r"|(?P<POWER>\*\*)"
    r"|(?P<OP>[()+\-*/])"
)


def insert_implicit_multiplication(text):

    tokens = []
    position = 0

    for match in TOKEN_PATTERN.finditer(text):

        gap = text[position:match.start()]

        if gap.strip():
            raise ScopeOutException("Unsupported character")

        kind = match.lastgroup
        value = match.group()

        if kind == "OP":

            if value == "(":
                kind = "LPAREN"

            elif value == ")":
                kind = "RPAREN"

        if tokens:

            previous_kind = tokens[-1][0]

            should_insert = (
                previous_kind in {"NUMBER", "NAME", "RPAREN"}
                and kind in {"NAME", "LPAREN"}
            ) or (
                previous_kind == "RPAREN"
                and kind == "NUMBER"
            )

            if should_insert:
                tokens.append(("OP", "*"))

        tokens.append((kind, value))
        position = match.end()

    if text[position:].strip():
        raise ScopeOutException("Unsupported character")

    return "".join(value for _, value in tokens)


# Step 6: Convert Python AST into our custom AST

def translate_ast(node, depth=0):

    if depth > 40:
        raise ScopeOutException("Expression is too deeply nested")

    # Numbers
    if isinstance(node, py_ast.Constant):

        value = node.value

        if type(value) not in (int, float):
            raise ScopeOutException("Invalid number")

        if isinstance(value, float) and not math.isfinite(value):
            raise ScopeOutException("Number must be finite")

        return Num(value)

    # Variables
    if isinstance(node, py_ast.Name):

        if node.id not in ALLOWED_NAMES:
            raise ScopeOutException(
                f"Unsupported variable: {node.id}"
            )

        return Var(node.id)

    # Negative and positive signs
    if isinstance(node, py_ast.UnaryOp):

        if isinstance(node.op, py_ast.USub):
            return Neg(
                translate_ast(node.operand, depth + 1)
            )

        elif isinstance(node.op, py_ast.UAdd):
            return translate_ast(node.operand, depth + 1)

        raise ScopeOutException("Unsupported unary operator")

    # Mathematical operations
    if isinstance(node, py_ast.BinOp):

        left = translate_ast(node.left, depth + 1)
        right = translate_ast(node.right, depth + 1)

        # Division becomes an explicit Fraction node
        if isinstance(node.op, py_ast.Div):
            return Frac(left, right)

        operator_map = {
            py_ast.Add: "+",
            py_ast.Sub: "-",
            py_ast.Mult: "*",
            py_ast.Pow: "**"
        }

        operator = operator_map.get(type(node.op))

        if operator is None:
            raise ScopeOutException("Unsupported operation")

        # Exponent limit
        if isinstance(node.op, py_ast.Pow):

            exponent = node.right

            if not (
                isinstance(exponent, py_ast.Constant)
                and type(exponent.value) is int
                and 0 <= exponent.value <= MAX_EXPONENT
            ):

                raise ScopeOutException(
                    f"Exponent must be between 0 and {MAX_EXPONENT}"
                )

        return BinOp(operator, left, right)

    # Everything else is rejected
    raise ScopeOutException(
        f"Unsupported syntax: {type(node).__name__}"
    )


# Step 7: Parse one mathematical expression

def parse_side(expression):

    expression = normalize_symbols(expression)

    expression = insert_implicit_multiplication(expression)

    try:

        parsed = py_ast.parse(
            expression,
            mode="eval"
        )

    except (SyntaxError, ValueError, RecursionError) as error:

        raise ScopeOutException(
            f"Invalid expression: {error}"
        ) from error

    return translate_ast(parsed.body)


# Step 8: Main parser

def parse(text):

    if not isinstance(text, str):
        raise ScopeOutException("Input must be text")

    if not text.strip():
        raise ScopeOutException("Expression cannot be empty")

    if len(text) > MAX_INPUT_LENGTH:
        raise ScopeOutException("Maximum input length is 120 characters")

    validate_balanced(text)

    if any(char in text for char in "<>≤≥≠!"):
        raise ScopeOutException("Inequalities are not supported")

    if re.search(r"\\(?:int|sin|lim)\b", text):
        raise ScopeOutException("Calculus and trigonometry are not supported")

    equality_count = text.count("=")

    if equality_count > 1:
        raise ScopeOutException("Multiple equality signs are not allowed")

    # Equation handling
    if equality_count == 1:

        left, right = text.split("=")

        left = left.strip()
        right = right.strip()

        if not left or not right:
            raise ScopeOutException("Both equation sides are required")

        return EqNode(
            parse_side(left),
            parse_side(right)
        )

    return parse_side(text.strip())


# Testing

if __name__ == "__main__":

    tests = [
        "2 + 3 * 4",
        "2x",
        "2(x+1)",
        "(a)(b)",
        "2 × 3",
        "6 ÷ 2",
        r"\frac{1}{2}",
        r"\frac{1}{2}+\frac{1}{3}",
        "x + 5 = 12",
        "(x+1)^4"
    ]

    for expression in tests:

        print("Input:", expression)
        print("AST:", parse(expression))
        print()