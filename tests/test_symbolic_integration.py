from buglens.symbolic.parser import parse
from buglens.symbolic.scope import scope_gate
from buglens.symbolic.operators import OPERATORS
from buglens.schemas import BugID


test_cases = [
    (r"\frac{1}{2}+\frac{1}{3}", BugID.B1_FRAC_ADD, "2/5"),
    ("(2+3)^2", BugID.B2_BINOM_SQ, "13"),
    ("2+3*4", BugID.B3_NO_PREC, "20"),
    ("10-(4-1)", BugID.B4_NEG_BRACKET, "5"),
    ("x+3=7", BugID.B5_TRANSP_SIGN, "10"),
]

passed = 0

print("BUGLENS PACKAGE INTEGRATION TEST\n")

for expression, bug_id, expected in test_cases:
    node = parse(expression)

    scope = scope_gate(node)

    operator_result = OPERATORS[bug_id](node)

    success = (
        scope.in_scope
        and bug_id in scope.eligible_families
        and operator_result.applicable
        and operator_result.predicted_expression == expected
        and len(operator_result.trace_steps) > 0
    )

    if success:
        print("PASS:", bug_id.value)
        passed += 1
    else:
        print("FAIL:", expression)
        print("Scope:", scope)
        print("Operator:", operator_result)

print(f"\nTotal Passed: {passed}/5")
