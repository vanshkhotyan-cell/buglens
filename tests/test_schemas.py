from buglens.schemas import (
    DiagnosisDecision,
    ParsedSubmission,
    ProbeCandidate,
    ReplayOutput,
)


def test_core_schemas_instantiate():
    sub = ParsedSubmission(
        raw_input="1/2 + 1/3 = 2/5",
        normalized_str="1/2 + 1/3",
        ast_json={"op": "Add"},
        sympy_latex=r"\frac{1}{2} + \frac{1}{3}",
        in_scope=True,
        eligible_families=["fraction_add"],
        decline_reason=None,
    )
    trace = ReplayOutput(
        bug_id="fraction_add",
        applicable=True,
        matches_student=True,
        matches_correct=False,
        predicted_latex=r"\frac{2}{5}",
        trace_steps=["(1+1)/(2+3) = 2/5"],
    )
    decision = DiagnosisDecision(
        posterior={"fraction_add": 0.90, "slip": 0.10},
        entropy_bits=0.47,
        action="TEACH",
        target_bug="fraction_add",
        confidence=0.90,
        evidence_traces=[trace],
    )
    probe = ProbeCandidate(
        template_id="frac_add_01",
        question_latex=r"\frac{2}{3} + \frac{1}{4}",
        worded_prompt="What is 2/3 + 1/4?",
        expected_answers={"correct": r"\frac{11}{12}", "fraction_add": r"\frac{3}{7}"},
        eig_bits=0.72,
        verified_distinct=True,
    )
    assert sub.in_scope is True
    assert decision.action == "TEACH"
    assert probe.verified_distinct is True
