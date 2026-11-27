from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
import sympy as sp

from buglens.pedagogy.remediation import FAMILIES, X, build_lesson
from buglens.schemas import DiagnosisDecision, ReplayOutput

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

CASES = [
    ("2/3 + 1/4", sp.Rational(11, 12), sp.Rational(3, 7), 1, sp.Rational(2, 4)),
    ("(x+3)^2", X**2 + 6 * X + 9, X**2 + 9, 4, 2),
    ("3+2×4", 11, 20, 1, 0),
    ("8-(2-6)", 12, 0, 10, 0),
    ("x+7=2", -5, 9, 0, 10),
]


@pytest.mark.parametrize("family,case", list(zip(FAMILIES, CASES, strict=True)))
def test_four_stage_content(family, case):
    raw, value, wrong, anchor, prediction = case
    lesson = build_lesson(raw)
    assert lesson.family == family
    assert lesson.replay is None
    assert (lesson.original.value, lesson.original.wrong_value) == (value, wrong)
    assert (lesson.anchor.value, lesson.anchor.wrong_value) == (anchor, prediction)
    assert lesson.family.rule and lesson.family.reflection
    assert lesson.original.question in lesson.original.steps
    assert lesson.practice == build_lesson(raw).practice
    assert build_lesson(family.practice[0]).practice.question != lesson.practice.question


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "1/0+2/3",
        "2/3+1/0",
        "x+1000=1",
        "1 2+3*4",
        "__import__('os')",
        "1+2*0;print(1)",
        "9" * 500,
        "(x+1)^200",
    ],
)
def test_unsupported(raw):
    assert build_lesson(raw) is None


def decision(family, action="TEACH", **changes):
    fields = {
        "bug_id": family.replay_id,
        "applicable": True,
        "matches_student": True,
        "matches_correct": False,
        "predicted_latex": "p",
        "trace_steps": ["Example replay step"],
    }
    trace = ReplayOutput(**(fields | changes))
    return DiagnosisDecision(
        posterior={},
        entropy_bits=0,
        action=action,
        target_bug=family.replay_id,
        confidence=1,
        evidence_traces=[trace],
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_live_contract(family):
    result = decision(family)
    lesson = build_lesson(family.anchor, result)
    assert lesson.replay == result.evidence_traces[0]
    for field, value in (
        ("action", "ABSTAIN"),
        ("action", "PROBE"),
        ("target_bug", "slip"),
        ("evidence_traces", []),
    ):
        assert build_lesson(family.anchor, result.model_copy(update={field: value})) is None
    for changes in (
        {"applicable": False},
        {"matches_student": False},
        {"matches_correct": True},
        {"bug_id": "other"},
    ):
        assert build_lesson(family.anchor, decision(family, **changes)) is None


def test_structure_and_coincident_rules():
    assert build_lesson("1/2 + 1/2").original.question == r"\frac{1}{2}+\frac{1}{2}"
    assert build_lesson("1/2 + 1/2").anchor.wrong == r"\frac{2}{4}"
    assert build_lesson("0+2*3", decision(FAMILIES[2])) is None
    assert build_lesson("0+2*3").anchor.value != build_lesson("0+2*3").anchor.wrong_value


def click(ui, label):
    next(b for b in ui.button if b.label == label).click().run()
    assert not ui.exception


@pytest.mark.parametrize(
    "topic,family",
    list(
        zip(
            [
                "Fraction addition",
                "Binomial square",
                "Operator precedence",
                "Bracket subtraction",
                "Equation",
            ],
            FAMILIES,
            strict=True,
        )
    ),
)
def test_fixture_review_and_practice(topic, family):
    ui = AppTest.from_file(str(APP)).run()
    ui.selectbox(key="input_bank_topic").set_value(topic).run()
    ui.text_area(key="input_attempt").set_value("My working").run()
    click(ui, "Confirm question and attempt")
    click(ui, "Open reasoning review")
    assert "Demonstration fixture" in ui.info[0].value
    assert ui.code[0].value == "My working"
    assert all(m.value == "Pending" for m in ui.metric)
    lesson = build_lesson(ui.session_state["submission"]["question"])
    assert any(lesson.anchor.steps in item.value for item in ui.latex)
    click(ui, "Continue to practice")
    assert family.reflection in [item.value for item in ui.markdown]
    ui.text_input(key="input_answer").set_value("My answer").run()
    assert ui.button[0].disabled
    ui.text_area(key="input_explanation").set_value("I checked my steps.").run()
    click(ui, "Save practice response")
    assert ui.button[0].disabled and lesson.practice.steps in ui.latex[-1].value
    ui.run()
    assert ui.session_state["reflection"]["question_latex"] == lesson.practice.question
    ui.radio(key="page").set_value("Mathematical workspace").run()
    ui.text_area(key="input_attempt").set_value("A new attempt").run()
    click(ui, "Confirm question and attempt")
    ui.radio(key="page").set_value("Practice & reflection").run()
    assert not ui.text_input(key="input_answer").value
    assert not ui.text_area(key="input_explanation").value
    assert "reflection" not in ui.session_state


def test_live_and_unsupported_review():
    ui = AppTest.from_file(str(APP)).run()
    ui.text_area(key="input_attempt").set_value("3/7").run()
    click(ui, "Confirm question and attempt")
    ui.session_state["diagnosis"] = decision(FAMILIES[0])
    click(ui, "Open reasoning review")
    assert ui.text[0].value == "Example replay step"
    assert not ui.info
    ui.session_state["diagnosis"] = decision(FAMILIES[0], action="ABSTAIN")
    ui.run()
    assert "unavailable" in ui.info[0].value
    assert not ui.text
    ui.radio(key="page").set_value("Mathematical workspace").run()
    ui.radio(key="input_source").set_value("Type Your Own Question").run()
    ui.text_input(key="input_custom").set_value(r"\input{file}").run()
    click(ui, "Confirm question and attempt")
    click(ui, "Open reasoning review")
    assert "unavailable" in ui.info[0].value
    assert ui.code[0].value == r"\input{file}"
    assert not ui.latex
    click(ui, "Continue to practice")
    assert "General practice" in ui.caption[0].value
