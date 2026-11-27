from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def ui():
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    return app


def button(ui, label):
    return next(item for item in ui.button if item.label == label)


def enter_attempt(ui):
    ui.text_area(key="input_attempt").set_value("3/7").run()
    button(ui, "Confirm question and attempt").click().run()
    assert ui.session_state["submission"]["attempt"] == "3/7"


def test_full_session(ui):
    assert button(ui, "Confirm question and attempt").disabled
    enter_attempt(ui)
    assert button(ui, "Confirm question and attempt").disabled
    button(ui, "Open reasoning review").click().run()
    assert ui.code[0].value == "3/7"
    assert all(metric.value == "Pending" for metric in ui.metric)
    button(ui, "Continue to practice").click().run()
    ui.text_input(key="input_answer").set_value("9/10").run()
    ui.text_area(key="input_explanation").set_value("   ").run()
    assert button(ui, "Save practice response").disabled
    ui.text_area(key="input_explanation").set_value("I used tenths.").run()
    button(ui, "Save practice response").click().run()
    assert ui.session_state["reflection"]["explanation"] == "I used tenths."
    ui.radio(key="page").set_value("Mathematical workspace").run()
    assert ui.text_area(key="input_attempt").value == "3/7"
    ui.radio(key="page").set_value("Practice & reflection").run()
    assert button(ui, "Save practice response").disabled
    ui.text_input(key="input_answer").set_value("0.9").run()
    assert "reflection" not in ui.session_state
    ui.radio(key="page").set_value("Mathematical workspace").run()
    ui.text_area(key="input_attempt").set_value("11/12").run()
    assert "submission" not in ui.session_state
    assert "answer" not in ui.session_state
    ui.radio(key="page").set_value("Reasoning review").run()
    button(ui, "Open workspace").click().run()
    assert ui.title[0].value == "Mathematical workspace"
    assert not ui.exception


def test_changed_topic_and_source_clear_confirmation(ui):
    enter_attempt(ui)
    ui.selectbox(key="input_bank_topic").set_value("Equation").run()
    assert "submission" not in ui.session_state
    button(ui, "Confirm question and attempt").click().run()
    ui.radio(key="input_source").set_value("Type Your Own Question").run()
    assert "submission" not in ui.session_state
    assert button(ui, "Confirm question and attempt").disabled
    text = r"<b>2+3</b> + \input{file}"
    ui.text_input(key="input_custom").set_value(text).run()
    assert ui.code[0].value == text
    button(ui, "Confirm question and attempt").click().run()
    assert ui.session_state["submission"]["latex"] is None
    ui.text_input(key="input_custom").set_value("   ").run()
    assert "submission" not in ui.session_state
    assert button(ui, "Confirm question and attempt").disabled


@pytest.mark.parametrize(
    ("topic", "raw"),
    [
        ("Fraction addition", "2/3 + 1/4"),
        ("Binomial square", "(x+3)^2"),
        ("Bracket subtraction", "10-(5-5)"),
        ("Equation", "x+5=5"),
    ],
)
def test_guided_question(ui, topic, raw):
    ui.radio(key="input_source").set_value("Type Your Own Question").run()
    ui.selectbox(key="input_template").set_value(topic).run()
    enter_attempt(ui)
    assert ui.session_state["submission"]["question"] == raw
    assert ui.session_state["submission"]["latex"] in ui.latex[0].value
    ui.number_input[0].set_value(9).run()
    assert "submission" not in ui.session_state
    assert not ui.exception
