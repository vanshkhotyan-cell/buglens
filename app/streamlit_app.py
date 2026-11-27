"""BugLens student workspace. All drafts and handoffs belong to one browser session."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from buglens.pedagogy.remediation import build_lesson
from buglens.schemas import DiagnosisDecision, ProbeCandidate

st.set_page_config(page_title="BugLens · InferenceX", page_icon="🔍", layout="wide")

# Each item carries a worked-on question and a separate, authored practice prompt.
BANK = {
    "Fraction addition": (
        "Add the fractions.",
        "2/3 + 1/4",
        r"\frac{2}{3}+\frac{1}{4}",
        r"\frac{1}{2}+\frac{2}{5}",
    ),
    "Binomial square": ("Expand the square.", "(x+3)^2", "(x+3)^2", "(x+4)^2"),
    "Operator precedence": ("Evaluate the expression.", "1+2×0", r"1+2\times0", r"8-3\times2"),
    "Bracket subtraction": ("Simplify the expression.", "10-(5-5)", "10-(5-5)", "12-(7-3)"),
    "Equation": ("Solve for x.", "x+5=5", "x+5=5", "x+7=12"),
}
TEMPLATES = [
    "Free entry",
    "Fraction addition",
    "Binomial square",
    "Bracket subtraction",
    "Equation",
]
PAGES = ["Mathematical workspace", "Reasoning review", "Practice & reflection"]
s = st.session_state

st.html(Path(__file__).with_name("workspace.css"))


def navigate(page):
    s.page = page


def invalidate():
    for key in (
        "submission",
        "diagnosis",
        "probe",
        "reflection",
        "answer",
        "explanation",
        "input_answer",
        "input_explanation",
        "practice_context",
    ):
        s.pop(key, None)


def field(widget, label, key, default, *, practice=False, **kwargs):
    """Retain draft values when Streamlit removes widgets hidden by navigation."""
    widget_key = f"input_{key}"
    if widget_key not in s:
        s[widget_key] = s.get(key, default)

    def changed():
        s[key] = s[widget_key]
        if practice:
            s.pop("reflection", None)
        else:
            invalidate()

    return widget(label, key=widget_key, on_change=changed, **kwargs)


def guided_question(topic):
    if topic == "Fraction addition":
        labels = ["First numerator", "First denominator", "Second numerator", "Second denominator"]
        defaults = [2, 3, 1, 4]
    elif topic == "Bracket subtraction":
        labels, defaults = (
            ["Starting value", "First bracket value", "Second bracket value"],
            [10, 5, 5],
        )
    elif topic == "Equation":
        labels, defaults = ["Add to x", "Right-hand side"], [5, 5]
    else:
        labels, defaults = ["Add to x"], [3]

    values = []
    for i, (column, label, default) in enumerate(zip(st.columns(len(labels)), labels, defaults)):
        with column:
            values.append(
                field(
                    st.number_input,
                    label,
                    f"{topic}_{i}",
                    default,
                    min_value=1 if "denominator" in label else 0,
                    max_value=999,
                )
            )
    a = values[0]
    if topic == "Fraction addition":
        a, b, c, d = values
        return f"{a}/{b} + {c}/{d}", rf"\frac{{{a}}}{{{b}}}+\frac{{{c}}}{{{d}}}"
    if topic == "Bracket subtraction":
        a, b, c = values
        return f"{a}-({b}-{c})", f"{a}-({b}-{c})"
    if topic == "Equation":
        return f"x+{a}={values[1]}", f"x+{a}={values[1]}"
    return f"(x+{a})^2", f"(x+{a})^2"


def workspace():
    st.title("Mathematical workspace")
    with st.container(key="question_controls"):
        source = field(
            st.radio,
            "Question source",
            "source",
            "Curated Question Bank",
            options=["Curated Question Bank", "Type Your Own Question"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if source == "Curated Question Bank":
            topic = field(
                st.selectbox, "Select a topic", "bank_topic", next(iter(BANK)), options=list(BANK)
            )
            prompt, raw, latex, _ = BANK[topic]
        else:
            topic = field(
                st.selectbox, "Input template", "template", TEMPLATES[0], options=TEMPLATES
            )
            if topic == "Free entry":
                raw = field(
                    st.text_input,
                    "Your question",
                    "custom",
                    "",
                    max_chars=300,
                    placeholder="For example, simplify 3(x + 2) − x.",
                ).strip()
                prompt, latex = "Question preview", None
            else:
                raw, latex = guided_question(topic)
                prompt = BANK[topic][0]
    with st.container(key="math_canvas"):
        st.subheader(prompt)
        if latex:
            st.latex(latex)
        elif raw:
            st.code(raw, language=None, wrap_lines=True)
        else:
            st.caption("Enter a question above to preview it here.")

    with st.container(border=True, key="response_card"):
        st.subheader("Your attempt")
        attempt = field(
            st.text_area,
            "Answer or working",
            "attempt",
            "",
            max_chars=1500,
            label_visibility="collapsed",
            placeholder="Write the steps you took, including your final answer.",
        ).strip()
        if attempt:
            st.caption("Attempt preview")
            st.code(attempt, language=None, wrap_lines=True)
        if st.button(
            "Confirm question and attempt",
            type="primary",
            disabled=not (raw and attempt) or "submission" in s,
        ):
            # Raw handoff for a future parser. Only authored or integer-built LaTeX is rendered.
            s.submission = {
                "source": source,
                "topic": topic,
                "question": raw,
                "latex": latex,
                "attempt": attempt,
            }
            st.rerun()
        if "submission" in s:
            st.success("Attempt confirmed for this session.")
            st.button("Open reasoning review", on_click=navigate, args=(PAGES[1],))


def confirmed():
    if "submission" in s:
        return True
    st.info("Confirm a question and attempt in the mathematical workspace to continue.")
    st.button("Open workspace", on_click=navigate, args=(PAGES[0],))
    return False


def review():
    st.title("Reasoning review")
    if not confirmed():
        return
    sub = s.submission
    lesson = build_lesson(sub["question"], s.get("diagnosis"))
    with st.container(key="math_canvas"):
        st.subheader("Dissatisfaction · Test the rule")
        if lesson:
            if lesson.replay is None:
                st.info(
                    "Demonstration fixture · This worked example illustrates a common rule error. Your attempt has no live diagnosis."
                )
            st.caption("Try both rules on this counterexample. Which result fits the question?")
            st.latex(lesson.anchor.question)
            wrong, correct = st.columns(2)
            wrong.caption("Misconception prediction")
            wrong.latex(lesson.anchor.wrong)
            correct.caption("Verified calculation")
            correct.latex(lesson.anchor.steps)
        else:
            st.info("Targeted remediation is unavailable for this submission.")
            if sub["latex"]:
                st.latex(sub["latex"])
            else:
                st.code(sub["question"], language=None, wrap_lines=True)
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Your submitted working")
        st.code(sub["attempt"], language=None, wrap_lines=True)
        if lesson and lesson.replay:
            st.caption("Live replay · matched incorrect reasoning")
            for step in lesson.replay.trace_steps:
                st.text(step)
            st.latex(lesson.replay.predicted_latex)
        elif lesson:
            st.caption("Demonstration fixture · sample incorrect reasoning")
            st.write(lesson.family.mistake)
            st.latex(lesson.original.wrong)
            if lesson.original.value == lesson.original.wrong_value:
                st.caption(
                    "Both rules give the same answer here. The counterexample above separates them."
                )
    with right.container(border=True):
        st.subheader("Corrected reasoning")
        if lesson:
            st.caption("Intelligibility · Why the rule works")
            st.write(lesson.family.rule)
            st.caption("Plausibility · Check the original problem")
            st.latex(lesson.original.steps)
        else:
            st.caption(
                "A verified correction requires a supported question and an eligible teaching decision when a diagnosis is available."
            )
    st.button("Continue to practice", type="primary", on_click=navigate, args=(PAGES[2],))


def practice():
    st.title("Practice & reflection")
    if not confirmed():
        return
    sub = s.submission
    lesson = build_lesson(sub["question"], s.get("diagnosis"))
    if lesson:
        latex = lesson.practice.question
        prompt = "Try this question using the same rule."
        label = (
            "Targeted practice" if lesson.replay else "Demonstration fixture · targeted practice"
        )
    else:
        topic = sub["topic"] if sub["topic"] in BANK else "Fraction addition"
        prompt, _, _, latex = BANK[topic]
        label = "General practice · no targeted lesson available"
    context = (sub["question"], sub["attempt"], label, latex)
    if s.get("practice_context") != context:
        for key in ("answer", "explanation", "input_answer", "input_explanation", "reflection"):
            s.pop(key, None)
        s.practice_context = context
    with st.container(key="math_canvas"):
        st.caption(f"Fruitfulness · {label}")
        st.subheader(prompt)
        st.latex(latex)
        if lesson:
            st.write(lesson.family.reflection)
    with st.container(border=True, key="response_card"):
        answer = field(st.text_input, "Practice answer", "answer", "", practice=True, max_chars=300)
        explanation = field(
            st.text_area,
            "Self-explanation (required)",
            "explanation",
            "",
            practice=True,
            max_chars=1500,
            placeholder="Which rule did you use, and how did you check your answer?",
        )
        if st.button(
            "Save practice response",
            type="primary",
            disabled=not (answer.strip() and explanation.strip()) or "reflection" in s,
        ):
            s.reflection = {
                "question_latex": latex,
                "answer": answer.strip(),
                "explanation": explanation.strip(),
            }
            st.rerun()
        if "reflection" in s:
            st.success("Answer and self-explanation saved for this session.")
            if lesson:
                st.caption("Compare your reasoning with the verified solution.")
                st.latex(lesson.practice.steps)


def judge(decision: DiagnosisDecision | None, probe: ProbeCandidate | None):
    with st.expander("Under the Hood · Judge console"):
        st.caption("Evidence for the confirmed attempt")
        st.markdown("**Posterior probabilities**")
        if decision:
            st.json(decision.posterior)
        else:
            st.caption("Pending diagnostic integration")
        entropy, gain, policy = st.columns(3)
        entropy.metric("Entropy (bits)", decision.entropy_bits if decision else "Pending")
        gain.metric("Expected information gain (bits)", probe.eig_bits if probe else "Pending")
        policy.metric("Active policy", decision.action if decision else "Pending")


with st.sidebar:
    st.html('<div class="brand"><i aria-hidden="true"></i>BugLens</div>')
    st.caption("Mathematical diagnostic tutor")
    st.divider()
    page = st.radio("Workspace navigation", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("**Team InferenceX**")
    st.caption("IIT Bombay Techfest")

{PAGES[0]: workspace, PAGES[1]: review, PAGES[2]: practice}[page]()
judge(s.get("diagnosis"), s.get("probe"))
