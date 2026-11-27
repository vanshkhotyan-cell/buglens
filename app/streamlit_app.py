"""BugLens student workspace. All drafts and handoffs belong to one browser session."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

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
    for key in ("submission", "diagnosis", "probe", "reflection", "answer", "explanation"):
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
    with st.container(key="math_canvas"):
        st.subheader("Cognitive conflict")
        st.caption("A targeted counterexample will appear here when remediation is connected.")
        sub = s.submission
        if sub["latex"]:
            st.latex(sub["latex"])
        else:
            st.code(sub["question"], language=None, wrap_lines=True)
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Incorrect reasoning")
        st.caption("Submitted working · error identification pending")
        st.code(sub["attempt"], language=None, wrap_lines=True)
    with right.container(border=True):
        st.subheader("Corrected reasoning")
        st.caption("Verified reasoning steps pending remediation integration.")
    st.button("Continue to practice", type="primary", on_click=navigate, args=(PAGES[2],))


def practice():
    st.title("Practice & reflection")
    if not confirmed():
        return
    # Free-entry questions have no inferred topic. Offer a labeled general exercise.
    topic = s.submission["topic"]
    if topic not in BANK:
        topic = "Fraction addition"
    with st.container(key="math_canvas"):
        st.caption(f"Curated practice · {topic} · adaptive selection pending")
        prompt, _, _, latex = BANK[topic]
        st.subheader(prompt)
        st.latex(latex)
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
