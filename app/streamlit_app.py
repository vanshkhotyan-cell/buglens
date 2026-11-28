"""BugLens student workspace. All drafts and handoffs belong to one browser session."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import streamlit as st

from buglens.pedagogy.bkt_tracer import DEFAULT_DB, MisconceptionTracer
from buglens.pedagogy.diagnostic_adapter import diagnose, log_coverage_gap, observe
from buglens.pedagogy.diagnostic_fixtures import CASES
from buglens.pedagogy.remediation import build_lesson
from buglens.pedagogy.review_queue import classify, review_queue

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
PAGES = [
    "Mathematical workspace",
    "Reasoning review",
    "Practice & reflection",
    "Learner profile",
    "Research analytics",
]
s = st.session_state

st.html(Path(__file__).with_name("workspace.css"))


def navigate(page):
    s.page = page


def invalidate():
    for key in (
        "submission",
        "diagnosis",
        "probe",
        "assessment",
        "replay_outputs",
        "probe_bank",
        "demo_case",
        "demo_result",
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
                "id": uuid4().hex,
                "source": source,
                "topic": topic,
                "question": raw,
                "latex": latex,
                "attempt": attempt,
            }
            if build_lesson(raw) is None:
                log_coverage_gap()
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
    decision = s.get("diagnosis")
    lesson = build_lesson(sub["question"], decision)
    if decision and decision.action == "PROBE":
        st.subheader("Follow-up question")
        probe = s.get("probe")
        if probe:
            st.latex(probe.question_latex)
            if s.get("assessment") and s.assessment.selected:
                response_controls(s.assessment, demo=False)
            else:
                st.caption("Response likelihoods unavailable.")
        else:
            st.info("A follow-up is needed. A verified probe is unavailable.")
    elif decision and decision.action == "ABSTAIN":
        st.info("Targeted teaching is unavailable because the diagnostic policy abstained.")
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
    decision = s.get("diagnosis")
    lesson = build_lesson(sub["question"], decision)
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
        if lesson and lesson.replay and "learner" in s:
            st.caption(
                "Your first saved answer counts toward learner memory. Use a number, fraction or expanded polynomial."
            )
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
            if lesson and lesson.replay and "learner" in s:
                store().observe(
                    s.learner,
                    s.memory_session,
                    lesson.family.replay_id,
                    sub["id"],
                    classify(lesson.practice, answer),
                    kind="practice",
                    completed=True,
                )
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


def response_controls(result, *, demo):
    probe = result.selected
    prefix = "demo" if demo else "confirmed"
    answer = next(iter(probe.expected_answers.values()))
    for matched, label in ((True, f"Matches {answer}"), (False, "Other response")):
        if st.button(label, key=f"{prefix}_{probe.template_id}_{matched}"):
            updated = observe(result, probe.template_id, matched)
            if demo:
                s.demo_result = updated
            else:
                s.assessment = updated
                s.diagnosis = updated.decision
                s.probe = updated.selected
            st.rerun()


def metrics(decision, result, probe):
    if decision is None:
        st.caption("Posterior, maximum probability, Shannon entropy, EIG and policy unavailable.")
        return
    st.markdown("**Posterior probabilities**")
    st.json(decision.posterior)
    maximum, entropy, policy = st.columns(3)
    maximum.metric(
        "Maximum posterior",
        max(decision.posterior.values()) if decision.posterior else "Unavailable",
    )
    entropy.metric("Shannon entropy (bits)", decision.entropy_bits)
    policy.metric("Selected policy", decision.action)
    if result and result.candidates:
        st.table(
            [{"Candidate": p.template_id, "EIG (bits)": p.eig_bits} for p in result.candidates]
        )
    elif probe:
        st.metric("Supplied probe EIG (bits)", probe.eig_bits)
    else:
        st.caption("Candidate EIG scores unavailable.")
    st.markdown("**Probe history**")
    if result and result.history:
        for item in result.history:
            st.caption(
                f"{item.template_id} · {'matched predicted answer' if item.matched else 'other response'}"
            )
            st.table(
                [
                    {"Class": c, "Before": p, "After": item.after[c], "Change": item.after[c] - p}
                    for c, p in item.before.items()
                ]
            )
    else:
        st.caption("No recorded observations. Posterior changes unavailable.")


def judge():
    with st.expander("Under the Hood · Judge console"):
        st.caption("Confirmed attempt · live diagnostic handoff")
        metrics(s.get("diagnosis"), s.get("assessment"), s.get("probe"))
        st.divider()
        case = st.selectbox("Contract demonstration", ["Off", *CASES], key="demo_case")
        if case == "Off":
            s.pop("demo_result", None)
            s.pop("active_case", None)
            return
        question, answer, replays, bank = CASES[case]
        if case != s.get("active_case") or "demo_result" not in s:
            s.demo_result = diagnose(replays, bank)
            s.active_case = case
        result = s.demo_result
        st.info("Contract fixture · simulated evidence for a separate worked example.")
        st.write(f"Fixture question: {question} · Scripted answer: {answer}")
        st.caption("Inputs · uniform prior, scripted replay and binary probe likelihoods")
        st.json(
            {
                "replay": [r.model_dump() for r in replays],
                "match_likelihood_by_class": {
                    p.template_id: dict(
                        zip(result.decision.posterior, p.match_likelihood, strict=True)
                    )
                    for p in bank
                },
            },
            expanded=False,
        )
        st.caption("Computed from fixture inputs · symbolic-only fusion and selective policy")
        metrics(result.decision, result, result.selected)
        if result.selected:
            st.write(f"Selected probe · {result.selected.template_id}")
            st.latex(result.selected.question_latex)
            st.caption("Choose a simulated response to the fixture probe.")
            response_controls(result, demo=True)
        elif result.decision.action == "TEACH":
            lesson = build_lesson(question, result.decision)
            if lesson:
                st.write(lesson.family.rule)
                st.latex(lesson.original.steps)
                st.caption("Targeted practice · fixture")
                st.latex(lesson.practice.question)
            else:
                st.caption("A verified lesson for this fixture question is unavailable.")
        else:
            st.caption("ABSTAIN · targeted teaching withheld.")


def store():
    if "memory_store" not in s:
        s.memory_store = MisconceptionTracer(os.environ.get("BUGLENS_LEARNER_DB", DEFAULT_DB))
    return s.memory_store


def attach(db, learner):
    invalidate()
    s.learner = learner
    s.memory_session = uuid4().hex
    db.start_session(learner, s.memory_session)


def memory_panel(db, learner, session, prefix):
    prior = db.prior(learner)
    events = db.history(learner)
    sessions = db.sessions(learner)
    older = next(
        (item for i, item in enumerate(sessions) if i and sessions[i - 1]["id"] == session), None
    )
    st.caption(f"Previous session (UTC) · {older['started'] if older else 'First session'}")
    st.table(
        [
            {"Family": family, "Estimated active probability": f"{p:.1%}"}
            for family, p in prior.active.items()
        ]
    )
    attempts = [e for e in events if e["kind"] in ("practice", "diagnostic", "review")]
    learned = sum(e["kind"] == "learn" for e in events)
    verified = sum(e["evidence"] in ("bug", "correct") for e in attempts)
    st.write(
        f"{len(attempts)} recorded attempts · {verified} classified responses · "
        f"{learned} completed remediations"
    )
    if events:
        with st.expander("Recent probability changes"):
            st.table(
                [
                    {k: e[k] for k in ("created", "family", "kind", "evidence", "before", "after")}
                    for e in events[:20]
                ]
            )
    st.subheader("Spaced review")
    queue = review_queue(db, learner, session)
    if not queue:
        st.caption("No recently remediated families are due this session.")
        return
    item = queue[0]
    st.caption("Due families · " + ", ".join(q.family for q in queue))
    st.latex(item.example.question)
    key = f"{prefix}_{session}_{item.family}"
    response = st.text_input("Review answer", key=key, max_chars=100)
    st.caption("Use a number, fraction or expanded polynomial such as x^2+8x+16.")
    if st.button("Submit review", key=key + "_submit", disabled=not response.strip()):
        db.observe(
            learner, session, item.family, key, classify(item.example, response), kind="review"
        )
        st.rerun()


def profile():
    st.title("Learner profile")
    st.caption(
        "Model estimates use provisional BKT parameters. Mastery and predictive accuracy have not been validated."
    )
    db = store()
    if "learner" not in s:
        if st.button("Create anonymous profile"):
            attach(db, db.create_learner())
            st.rerun()
        code = st.text_input(
            "Resume with your private anonymous learner ID", type="password", max_chars=32
        )
        if st.button("Resume profile", disabled=not code):
            if db.prior(code).active:
                attach(db, code)
                st.rerun()
            st.warning("Learner ID unavailable.")
    else:
        st.caption("Private anonymous learner ID · Save this ID to resume in another session.")
        st.code(s.learner)
        st.caption(
            "Learner history uses verified live lesson matches and classified practice or review responses. Live replay and neural inference are currently unavailable."
        )
        memory_panel(db, s.learner, s.memory_session, "live")
        confirmed_delete = st.checkbox("I confirm deletion of this learner's stored history")
        if st.button("Clear learner data", disabled=not confirmed_delete):
            db.clear_learner(s.learner, confirmed=confirmed_delete)
            s.clear()
            st.rerun()
    with st.expander("Demonstration-only learner memory"):
        st.caption("Scripted fraction and equation attempts in a separate temporary database.")
        if "demo_memory" not in s:
            if st.button("Load demonstration history"):
                directory = TemporaryDirectory()
                demo = MisconceptionTracer(Path(directory.name) / "demo.db")
                learner = demo.create_learner()
                demo.start_session(learner, "lesson")
                for family in ("fraction_add", "transposition_sign"):
                    demo.observe(learner, "lesson", family, family, "bug")
                    demo.observe(
                        learner,
                        "lesson",
                        family,
                        family,
                        "correct",
                        kind="practice",
                        completed=True,
                    )
                demo.start_session(learner, "review")
                s.demo_memory = directory, learner
                st.rerun()
        else:
            directory, learner = s.demo_memory
            memory_panel(
                MisconceptionTracer(Path(directory.name) / "demo.db"), learner, "review", "demo"
            )


def research():
    st.title("Research analytics")
    st.caption("Phase 5 · Simulated learner study")
    directory = Path(__file__).resolve().parents[1] / "reports" / "simulation"
    try:
        report = json.loads((directory / "summary.json").read_text())
    except (OSError, ValueError):
        st.info("Simulation results unavailable. Generate the study outputs to view this page.")
        return
    cfg = report["config"]
    scenario = st.selectbox("Simulation scenario", list(cfg["scenarios"]))
    st.write(
        f"{cfg['cohort']} learners per group per seed · {len(cfg['seeds'])} seeds · "
        f"{cfg['sessions']} sessions · {cfg['budget']} activity slots per active session"
    )
    st.write(
        "Each group starts with copies of the same learners. Probes use a slot. "
        "Practice and review use the same learning probability in both active groups."
    )
    rows = [r for r in report["groups"] if r["scenario"] == scenario]
    st.dataframe(
        [
            {
                "Group": r["group"],
                "Pre-test": r["pre_mean"],
                "Post-test": r["post_mean"],
                "Mean normalized gain": r["gain_mean"],
                "Error recurrence": r["recurrence"],
                "Median sessions to first resolution": r["median_resolution_session"],
                "Restricted mean sessions": r["restricted_mean_sessions"],
                "Initially active": r["initial"],
                "Censored": r["censored"],
            }
            for r in rows
        ],
        hide_index=True,
    )
    st.caption(
        "Censored families stayed active through the final session. A blank median means "
        "fewer than half reached first resolution. Later relapse is possible."
    )
    with st.expander("Model assumptions and paired differences"):
        st.json(
            {"population": cfg["scenarios"][scenario], "tutor": cfg["tutor"], "seeds": cfg["seeds"]}
        )
        st.dataframe([r for r in report["contrasts"] if r["scenario"] == scenario], hide_index=True)
    for name, caption in (
        ("gain_mean", "Mean normalized gain and 95% learner-bootstrap intervals"),
        ("recurrence", "Fraction of pre-test errors repeated on the post-test"),
        ("curve", "Time to first resolution across all scenarios"),
    ):
        chart = directory / f"{name}.png"
        if chart.is_file():
            st.image(str(chart), caption=caption)
        else:
            st.info(f"Visualization unavailable · {caption}")
    st.write(
        "These simulations cannot establish educational efficacy. Binary responses approximate "
        "student work. The model omits differences in item difficulty and individual responses "
        "to explanations. Live symbolic replay and neural diagnosis remain integration work."
    )


with st.sidebar:
    st.html('<div class="brand"><i aria-hidden="true"></i>BugLens</div>')
    st.caption("Mathematical diagnostic tutor")
    st.divider()
    page = st.radio("Workspace navigation", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("**Team InferenceX**")
    st.caption("IIT Bombay Techfest")

if "submission" in s and "replay_outputs" in s and "diagnosis" not in s:
    result = diagnose(s.replay_outputs, s.get("probe_bank", ()))
    if result is not None:
        s.assessment = result
        s.diagnosis = result.decision
        s.probe = result.selected

if "learner" in s:
    if not store().prior(s.learner).active:
        s.clear()
        st.rerun()
    if "submission" in s and s.get("diagnosis"):
        lesson = build_lesson(s.submission["question"], s.diagnosis)
        if lesson and lesson.replay:
            store().observe(
                s.learner, s.memory_session, lesson.family.replay_id, s.submission["id"], "bug"
            )

{PAGES[0]: workspace, PAGES[1]: review, PAGES[2]: practice, PAGES[3]: profile, PAGES[4]: research}[
    page
]()
if page != PAGES[4]:
    judge()
