import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from buglens.modeling.fusion import ALL_CLASSES, shannon_entropy_bits
from buglens.pedagogy.diagnostic_adapter import diagnose, observe
from buglens.pedagogy.diagnostic_fixtures import CASES, PROBES

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def click(app, label):
    next(b for b in app.button if b.label == label).click().run()
    assert not app.exception


def scenario(name):
    _, _, replays, bank = CASES[name]
    return diagnose(replays, bank)


def test_missing_vs_completed_replay():
    assert diagnose(None) is None
    abstain = diagnose([]).decision
    assert abstain.action == "ABSTAIN"
    assert abstain.posterior["slip"] == max(abstain.posterior.values())
    assert abstain.confidence < abstain.posterior["slip"]
    teach = scenario("Fraction error").decision
    assert teach.action == "TEACH" and teach.target_bug == "fraction_add"
    assert tuple(teach.posterior) == ALL_CLASSES


@pytest.mark.parametrize("matched", [True, False])
def test_eig_and_observation_share_one_model(matched):
    initial = scenario("Ambiguous working")
    top = initial.selected
    assert top.eig_bits == max(p.eig_bits for p in initial.candidates)
    prior = np.array(list(initial.decision.posterior.values()))
    model = np.array(next(p.match_likelihood for p in PROBES if p.template_id == top.template_id))
    post = observe(initial, top.template_id, matched)
    conditional = prior * (model if matched else 1 - model)
    assert list(post.decision.posterior.values()) == pytest.approx(conditional / conditional.sum())
    assert post.history[0].before == initial.decision.posterior
    assert post.history[0].after == post.decision.posterior
    assert observe(post, top.template_id, matched) is post
    with pytest.raises(ValueError, match="selected probe"):
        observe(initial, "stale", matched)
    expected_entropy = sum(
        (prior @ p) * shannon_entropy_bits(prior * p / (prior @ p)) for p in (model, 1 - model)
    )
    assert top.eig_bits == pytest.approx(shannon_entropy_bits(prior) - expected_entropy)


def test_probe_budget():
    first = scenario("Ambiguous working")
    second = observe(first, first.selected.template_id, False)
    final = observe(second, second.selected.template_id, False)
    assert final.selected is None and len(final.history) == 2


def test_console_demo_isolation_and_reset():
    app = AppTest.from_file(str(APP)).run()
    app.selectbox(key="demo_case").set_value("Ambiguous working").run()
    assert "diagnosis" not in app.session_state
    click(app, "Other response")
    app.radio(key="page").set_value("Reasoning review").run()
    result = app.session_state["demo_result"]
    assert len(result.history) == 1
    assert any("Change" in table.value for table in app.table)
    app.selectbox(key="demo_case").set_value("Fraction error").run()
    assert not app.session_state["demo_result"].history
    assert any("Targeted practice" in c.value for c in app.caption)
    app.selectbox(key="demo_case").set_value("Unmatched response").run()
    assert next(m for m in app.metric if m.label == "Selected policy").value == "ABSTAIN"
    app.radio(key="page").set_value("Mathematical workspace").run()
    app.text_area(key="input_attempt").set_value("changed").run()
    assert "demo_result" not in app.session_state and "diagnosis" not in app.session_state


def test_typed_probe_handoff_and_invalidation():
    app = AppTest.from_file(str(APP)).run()
    app.text_area(key="input_attempt").set_value("test").run()
    click(app, "Confirm question and attempt")
    app.session_state["replay_outputs"] = CASES["Ambiguous working"][2]
    app.session_state["probe_bank"] = PROBES
    click(app, "Open reasoning review")
    assert app.session_state["diagnosis"].action == "PROBE"
    click(app, "Matches 2")
    assert app.session_state["diagnosis"].action == "ABSTAIN"
    app.run()
    assert len(app.session_state["assessment"].history) == 1
    app.radio(key="page").set_value("Mathematical workspace").run()
    app.text_area(key="input_attempt").set_value("new").run()
    assert all(
        k not in app.session_state
        for k in ("replay_outputs", "diagnosis", "probe_bank", "assessment")
    )


def test_coverage_event_once_without_raw_text(caplog):
    app = AppTest.from_file(str(APP)).run()
    app.radio(key="input_source").set_value("Type Your Own Question").run()
    app.text_input(key="input_custom").set_value("private question").run()
    app.text_area(key="input_attempt").set_value("private answer").run()
    click(app, "Confirm question and attempt")
    app.run()
    click(app, "Open reasoning review")
    events = [
        json.loads(r.message) for r in caplog.records if r.name.endswith("diagnostic_adapter")
    ]
    assert events == [
        {"event": "diagnostic_coverage_gap", "reason": "unsupported_remediation_shape"}
    ]
