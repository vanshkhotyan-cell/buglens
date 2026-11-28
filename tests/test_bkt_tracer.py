from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from buglens.pedagogy.bkt_tracer import MisconceptionTracer, Parameters
from buglens.pedagogy.diagnostic_fixtures import CASES

F = "fraction_add"


def test_observations_transitions_reopen_and_isolation(tmp_path):
    path = tmp_path / "nested" / "store.db"
    db = MisconceptionTracer(path)
    a, b = db.create_learner(), db.create_learner()
    db.start_session(a, "one")
    assert db.observe(a, "one", F, "original", "bug")
    assert db.prior(a).active[F] == pytest.approx(0.9)

    def submit(_):
        return MisconceptionTracer(path).observe(
            a, "one", F, "response", "correct", kind="practice", completed=True
        )

    with ThreadPoolExecutor(2) as executor:
        assert sorted(executor.map(submit, range(2))) == [False, True]
    assert db.history(a)[1]["after"] == pytest.approx(0.5)
    assert db.prior(a).active[F] == pytest.approx(0.35)
    db.start_session(a, "two")
    db.start_session(a, "two")
    db.start_session(a, "one")
    assert len(db.history(a)) == 8
    assert MisconceptionTracer(path).prior(a).active[F] == pytest.approx(0.3825)
    assert db.prior(a).active["binomial_expand"] == 0.525
    assert set(db.prior(b).active.values()) == {0.5}
    with pytest.raises(ValueError):
        db.clear_learner(a)
    db.clear_learner(a, confirmed=True)
    assert not db.sessions(a) and not db.history(a) and not db.prior(a).active
    assert len(db.prior(b).active) == 5


@pytest.mark.parametrize("bad", [-1, 1.01, float("nan"), float("inf"), True, None])
def test_invalid_parameters(bad):
    with pytest.raises(ValueError):
        Parameters(s_bug=bad)


def test_uninformative_and_impossible(tmp_path):
    db = MisconceptionTracer(tmp_path / "state", {F: Parameters(initial=1, g_bug=0, p_relapse=1)})
    learner = db.create_learner()
    db.start_session(learner, "s")
    for family, evidence in [(F, None), ("unknown", "correct"), (F, "unsupported")]:
        assert not db.observe(learner, "s", family, "skip", evidence)
    db.observe(learner, "s", F, "a", "correct", kind="practice", completed=True)
    db.observe(learner, "s", F, "b", "other", kind="practice", completed=True)
    assert [e["evidence"] for e in db.history(learner)] == ["other", "impossible"]
    assert db.prior(learner).active[F] == 1


def test_streamlit_memory_reruns_and_isolated_demo(tmp_path, monkeypatch):
    path = tmp_path / "memory"
    monkeypatch.setenv("BUGLENS_LEARNER_DB", str(path))
    ui = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")).run()

    def click(label):
        next(b for b in ui.button if b.label == label).click().run()
        assert not ui.exception

    ui.radio(key="page").set_value("Learner profile").run()
    click("Create anonymous profile")
    learner = ui.session_state["learner"]
    db = MisconceptionTracer(path)
    click("Load demonstration history")
    ui.run()
    assert not db.history(learner)
    ui.radio(key="page").set_value("Mathematical workspace").run()
    ui.text_area(key="input_attempt").set_value("3/7").run()
    click("Confirm question and attempt")
    # Inject the producer contract only in this temporary test database.
    ui.session_state["replay_outputs"] = CASES["Fraction error"][2]
    click("Open reasoning review")
    click("Continue to practice")
    ui.text_input(key="input_answer").set_value("9/10").run()
    ui.text_area(key="input_explanation").set_value("I used tenths").run()
    click("Save practice response")
    ui.run()
    assert [e["kind"] for e in db.history(learner)] == ["learn", "practice", "diagnostic"]
    assert len(db.sessions(learner)) == 1

    # Resume the same private code from a separate browser session.
    resumed = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")
    ).run()
    resumed.radio(key="page").set_value("Learner profile").run()
    resumed.text_input[0].set_value(learner).run()
    next(b for b in resumed.button if b.label == "Resume profile").click().run()
    resumed.run()
    assert len(db.sessions(learner)) == 2
    assert db.prior(learner).active[F] == pytest.approx(0.3825)
    assert next(b for b in resumed.button if b.label == "Clear learner data").disabled
    resumed.checkbox[0].check().run()
    next(b for b in resumed.button if b.label == "Clear learner data").click().run()
    assert not resumed.exception and not db.prior(learner).active
    ui.run()
    assert not ui.exception and "learner" not in ui.session_state
