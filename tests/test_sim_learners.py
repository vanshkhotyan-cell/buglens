import json
from pathlib import Path

from experiments import sim_learners as sim
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cohort": 0},
        {"sessions": -1},
        {"seeds": []},
        {"budget": 0},
        {"seeds": [1, 1]},
        {"scenarios": {}},
    ],
)
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        sim.Config(**kwargs)


def test_validation():
    with pytest.raises(ValueError):
        sim.Population(learn=float("nan"))


def test_information_and_parameter_isolation(monkeypatch):
    cfg = sim.Config(cohort=1, sessions=2)
    actions = []
    original = sim.Targeted.observe

    def record(self, slot, i, kind, correct):
        # The boundary accepts only the scheduled item and one observed bit.
        assert self.params == cfg.tutor
        assert not any(isinstance(v, sim.Population) for v in vars(self).values())
        actions.append((slot, i, kind, correct))
        return original(self, slot, i, kind, correct)

    monkeypatch.setattr(sim.Targeted, "observe", record)
    pop = sim.Population(initial=1, learn=0, slip=0.5, accidental_correct=0.5)
    sim.participant(cfg, pop, 1, 0, sim.GROUPS[0])
    first = list(actions)
    actions.clear()
    sim.participant(
        cfg,
        sim.Population(initial=0, learn=1, slip=0.5, accidental_correct=0.5),
        1,
        0,
        sim.GROUPS[0],
    )
    assert first == actions
    assert sim.binary_gain(0.5, sim.Parameters(g_bug=0.5, s_bug=0.5)) == pytest.approx(0)


def test_metrics_and_censoring():
    cfg = sim.Config(sessions=2, budget=1)
    pop = sim.Population(initial=1, learn=1, relapse=0, accidental_correct=0, slip=0)
    rows = [sim.participant(cfg, pop, 42, 0, g) for g in sim.GROUPS]
    for row, count in zip(rows, (2, 2, 0), strict=True):
        assert sum(row[k] for k in ("practice", "diagnostic", "review")) == count
        assert row["pre"] == 0
    control = sim.summarize(pd.DataFrame([rows[-1]]), 2)
    assert control["curve"] == [1, 1, 1] and control["censored"] == 5
    assert control["median_resolution_session"] is None
    assert control["restricted_mean_sessions"] == 2 and control["recurrence"] == 1
    assert sim.gain(1, 0) is None and sim.gain(0.5, 0.25) == -0.5
    assert sim.gain(0, 1) == 1
    assert rows[0]["censored"] == 5 and rows[1]["censored"] < 5
    assert set(sim.assessment_form(0)).isdisjoint(sim.assessment_form(1))


def test_artifacts_reproducible_and_isolated(tmp_path, monkeypatch):
    cfg = sim.Config(
        cohort=3,
        sessions=2,
        seeds=(15,),
        scenarios={"perfect": sim.Population(initial=0, slip=0, relapse=0)},
    )
    connect = sim.sqlite3.connect

    def memory_only(path):
        assert path == ":memory:"
        return connect(path)

    monkeypatch.setattr(sim.sqlite3, "connect", memory_only)
    out = tmp_path / "first"
    sim.run(cfg, out)
    second = tmp_path / "second"
    sim.run(sim.Config.load(out / "config.json"), second)
    assert {p.name for p in out.iterdir()} == {
        "config.json",
        "summary.json",
        "aggregate.csv",
        "learners.csv",
        "sensitivity.csv",
        "gain_mean.png",
        "recurrence.png",
        "curve.png",
    }
    for p in out.iterdir():
        assert p.read_bytes() == (second / p.name).read_bytes()
    for row in json.loads((out / "summary.json").read_text())["groups"]:
        assert row["gain_mean"] is None and row["recurrence"] is None
        assert row["ci95"]["gain_mean"] == [None, None] and row["initial"] == 0


def test_dashboard(monkeypatch, tmp_path):
    app_path = Path(sim.ROOT / "app" / "streamlit_app.py")
    app = AppTest.from_file(str(app_path)).run()
    app.radio(key="page").set_value("Research analytics").run()
    assert not app.exception and len(app.dataframe[0].value) == 3
    assert len(app.image) == 3
    app.selectbox[0].select_index(1).run()
    assert not app.exception
    original = Path.read_text

    def missing(self, *args, **kwargs):
        if self.name == "summary.json":
            raise FileNotFoundError
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", missing)
    app.run()
    assert not app.exception and "unavailable" in app.info[0].value
