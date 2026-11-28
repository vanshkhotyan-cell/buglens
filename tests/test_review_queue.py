from datetime import UTC, datetime, timedelta

import pytest

from buglens.pedagogy.bkt_tracer import MisconceptionTracer, Parameters
from buglens.pedagogy.remediation import FAMILIES, worked
from buglens.pedagogy.review_queue import classify, review_queue


@pytest.mark.parametrize(
    ("p", "due"), [(0.1499, False), (0.15, True), (0.4, True), (0.4001, False)]
)
def test_boundaries_and_resolution_history(tmp_path, p, due):
    f = "fraction_add"
    db = MisconceptionTracer(
        tmp_path / "q", {f: Parameters(initial=p, s_bug=0.5, g_bug=0.5, p_learn=0, p_relapse=0)}
    )
    learner = db.create_learner()
    db.start_session(learner, "first")
    db.start_session(learner, "second")
    assert not review_queue(db, learner, "second")
    db.observe(learner, "second", f, "done", "correct", kind="practice", completed=True)
    assert not review_queue(db, learner, "second")
    db.start_session(learner, "third")
    assert bool(review_queue(db, learner, "third")) == due
    assert not review_queue(db, learner, "third", now=datetime.now(UTC) + timedelta(days=31))


def test_interleaving_and_repeat_delivery(tmp_path):
    db = MisconceptionTracer(tmp_path / "q")
    learner = db.create_learner()
    db.start_session(learner, "a")
    for family in FAMILIES[:2]:
        db.observe(learner, "a", family.replay_id, family.id, "bug")
        db.observe(
            learner, "a", family.replay_id, family.id, "correct", kind="practice", completed=True
        )
    db.start_session(learner, "b")
    queue = review_queue(db, learner, "b")
    assert queue == review_queue(db, learner, "b") and len(queue) == 2
    assert db.observe(learner, "b", queue[0].family, "first", "other", kind="review")
    assert not db.observe(learner, "b", queue[0].family, "retry", "other", kind="review")
    assert review_queue(db, learner, "b") == queue[1:]


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        ("9/10", "correct"),
        ("0.90", "correct"),
        ("3/7", "bug"),
        ("1/0", "other"),
        ("9/1 0", "other"),
        ("unrelated", "other"),
    ],
)
def test_verified_answer(answer, outcome):
    example = worked(FAMILIES[0], FAMILIES[0].practice[0])
    assert classify(example, answer) == outcome
