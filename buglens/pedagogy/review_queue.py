"""Review once per later session within 30 days of successful targeted remediation."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
import re

from buglens.pedagogy.bkt_tracer import MisconceptionTracer
from buglens.pedagogy.remediation import FAMILIES, Example, worked


def classify(example: Example, answer: str) -> str:
    """Bounded exact numeric or canonical polynomial input. Never evaluate student text."""
    if re.search(r"\d\s+\d", answer):
        return "other"
    compact = answer.strip().lower().replace(" ", "").replace("−", "-")
    compact = compact.removeprefix("x=") if "=" in example.question else compact
    if not compact or len(compact) > 100 or example.value == example.wrong_value:
        return "other"
    for label, value in (("correct", example.value), ("bug", example.wrong_value)):
        if compact.replace("^", "**") == str(value).replace(" ", ""):
            return label
        if value.free_symbols:
            if compact == str(value).replace(" ", "").replace("**", "^").replace("*x", "x"):
                return label
        elif re.fullmatch(r"[+-]?\d{1,12}(?:/\d{1,12}|\.\d{1,12})?", compact):
            try:
                if Fraction(compact) == Fraction(int(value.p), int(value.q)):
                    return label
            except ZeroDivisionError:
                pass
    return "other"


@dataclass(frozen=True)
class ReviewItem:
    family: str
    example: Example


def review_queue(
    tracer: MisconceptionTracer, learner: str, session: str, *, now=None
) -> list[ReviewItem]:
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    with tracer.connection() as db:
        rows = db.execute(
            """
            SELECT s.family, count(r.id) AS reviews, coalesce(max(r.rowid),0) AS last_review
            FROM states s LEFT JOIN events r ON r.learner=s.learner AND r.family=s.family AND r.kind='review'
            WHERE s.learner=? AND s.active BETWEEN 0.15 AND 0.40
            AND EXISTS (SELECT 1 FROM events e JOIN sessions old ON old.learner=e.learner AND old.id=e.session
                JOIN sessions current ON current.learner=e.learner AND current.id=?
                WHERE e.learner=s.learner AND e.family=s.family AND e.kind='learn'
                AND e.after<0.5 AND e.created>=? AND old.rowid<current.rowid
                AND NOT EXISTS (SELECT 1 FROM events later WHERE later.learner=e.learner
                    AND later.family=e.family AND later.rowid>e.rowid AND later.evidence='bug'))
            GROUP BY s.family HAVING sum(CASE WHEN r.session=? THEN 1 ELSE 0 END)=0
            ORDER BY last_review,s.family
        """,
            (learner, session, cutoff, session),
        ).fetchall()
    families = {f.replay_id: f for f in FAMILIES}
    items = []
    for row in rows:
        family = families[row["family"]]
        prompt = family.practice[(1 + row["reviews"]) % len(family.practice)]
        example = worked(family, prompt)
        if example is None:
            raise ValueError("Unsupported review prompt")
        items.append(ReviewItem(family.replay_id, example))
    return items
