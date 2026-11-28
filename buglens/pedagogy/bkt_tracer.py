"""Independent inverse-BKT marginals. All parameter defaults are provisional."""

from collections.abc import Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from uuid import uuid4

from buglens.modeling.fusion import BUG_FAMILIES, bkt_inverse_polarity_update

DEFAULT_DB = Path("data/interim/learner_state.db")


@dataclass(frozen=True)
class Parameters:
    initial: float = 0.5
    s_bug: float = 0.1
    g_bug: float = 0.1
    p_learn: float = 0.3
    p_relapse: float = 0.05

    def __post_init__(self):
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1
            for v in vars(self).values()
        ):
            raise ValueError("BKT probabilities must be finite numbers between zero and one")


@dataclass(frozen=True)
class LearnerPrior:
    """Five Bernoulli marginals. Fusion needs a separate approved six-class mapping."""

    active: Mapping[str, float]


SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    id TEXT PRIMARY KEY NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions (
    learner TEXT NOT NULL REFERENCES learners ON DELETE CASCADE,
    id TEXT NOT NULL, started TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (learner,id)
);
CREATE TABLE IF NOT EXISTS states (
    learner TEXT NOT NULL REFERENCES learners ON DELETE CASCADE, family TEXT NOT NULL,
    active REAL NOT NULL CHECK(active BETWEEN 0 AND 1),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts>=0),
    PRIMARY KEY (learner,family)
);
CREATE TABLE IF NOT EXISTS events (
    learner TEXT NOT NULL, id TEXT NOT NULL, session TEXT NOT NULL, family TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('diagnostic','practice','learn','relapse','review')),
    evidence TEXT CHECK(evidence IN ('bug','correct','other','impossible')),
    before REAL NOT NULL CHECK(before BETWEEN 0 AND 1),
    after REAL NOT NULL CHECK(after BETWEEN 0 AND 1),
    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(learner,id),
    FOREIGN KEY(learner,session) REFERENCES sessions(learner,id) ON DELETE CASCADE,
    FOREIGN KEY(learner,family) REFERENCES states(learner,family) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS one_review ON events(learner,session,family) WHERE kind='review';
"""


class MisconceptionTracer:
    def __init__(self, path=DEFAULT_DB, parameters: Mapping[str, Parameters] | None = None):
        self.path = Path(path)
        self.parameters = dict.fromkeys(BUG_FAMILIES, Parameters())
        for family, params in (parameters or {}).items():
            if family not in self.parameters or not isinstance(params, Parameters):
                raise ValueError("Unsupported family or BKT parameters")
            self.parameters[family] = params
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connection(self, *, write=False):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db

    def create_learner(self) -> str:
        learner = uuid4().hex
        with self.connection(write=True) as db:
            db.execute("INSERT INTO learners(id) VALUES (?)", (learner,))
            db.executemany(
                "INSERT INTO states(learner,family,active) VALUES (?,?,?)",
                [(learner, f, p.initial) for f, p in self.parameters.items()],
            )
        return learner

    def start_session(self, learner: str, session: str):
        with self.connection(write=True) as db:
            previous = db.execute(
                "SELECT 1 FROM sessions WHERE learner=? LIMIT 1", (learner,)
            ).fetchone()
            inserted = db.execute(
                "INSERT INTO sessions(learner,id) VALUES (?,?) ON CONFLICT DO NOTHING",
                (learner, session),
            ).rowcount
            if previous and inserted:
                for family, params in self.parameters.items():
                    p = self._active(db, learner, family)
                    self._record(
                        db,
                        learner,
                        session,
                        family,
                        "relapse:" + session + ":" + family,
                        "relapse",
                        None,
                        p,
                        p + (1 - p) * params.p_relapse,
                    )

    def prior(self, learner: str) -> LearnerPrior:
        with self.connection() as db:
            return LearnerPrior(
                dict(db.execute("SELECT family,active FROM states WHERE learner=?", (learner,)))
            )

    @staticmethod
    def _active(db, learner, family):
        return db.execute(
            "SELECT active FROM states WHERE learner=? AND family=?", (learner, family)
        ).fetchone()[0]

    @staticmethod
    def _record(db, learner, session, family, event, kind, evidence, before, after):
        added = db.execute(
            """INSERT INTO events(learner,session,family,id,kind,evidence,before,after)
                             VALUES (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
            (learner, session, family, event, kind, evidence, before, after),
        ).rowcount
        if added:
            db.execute(
                "UPDATE states SET active=?, attempts=attempts+? WHERE learner=? AND family=?",
                (after, int(kind in ("diagnostic", "practice", "review")), learner, family),
            )
        return bool(added)

    def observe(
        self, learner, session, family, event, evidence, *, kind="diagnostic", completed=False
    ):
        """Record one classified attempt. Completed correct targeted practice also learns atomically."""
        if (
            family not in self.parameters
            or evidence not in ("bug", "correct", "other")
            or kind not in ("diagnostic", "practice", "review")
            or not event
        ):
            return False
        with self.connection(write=True) as db:
            p = self._active(db, learner, family)
            post = p
            params = self.parameters[family]
            if evidence != "other":
                # Pass the likelihood of the observed outcome directly. This preserves
                # small s_bug values that would vanish in 1 - (1 - s_bug).
                likelihoods = (
                    (params.g_bug, 1 - params.s_bug)
                    if evidence == "correct"
                    else (1 - params.g_bug, params.s_bug)
                )
                try:
                    post, _ = bkt_inverse_polarity_update(p, True, *likelihoods, 0, 0)
                except ZeroDivisionError:
                    evidence = "impossible"
            added = self._record(
                db, learner, session, family, kind + ":" + event, kind, evidence, p, post
            )
            if added and completed and kind == "practice" and evidence == "correct":
                self._record(
                    db,
                    learner,
                    session,
                    family,
                    "learn:" + event,
                    "learn",
                    None,
                    post,
                    post * (1 - params.p_learn),
                )
            return added

    def history(self, learner):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM events WHERE learner=? ORDER BY rowid DESC", (learner,)
                )
            ]

    def sessions(self, learner):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,started FROM sessions WHERE learner=? ORDER BY rowid DESC",
                    (learner,),
                )
            ]

    def clear_learner(self, learner, *, confirmed=False):
        if confirmed is not True:
            raise ValueError("Explicit confirmation required")
        with self.connection(write=True) as db:
            db.execute("DELETE FROM learners WHERE id=?", (learner,))
