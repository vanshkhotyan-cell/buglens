from __future__ import annotations

import re

import numpy as np
import pandas as pd

from buglens.modeling.fusion import ALL_CLASSES, CLASS_TO_IDX


def format_deberta_sequence(norm_q: str, norm_corr: str, norm_stud: str) -> str:
    """Step 2: Strictly serialize student attempt without leaking symbolic replay outputs."""
    return f"[CLS] Question: {norm_q.strip()} [SEP] Correct: {norm_corr.strip()} [SEP] Student: {norm_stud.strip()} [SEP]"


def apply_bug_plus_slip_perturbation(expr: str, rng: np.random.Generator) -> str:
    """Step 2: Apply a +/-1 arithmetic slip or dropped coefficient to ~15% of synthetic items."""
    tokens = list(re.finditer(r"\d+", expr))
    if not tokens:
        return expr

    choice = int(rng.integers(0, 2))
    if choice == 0 or len(tokens) == 1:
        match = tokens[int(rng.integers(0, len(tokens)))]
        val = int(match.group(0))
        delta = 1 if (val <= 1 or rng.random() < 0.5) else -1
        new_val = str(max(1, val + delta))
        return expr[: match.start()] + new_val + expr[match.end() :]

    coeff_match = re.search(r"(?<![0-9a-zA-Z])([2-9])([a-zA-Z])", expr)
    if coeff_match:
        return expr[: coeff_match.start(1)] + expr[coeff_match.end(1) :]

    match = tokens[0]
    val = int(match.group(0))
    return expr[: match.start()] + str(val + 1) + expr[match.end() :]


def augment_with_bug_plus_slip(
    df: pd.DataFrame,
    slip_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Step 2: Create ~15% noisy bug-plus-slip training rows while preserving true bug_id."""
    rng = np.random.default_rng(seed)
    eligible = df[df["bug_id"].isin(ALL_CLASSES[:5])].copy()
    if len(eligible) == 0:
        return df.copy()

    n_aug = max(1, int(round(len(eligible) * slip_fraction)))
    sampled = eligible.sample(n=min(n_aug, len(eligible)), random_state=seed).copy()
    sampled["StudentAnswerText"] = (
        sampled["StudentAnswerText"]
        .astype(str)
        .apply(lambda s: apply_bug_plus_slip_perturbation(s, rng))
    )
    sampled["source"] = "synthetic_bug_plus_slip"
    return pd.concat([df, sampled], ignore_index=True)


def compute_class_weights(labels: np.ndarray, num_classes: int = len(ALL_CLASSES)) -> np.ndarray:
    """Inverse-frequency class weights for imbalanced 6-class Cross-Entropy."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return weights / np.mean(weights)


def prepare_training_records(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Serialize DataFrame rows into DeBERTa input strings and integer labels [0..5]."""
    texts = [
        format_deberta_sequence(
            str(r["QuestionText"]),
            str(r["CorrectAnswerText"]),
            str(r["StudentAnswerText"]),
        )
        for _, r in df.iterrows()
    ]
    labels = np.array([CLASS_TO_IDX[str(b)] for b in df["bug_id"]], dtype=int)
    return texts, labels
